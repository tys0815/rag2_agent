"""FunctionCallAgent - 使用OpenAI函数调用范式的Agent实现"""

from __future__ import annotations

import json
import logging
from typing import Iterator, Optional, Union, TYPE_CHECKING, Any, Dict, Callable

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _map_parameter_type(param_type: str) -> str:
    """将工具参数类型映射为JSON Schema允许的类型"""
    normalized = (param_type or "").lower()
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    return "string"


class FunctionCallAgent(Agent):
    """基于OpenAI原生函数调用机制的Agent"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        enable_tool_calling: bool = True,
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 3,
        tool_call_listener: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations
        self._tool_call_listener = tool_call_listener

    def _get_system_prompt(self) -> str:
        """构建系统提示词，注入工具描述"""
        base_prompt = self.system_prompt or "你是一个可靠的AI助理，能够在需要时调用工具完成任务。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        prompt = base_prompt + "\n\n## 可用工具\n"
        prompt += "当你判断需要外部信息或执行动作时，可以直接通过函数调用使用以下工具：\n"
        prompt += tools_description + "\n"
        prompt += "\n请主动决定是否调用工具，合理利用多次调用来获得完备答案。"
        return prompt

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        if not self.enable_tool_calling or not self.tool_registry:
            return []

        schemas: list[dict[str, Any]] = []

        # Tool对象
        for tool in self.tool_registry.get_all_tools():
            properties: Dict[str, Any] = {}
            required: list[str] = []

            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            for param in parameters:
                properties[param.name] = {
                    "type": _map_parameter_type(param.type),
                    "description": param.description or ""
                }
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                if getattr(param, "required", True):
                    required.append(param.name)

            schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        # register_function 注册的工具（直接访问内部结构）
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "输入文本"
                                }
                            },
                            "required": ["input"]
                        }
                    }
                }
            )

        return schemas

    @staticmethod
    def _extract_message_content(raw_content: Any) -> str:
        """从OpenAI响应的message.content中安全提取文本"""
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(raw_content)

    @staticmethod
    def _parse_function_call_arguments(arguments: Optional[str]) -> dict[str, Any]:
        """解析模型返回的JSON字符串参数"""
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _convert_parameter_types(self, tool_name: str, param_dict: dict[str, Any]) -> dict[str, Any]:
        """根据工具定义尽可能转换参数类型"""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        type_mapping = {param.name: param.type for param in tool_params}
        converted: dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """执行工具调用并返回字符串结果"""
        logger.info(f"[TOOL EXECUTION] 开始执行工具调用 - 工具名称: {tool_name}, 参数数量: {len(arguments)}")
        if not self.tool_registry:
            logger.error("[TOOL EXECUTION] 错误：未配置工具注册表")
            return "❌ 错误：未配置工具注册表"

        result = ""
        parsed_arguments = arguments.copy()
        success = False

        try:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                logger.info(f"[TOOL EXECUTION] 找到Tool对象: {tool_name}")
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                logger.debug(f"[TOOL EXECUTION] 参数类型转换完成 - 原始参数: {arguments}, 转换后: {typed_arguments}")
                result = tool.run(typed_arguments)
                parsed_arguments = typed_arguments
                success = "❌" not in result and "错误：" not in result and "失败" not in result
                logger.info(f"[TOOL EXECUTION] Tool执行完成 - 成功: {success}, 结果长度: {len(result)}")
            else:
                func = self.tool_registry.get_function(tool_name)
                if func:
                    logger.info(f"[TOOL EXECUTION] 找到函数工具: {tool_name}")
                    input_text = arguments.get("input", "")
                    result = func(input_text)
                    success = "❌" not in result and "错误：" not in result and "失败" not in result
                    logger.info(f"[TOOL EXECUTION] 函数工具执行完成 - 成功: {success}, 结果长度: {len(result)}")
                else:
                    logger.error(f"[TOOL EXECUTION] 未找到工具: {tool_name}")
                    result = f"❌ 错误：未找到工具 '{tool_name}'"
        except Exception as exc:
            logger.error(f"[TOOL EXECUTION] 工具调用异常 - 工具: {tool_name}, 异常: {exc}")
            result = f"❌ 工具调用失败：{exc}"
            success = False

        # 通知监听器
        if self._tool_call_listener:
            try:
                logger.debug("[TOOL EXECUTION] 调用工具调用监听器")
                self._tool_call_listener({
                    "agent_name": self.name,
                    "tool_name": tool_name,
                    "raw_arguments": arguments,  # 原始参数字典
                    "parsed_arguments": parsed_arguments,  # 转换后的参数
                    "result": result,
                    "success": success
                })
            except Exception as e:
                logger.warning(f"工具调用监听器失败: {e}")

        logger.info(f"[TOOL EXECUTION] 工具调用执行结束 - 工具: {tool_name}, 成功: {success}")
        return result

    def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """调用底层OpenAI客户端执行函数调用"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("HelloAgentsLLM 未正确初始化客户端，无法执行函数调用。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> str:
        """
        执行函数调用范式的对话流程
        """
        logger.info(f"[FUNCTION CALL AGENT] 开始执行 - 输入长度: {len(input_text)}")
        messages: list[dict[str, Any]] = []
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        logger.info(f"[FUNCTION CALL AGENT] 系统提示词长度: {len(system_prompt)}")

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        tool_schemas = self._build_tool_schemas()
        logger.info(f"[FUNCTION CALL AGENT] 构建工具模式 - 可用工具数量: {len(tool_schemas)}")
        if not tool_schemas:
            logger.info("[FUNCTION CALL AGENT] 无可用工具，直接调用LLM")
            response_text = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response_text, "assistant"))
            logger.info(f"[FUNCTION CALL AGENT] 直接LLM响应长度: {len(response_text)}")
            return response_text

        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice

        logger.info(f"[FUNCTION CALL AGENT] 迭代设置 - 最大迭代次数: {iterations_limit}, 工具选择策略: {effective_tool_choice}")

        current_iteration = 0
        final_response = ""

        logger.info(f"[FUNCTION CALL AGENT] 开始工具调用迭代循环 (当前迭代: {current_iteration}/{iterations_limit})")
        while current_iteration < iterations_limit:
            logger.info(f"[FUNCTION CALL AGENT] 迭代 {current_iteration + 1}/{iterations_limit} - 调用LLM进行工具选择")
            response = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice,
                **kwargs,
            )

            choice = response.choices[0]
            assistant_message = choice.message
            content = self._extract_message_content(assistant_message.content)
            tool_calls = list(assistant_message.tool_calls or [])
            logger.info(f"[FUNCTION CALL AGENT] LLM响应 - 内容长度: {len(content)}, 工具调用数量: {len(tool_calls)}")

            if tool_calls:
                logger.info(f"[FUNCTION CALL AGENT] 检测到工具调用 - 将执行 {len(tool_calls)} 个工具调用")
                assistant_payload: dict[str, Any] = {"role": "assistant", "content": content}
                assistant_payload["tool_calls"] = []

                for tool_call in tool_calls:
                    assistant_payload["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
                messages.append(assistant_payload)

                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.function.name
                    arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                    logger.info(f"[FUNCTION CALL AGENT] 执行工具调用 {i+1}/{len(tool_calls)}: {tool_name}, 参数长度: {len(str(arguments))}")
                    result = self._execute_tool_call(tool_name, arguments)
                    logger.info(f"[FUNCTION CALL AGENT] 工具调用 {tool_name} 完成 - 结果长度: {len(result)}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                        }
                    )

                current_iteration += 1
                logger.info(f"[FUNCTION CALL AGENT] 迭代 {current_iteration}/{iterations_limit} 完成，继续下一轮迭代")
                continue

            final_response = content
            logger.info(f"[FUNCTION CALL AGENT] 无工具调用，获取最终响应 - 长度: {len(final_response)}")
            messages.append({"role": "assistant", "content": final_response})
            break

        if current_iteration >= iterations_limit and not final_response:
            logger.info(f"[FUNCTION CALL AGENT] 达到最大迭代次数 ({iterations_limit}) 但无最终响应，强制调用LLM生成最终答案")
            final_choice = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice="none",
                **kwargs,
            )
            final_response = self._extract_message_content(final_choice.choices[0].message.content)
            logger.info(f"[FUNCTION CALL AGENT] 强制生成的最终响应长度: {len(final_response)}")
            messages.append({"role": "assistant", "content": final_response})

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        logger.info(f"[FUNCTION CALL AGENT] 执行完成 - 总迭代次数: {current_iteration}, 最终响应长度: {len(final_response)}")
        return final_response

    def add_tool(self, tool) -> None:
        """便捷方法：将工具注册到当前Agent"""
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        if hasattr(tool, "auto_expand") and getattr(tool, "auto_expand"):
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for expanded_tool in expanded_tools:
                    self.tool_registry.register_tool(expanded_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return

        self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False

    def list_tools(self) -> list[str]:
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式调用暂未实现，直接回退到一次性调用"""
        result = self.run(input_text, **kwargs)
        yield result
