"""
Hybrid ReAct Agent - 企业级架构实现
结合 ReAct 的思维链 (CoT) 与 Function Calling 的结构化执行
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# 默认 Hybrid ReAct 提示词模板
# 核心逻辑：强制要求先输出 Thought，再决定是否调用工具
DEFAULT_HYBRID_REACT_PROMPT = """你是一个具备深度推理和行动能力的企业级 AI 助手。
你需要解决复杂问题，可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
你可以通过函数调用 (Function Calling) 使用以下工具：
{tools}

## 工作流程 (严格遵循)
请严格按照以下步骤进行回应：

1. **Thought (思考)**: 
   - 在每次回应开始时，你必须先在回复的文本内容 (content) 中详细分析当前情况。
   - 说明你已知什么，还需要什么信息，以及为什么决定调用某个工具（或者为什么决定直接回答）。
   - 即使你决定调用工具，也必须先写这段思考。

2. **Action (行动)**: 
   - 如果你需要外部信息，请直接使用系统提供的 **工具调用功能 (tool_calls)**。
   - 不要试图在文本内容中模拟工具调用格式（如 tool_name[args]），必须使用标准的函数调用结构。
   - 你可以一次性调用多个工具。

3. **Observation (观察)**: 
   - 系统将执行你的工具调用并返回结果。
   - 基于新的观察结果，重复上述“思考 -> 行动”过程。

4. **Finish (结束)**: 
   - 当你拥有足够信息回答问题时，**停止调用工具**。
   - 直接在回复的文本内容 (content) 中给出最终结论，不要再包含任何 tool_calls。

## 重要提醒
- 每次回应必须包含详细的思考过程 (Thought)。
- 只有当你确信有足够信息回答问题时，才停止调用工具并给出最终答案。
- 如果工具返回错误或信息不足，请在下一次思考中分析原因并调整策略。

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动（请先输出 Thought）："""

class ReActAgent(Agent):
    """
    企业级 Hybrid ReAct Agent
    
    架构特点：
    1. 使用 ReAct 范式引导 LLM 进行显式思维链 (Chain of Thought) 推理。
    2. 使用 Function Calling 协议进行结构化、高可靠性的工具执行。
    3. 支持并行工具调用、错误恢复和人机协作扩展点。
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        super().__init__(name, llm, system_prompt, config)

        # 初始化工具注册表
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry

        self.max_steps = max_steps
        # 存储标准消息格式 [{'role': 'user', 'content': ...}, {'role': 'assistant', ...}, {'role': 'tool', ...}]
        self.conversation_history: List[Dict[str, Any]] = [] 
        
        # 设置提示词模板
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_HYBRID_REACT_PROMPT
        
        # 预构建工具 Schema (优化性能，避免每次循环都构建)
        self._tool_schemas = self._build_tool_schemas()

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """构建 OpenAI 兼容的工具 Schema"""
        if not self.tool_registry:
            return []
        
        schemas = []
        
        # 1. 处理 Tool 对象
        for tool in self.tool_registry.get_all_tools():
            properties = {}
            required = []
            try:
                params = tool.get_parameters()
                for p in params:
                    p_type = (p.type or "string").lower()
                    # 映射类型到 JSON Schema 标准
                    if p_type not in ["string", "number", "integer", "boolean", "array", "object"]:
                        p_type = "string"
                    
                    properties[p.name] = {
                        "type": p_type,
                        "description": p.description or ""
                    }
                    if getattr(p, "required", True):
                        required.append(p.name)
            except Exception:
                # 如果无法获取参数，默认为单字符串输入
                properties["input"] = {"type": "string", "description": "输入内容"}
                required = ["input"]
            
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
            schemas.append(schema)
            
        # 2. 处理 register_function 注册的简单函数 (兼容性处理)
        func_map = getattr(self.tool_registry, "_functions", {})
        for name, info in func_map.items():
            # 避免重复添加
            if not any(s['function']['name'] == name for s in schemas):
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string", "description": "输入文本"}
                            },
                            "required": ["input"]
                        }
                    }
                })
            
        return schemas

    def add_tool(self, tool):
        """添加工具并重建 Schema"""
        # 处理 MCP 自动展开逻辑 (复用原有逻辑)
        if hasattr(tool, 'auto_expand') and tool.auto_expand:
            if hasattr(tool, '_available_tools') and tool._available_tools:
                for mcp_tool in tool._available_tools:
                    from ..tools.base import Tool as BaseTool
                    wrapped_tool = BaseTool(
                        name=f"{tool.name}_{mcp_tool['name']}",
                        description=mcp_tool.get('description', ''),
                        func=lambda input_text, t=tool, tn=mcp_tool['name']: t.run({
                            "action": "call_tool",
                            "tool_name": tn,
                            "arguments": {"input": input_text}
                        })
                    )
                    self.tool_registry.register_tool(wrapped_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(tool._available_tools)} 个独立工具")
                self._tool_schemas = self._build_tool_schemas()
                return
        else:
            self.tool_registry.register_tool(tool)
        
        # 重新构建 Schema
        self._tool_schemas = self._build_tool_schemas()

    def _get_system_prompt(self, question: str, history_str: str) -> str:
        tools_desc = self.tool_registry.get_tools_description()
        return self.prompt_template.format(
            tools=tools_desc,
            question=question,
            history=history_str
        )

    def _convert_params(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """根据工具定义转换参数类型 (增强鲁棒性)"""
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return args
        
        try:
            params = tool.get_parameters()
            type_map = {p.name: p.type for p in params}
            
            converted = {}
            for k, v in args.items():
                t = type_map.get(k, "string").lower()
                try:
                    if t in ["integer", "int"]:
                        converted[k] = int(v)
                    elif t in ["number", "float"]:
                        converted[k] = float(v)
                    elif t in ["boolean", "bool"]:
                        converted[k] = str(v).lower() in ["true", "1", "yes"]
                    else:
                        converted[k] = v
                except (ValueError, TypeError):
                    converted[k] = v
            return converted
        except Exception:
            return args

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Hybrid ReAct Agent
        """
        self.conversation_history = []
        current_step = 0
        
        print(f"\n🤖 {self.name} 开始处理问题：{input_text}")
        
        # 1. 初始化 System Prompt 和 User Message
        # 注意：在 Hybrid 模式下，我们将 System Prompt 动态注入到每一轮或第一轮，
        # 这里为了简化，我们在第一轮构建完整上下文，后续只追加历史
        system_prompt = self._get_system_prompt(input_text, "无历史记录")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        final_answer = ""
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            
            # 2. 调用 LLM (支持 Tools)
            try:
                # 检查 LLM 客户端是否支持 tools 参数
                # 假设 HelloAgentsLLM 有 invoke_with_tools 方法或者直接暴露 client
                # 这里模拟直接调用底层 client，实际需根据 HelloAgentsLLM 的实现调整
                
                if hasattr(self.llm, '_client') and self.llm._client:
                    client = self.llm._client
                    response = client.chat.completions.create(
                        model=self.llm.model,
                        messages=messages,
                        tools=self._tool_schemas if self._tool_schemas else None,
                        tool_choice="auto" if self._tool_schemas else None,
                        temperature=self.llm.temperature,
                        max_tokens=self.llm.max_tokens
                    )
                    
                    choice = response.choices[0]
                    assistant_message = choice.message
                    
                    # 提取内容 (Thought) 和 工具调用 (Action)
                    content = assistant_message.content or ""
                    tool_calls = assistant_message.tool_calls or []
                    
                else:
                    # 降级处理：如果不支持原生 tool_call，则回退到纯文本 ReAct (不推荐)
                    print("⚠️ 警告：LLM 客户端不支持原生 Function Calling，回退到文本解析模式。")
                    # 这里可以调用旧的解析逻辑，但为了保持架构纯净，我们抛出异常或简单处理
                    raise RuntimeError("当前 LLM 实例不支持 Function Calling，请使用支持的模型。")

            except Exception as e:
                print(f"❌ LLM 调用失败：{e}")
                break

            # 3. 处理 Thought (ReAct 的核心价值：可解释性)
            if content:
                print(f"🧠 [Thought]: {content.strip()}")
                # 将 Assistant 消息加入历史 (包含 content 和 tool_calls)
                msg_payload = {"role": "assistant", "content": content}
                if tool_calls:
                    # 格式化 tool_calls 以便存入历史 (不同 SDK 格式略有不同，需标准化)
                    msg_payload["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in tool_calls
                    ]
                messages.append(msg_payload)

            # 4. 判断是否调用工具 (Function Call 的核心价值：结构化执行)
            if not tool_calls:
                # 没有工具调用，且有内容 -> 判定为最终回答
                final_answer = content
                print(f"🎉 [Final Answer]: {final_answer}")
                break
            
            # 5. 执行工具调用 (支持并行)
            print(f"🛠️ 检测到 {len(tool_calls)} 个工具调用请求")
            
            for tc in tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                print(f"   -> 调用：{func_name}({args})")
                
                # [企业级扩展点]：在此处插入权限校验、审计日志、人工审批
                # if not self.security_check(func_name, args): ...
                
                # 参数类型转换
                typed_args = self._convert_params(func_name, args)
                
                # 执行工具
                try:
                    # 尝试作为 Tool 对象执行
                    tool_obj = self.tool_registry.get_tool(func_name)
                    if tool_obj:
                        #测试 需删除
                        typed_args = {
                            "user_id": kwargs.get("user_id"),
                            "action": "search",
                            "query": "入职流程"
                        }
                        observation = tool_obj.run(typed_args)
                    else:
                        # 尝试作为注册函数执行
                        func = self.tool_registry.get_function(func_name)
                        if func:
                            observation = func(typed_args.get("input", ""))
                        else:
                            observation = f"Error: Tool '{func_name}' not found."
                    
                    print(f"   <- 观察：{str(observation)[:200]}...")
                    
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
                    print(f"   <- 错误：{observation}")

                # 6. 将结果回传 (Observation)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": func_name,
                    "content": str(observation)
                })
            
            # 循环继续，LLM 将看到完整的 Thought -> Action -> Observation 链条
        
        # 7. 收尾
        if not final_answer:
            final_answer = "抱歉，我无法在限定步数内完成这个任务，或者遇到了错误。"
            print("⏰ 已达到最大步数，流程终止。")
        
        # 保存到 Agent 的历史记录 (用于多轮对话上下文)
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        兼容旧版 ReAct 的解析方法 (仅在降级模式下使用)
        在 Hybrid 模式下，主要依赖 tool_calls 对象，此方法仅用于提取 Thought 中的关键信息如果需要
        """
        thought_match = re.search(r"Thought: (.*)", text, re.DOTALL)
        # 在 Hybrid 模式下，Action 不再从文本解析，所以这里返回 None
        thought = thought_match.group(1).strip() if thought_match else text
        return thought, None

    def _parse_action(self, action_text: str, **kwargs) -> Tuple[Optional[str], Optional[str]]:
        """兼容旧版方法，Hybrid 模式下不再使用"""
        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        """兼容旧版方法"""
        return action_text