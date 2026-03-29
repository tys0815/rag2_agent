"""
Hybrid ReAct Agent - 企业级架构实现
结合 ReAct 的思维链 (CoT) 与 Function Calling 的结构化执行
+ 四层自动记忆 + 长期记忆自动注入（真正个性化）
"""

import asyncio
import json
import re
from typing import Optional, List, Dict, Any, Tuple

from backend_app.helloAgents.tools.builtin.memory_tool import MemoryTool
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry


# 默认 Hybrid ReAct 提示词模板
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

## 用户的记忆
{memory}

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动（请先输出 Thought）："""


class ReActAgent(Agent):
    """
    企业级 Hybrid ReAct Agent + 四层自动记忆 + 长期记忆自动注入
    架构：记忆完全基座化，对 LLM 透明，业务工具无耦合
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

        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry

        self.max_steps = max_steps
        self.conversation_history: List[Dict[str, Any]] = []
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_HYBRID_REACT_PROMPT
        self._tool_schemas = self._build_tool_schemas()

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """构建 OpenAI 兼容的工具 Schema（memory 工具对 LLM 隐藏）"""
        if not self.tool_registry:
            return []

        schemas = []
        for tool in self.tool_registry.get_all_tools():
            # ======================
            # ✅ 关键：memory 不暴露给 LLM
            # ======================
            if tool.name == "memory" or tool.name == "rag":
                continue

            properties = {}
            required = []
            try:
                params = tool.get_parameters()
                for p in params:
                    p_type = (p.type or "string").lower()
                    if p_type not in ["string", "number", "integer", "boolean", "array", "object"]:
                        p_type = "string"

                    properties[p.name] = {
                        "type": p_type,
                        "description": p.description or ""
                    }
                    if getattr(p, "required", True):
                        required.append(p.name)
            except Exception:
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

        func_map = getattr(self.tool_registry, "_functions", {})
        for name, info in func_map.items():
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
        self.tool_registry.register_tool(tool)
        self._tool_schemas = self._build_tool_schemas()

    def _get_system_prompt(self, question: str, history_str: str, memory: str) -> str:
        tools_desc = self.tool_registry.get_tools_description()
        return self.prompt_template.format(
            tools=tools_desc,
            question=question,
            history=history_str,
            memory=memory
        )

    def _convert_params(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
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

    # =========================================================================
    # ✅ 个性化核心：每轮自动加载长期记忆并注入 Prompt
    # =========================================================================
    async def _get_user_long_term_memory(self, input_text: str, **kwargs) -> str:
        try:
            memory_tool: MemoryTool = self.tool_registry.get_tool("memory")
            if not memory_tool:
                return "无用户记忆"

            user_id = kwargs.get("user_id", "default_user")
            session_id = kwargs.get("session_id")

            # ==============================
            # 🔥 一次查询：同时查 3 种记忆（高性能版）
            # ==============================
            task = {
                "tool_name": "memory",
                "input_data": {
                    "action": "search",
                    "query": input_text,
                    "memory_types": ["working", "semantic", "episodic"],  # 👈 多类型！
                    "user_id": user_id,
                    "session_id": session_id,
                    "limit": 8
                }
            }

            # ==============================
            # ✅ 只执行一次！
            # ==============================
            from ..tools.async_executor import run_parallel_tools
            results = await run_parallel_tools(self.tool_registry, [task])
            memory_result = results[0]["result"] if results else ""

            return f"\n{memory_result}" if memory_result else "无用户记忆"

        except Exception:
            return "无用户记忆"

    # =========================================================================
    # ✅ 四层记忆自动保存（唯一入口）
    # =========================================================================
    async def _auto_save_all_memories(self, user_input: str, final_answer: str, **kwargs):
        try:
            memory_tool: MemoryTool = self.tool_registry.get_tool("memory")
            if not memory_tool:
                return

            user_id = kwargs.get("user_id")
            session_id = kwargs.get("session_id")

            # 统一用你项目的 AsyncToolExecutor 执行并行保存
            from ..tools.async_executor import AsyncToolExecutor
            executor = AsyncToolExecutor(self.tool_registry, max_workers=3)

            # 构建并行保存任务
            tasks = []

            # ------------------------------
            # 任务1：保存 working 记忆
            # ------------------------------
            tasks.append({
                "tool_name": "memory",
                "input_data": {
                    "action": "add",
                    "content": f"用户：{user_input}",
                    "memory_type": "working",
                    "user_id": user_id,
                    "session_id": session_id,
                    "importance": 0.3
                }
            })
            tasks.append({
                "tool_name": "memory",
                "input_data": {
                    "action": "add",
                    "content": f"助手：{final_answer}",
                    "memory_type": "working",
                    "user_id": user_id,
                    "session_id": session_id,
                    "importance": 0.3
                }
            })

            # ------------------------------
            # 任务2+3：LLM一次抽取 + 保存 semantic + episodic
            # ------------------------------
            tasks.append({
                    "tool_name": "memory",
                    "input_data": {
                        "action": "add",
                        "content": user_input,
                        "memory_type": "semantic",
                        "user_id": user_id,
                        "session_id": None,
                        "importance": 0.8,  # 语义记忆重要性较高
                    }
                })
                

            tasks.append({
                    "tool_name": "memory",
                    "input_data": {
                        "action": "add",
                        "content": user_input,
                        "memory_type": "episodic",
                        "user_id": user_id,
                        "session_id": session_id,
                        "importance": 0.5,  # 事件记忆重要性较高
                    }
                })
            tasks.append({
                    "tool_name": "memory",
                    "input_data": {
                        "action": "add",
                        "content": final_answer,
                        "memory_type": "episodic",
                        "user_id": user_id,
                        "session_id": session_id,
                        "importance": 0.5,  # 事件记忆重要性较高
                    }
                })
                

            # ------------------------------
            # 🔥 真正异步并行保存（用你项目统一执行器）
            # ------------------------------
            if tasks:
                await executor.execute_tools_parallel(tasks)

        except Exception:
            import traceback
            traceback.print_exc()

    # =========================================================================
    # ReAct 主运行流程
    # =========================================================================
    async def run(self, input_text: str, **kwargs) -> str:
        self.conversation_history = []
        current_step = 0
        final_answer = ""
        messages = []

        print(f"\n🤖 {self.name} 开始处理问题：{input_text}")
        # ======================
        # ✅ 自动加载长期记忆
        # ======================
        long_memory = await self._get_user_long_term_memory(input_text, **kwargs)
        history_str = self._build_react_history_str(messages)
        system_prompt = self._get_system_prompt(input_text, history_str, long_memory)
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            print(f"\n--- 系统提示 ---\n{system_prompt} ---")

            if current_step == 1:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]

            try:
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
                    content = assistant_message.content or ""
                    tool_calls = assistant_message.tool_calls or []
                else:
                    raise RuntimeError("当前 LLM 不支持 Function Calling")
            except Exception as e:
                print(f"❌ LLM 调用失败：{e}")
                break

            if content:
                msg_payload = {"role": "assistant", "content": content}
                if tool_calls:
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

            if not tool_calls:
                final_answer = content
                break

            parallel_tasks = []
            tool_call_map = {}
            for idx, tc in enumerate(tool_calls):
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                args['user_id'] = kwargs.get("user_id")
                args['session_id'] = kwargs.get("session_id")
                # args['action'] = "search"  # 统一加个 action，方便工具区分查询/添加等操作
                # args['query'] = "入职流程"  # 兼容老版本工具参数
                typed_args = self._convert_params(func_name, args)
                
                parallel_tasks.append({
                    "tool_name": func_name,
                    "input_data": typed_args
                })
                tool_call_map[idx] = tc

            from ..tools.async_executor import run_parallel_tools
            parallel_results = await run_parallel_tools(
                registry=self.tool_registry,
                tasks=parallel_tasks,
                max_workers=4
            )

            for idx, result in enumerate(parallel_results):
                tc = tool_call_map[idx]
                func_name = result["tool_name"]
                observation = result["result"]
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": func_name,
                    "content": str(observation)
                })

        if not final_answer:
            final_answer = "抱歉，我无法在限定步数内完成任务。"
            print("⏰ 已达到最大步数")

        # ======================
        # ✅ 统一自动保存四层记忆
        # ======================
        try:
            # 创建后台任务
            task = asyncio.create_task(
                self._auto_save_all_memories(input_text, final_answer, **kwargs)
            )
            # 可选：不等待，但捕获任务内部异常
            def _done_callback(t):
                try:
                    task.result()
                except Exception as e:
                    print(f"⚠️ 后台记忆任务异常: {e}")

            task.add_done_callback(_done_callback)

        except Exception as e:
            print(f"⚠️ 创建记忆任务失败: {e}")  

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    # =========================================================================
    # 工具方法
    # =========================================================================
    def _build_react_history_str(self, messages: List[Dict[str, Any]]) -> str:
        lines = []
        step = 1
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "").strip()
            if not content:
                continue
            if role == "assistant":
                lines.append(f"Step {step} - Thought: {content}")
                step += 1
            elif role == "tool":
                lines.append(f"Step {step} - Observation: {content}")
                step += 1
        return "\n".join(lines) if lines else "无历史记录"

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        thought_match = re.search(r"Thought: (.*)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else text
        return thought, None

    def _parse_action(self, action_text: str, **kwargs) -> Tuple[Optional[str], Optional[str]]:
        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        return action_text