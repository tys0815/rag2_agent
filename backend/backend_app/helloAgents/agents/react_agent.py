"""
Hybrid ReAct Agent - 企业级架构实现
结合 ReAct 的思维链 (CoT) 与 Function Calling 的结构化执行
+ 四层自动记忆 + 长期记忆自动注入（真正个性化）
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any

from helloAgents.tools.builtin.memory_tool import MemoryTool
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..tools.registry import ToolRegistry
from redis_config import get_redis, QUEUE_MEMORY
redis = get_redis()

# 默认 Hybrid ReAct 提示词模板（企业级干净版）
DEFAULT_HYBRID_REACT_PROMPT = """你是一个具备深度推理和行动能力的企业级 AI 助手。

## 可用工具
{tools}

## 工作流程（严格遵守）
1. **思考**：在 content 中自然写出你的分析、推理、判断。
   - 不需要任何前缀，不要写 Thought、Action、Finish。

2. **行动**：需要外部信息时，使用 tool_calls 调用工具。
   - 不要在文本里模拟工具调用。

3. **结束**：信息足够回答用户时，
   - 直接输出最终答案
   - 不输出 tool_calls
   - 不再写思考过程

## 重要规则
- 每次必须先思考再决定是否调用工具。
- 无 tool_calls 代表直接回答。
- 最终回答必须清晰、直接、干净。

## 用户的记忆
{memory}

## 当前任务
**Question:** {question}

## 执行历史
{history}

请根据以上信息进行推理并回应："""


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

        self.tool_registry = tool_registry if tool_registry else ToolRegistry()
        self.max_steps = max_steps
        self.prompt_template = custom_prompt or DEFAULT_HYBRID_REACT_PROMPT
        self._tool_schemas = self._build_tool_schemas()

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """构建 OpenAI 兼容的工具 Schema（memory 工具对 LLM 隐藏）"""
        if not self.tool_registry:
            return []

        schemas = []
        for tool in self.tool_registry.get_all_tools():
            if tool.name == "memory":
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

            schemas.append({
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
            })

        # 兼容函数注册表中的裸函数
        func_map = getattr(self.tool_registry, "_functions", {})
        for name, info in func_map.items():
            if not any(s["function"]["name"] == name for s in schemas):
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string", "description": "输入文本"}},
                            "required": ["input"]
                        }
                    }
                })
        return schemas

    def add_tool(self, tool):
        self.tool_registry.register_tool(tool)
        self._tool_schemas = self._build_tool_schemas()

    def _convert_params(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return args

        try:
            params = tool.get_parameters()
            type_map = {p.name: p.type.lower() for p in params}
            converted = {}
            for k, v in args.items():
                t = type_map.get(k, "string")
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
    # 个性化核心：每轮自动加载长期记忆并注入 Prompt
    # =========================================================================
    async def _get_user_long_term_memory(self, input_text: str, **kwargs) -> str:
        try:
            memory_tool: MemoryTool = self.tool_registry.get_tool("memory")
            if not memory_tool:
                return "无用户记忆"

            task = {
                "tool_name": "memory",
                "input_data": {
                    "action": "search",
                    "query": input_text,
                    "memory_types": ["working", "semantic", "episodic"],
                    "user_id": kwargs.get("user_id"),
                    "session_id": kwargs.get("session_id"),
                    "limit": 8
                }
            }

            from ..tools.async_executor import run_parallel_tools
            results = await run_parallel_tools(self.tool_registry, [task])
            memory_result = results[0]["result"] if results else ""
            return f"\n{memory_result}" if memory_result else "无用户记忆"
        except Exception:
            return "无用户记忆"

    # =========================================================================
    # 四层记忆自动保存
    # =========================================================================
    def _auto_save_all_memories(self, user_input: str, final_answer: str, system_prompt: str, **kwargs):
        # 工作记忆
        work_tasks = {
            "action": "add",
            "user_content": user_input,
            "assistant_content": final_answer,
            "memory_type": "working",
            "user_id": kwargs.get("user_id"),
            "session_id": kwargs.get("session_id"),
            "importance": 0.3
        }
        redis.lpush(QUEUE_MEMORY, json.dumps(work_tasks))

        # 语义记忆
        semantic_tasks = {
            "action": "add",
            "content": user_input,
            "memory_type": "semantic",
            "user_id": kwargs.get("user_id"),
            "session_id": None,
            "importance": 0.8
        }
        redis.lpush(QUEUE_MEMORY, json.dumps(semantic_tasks))

        # 事件记忆
        episodic_tasks = {
            "action": "add",
            "role": "user",
            "content": user_input,
            "memory_type": "episodic",
            "user_id": kwargs.get("user_id"),
            "session_id": kwargs.get("session_id"),
            "importance": 0.5
        }
        redis.lpush(QUEUE_MEMORY, json.dumps(episodic_tasks))

        episodic_answer_tasks = {
            "action": "add",
            "role": "assistant",
            "content": final_answer,
            "memory_type": "episodic",
            "user_id": kwargs.get("user_id"),
            "session_id": kwargs.get("session_id"),
            "importance": 0.5
        }
        redis.lpush(QUEUE_MEMORY, json.dumps(episodic_answer_tasks))

        episodic_observation_tasks = {
            "action": "add",
            "role": "observation",
            "content": system_prompt,
            "memory_type": "episodic",
            "user_id": kwargs.get("user_id"),
            "session_id": kwargs.get("session_id"),
            "importance": 0.5
        }
        redis.lpush(QUEUE_MEMORY, json.dumps(episodic_observation_tasks))

        summary = self._extract_summary_by_llm(user_input, final_answer, **kwargs)
        if summary:
            summary_tasks = {
                "action": "add",
                "role": "summary",
                "content": summary,
                "memory_type": "episodic",
                "user_id": kwargs.get("user_id"),
                "session_id": kwargs.get("session_id"),
                "importance": 0.5
            }
            redis.lpush(QUEUE_MEMORY, json.dumps(summary_tasks))

    def _extract_summary_by_llm(self, user_input: str, final_answer: str, **kwargs) -> str:
        """
        让 LLM 根据用户问题 + 助手回复 提取核心摘要，用于精简记忆存储
        """
        try:
            # 构造简洁的摘要提取提示词
            prompt = f"""
            请你对以下【用户问题】和【助手回复】进行精简摘要提取：
            要求：
            1. 只保留核心信息，不超过50字
            2. 语言简洁、客观、通顺
            3. 直接输出摘要，不要额外解释
            
            用户问题：{user_input}
            助手回复：{final_answer}
            摘要：
            """
            
            # 调用你的 LLM 接口（根据你项目的实际 LLM 调用方式修改）
            # 这里是通用示例，你可以替换成项目真实调用方法
            messages = [{"role": "user", "content": prompt}]
            summary = self.llm.invoke(messages)

            # 清理空白字符
            summary = summary.strip()
            return summary if summary else ""
        
        except Exception as e:
            print(f"LLM 提取摘要失败：{str(e)}")
            return ""

    # =========================================================================
    # ReAct 主运行流程（企业级最终版）
    # =========================================================================
    async def run(self, input_text: str, **kwargs) -> str:
        current_step = 0
        final_answer = ""
        react_trace = []  # 仅用于构建执行历史，不发给 LLM ✅
        system_prompt = ""
        print(f"\n🤖 {self.name} 开始处理：{input_text}")
        memory = await self._get_user_long_term_memory(input_text, **kwargs)
        tools_desc = self.tool_registry.get_tools_description() 
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 核心：所有思考轨迹 → 放入 ## 执行历史
            history_str = self._build_react_history_str(react_trace)
            system_prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str,
                memory=memory
            )

            # LLM 永远只收到这一条消息 ✅ 干净稳定
            llm_messages = [{"role": "user", "content": system_prompt}]

            try:
                client = self.llm._client
                response = client.chat.completions.create(
                    model=self.llm.model,
                    messages=llm_messages,
                    tools=self._tool_schemas or None,
                    tool_choice="auto" if self._tool_schemas else None,
                    temperature=self.llm.temperature,
                    max_tokens=self.llm.max_tokens
                )
                assistant_message = response.choices[0].message
                content = assistant_message.content or ""
                tool_calls = assistant_message.tool_calls or []
            except Exception as e:
                print(f"❌ LLM 调用失败：{e}")
                assistant_entry = {
                    "role": "assistant",
                    "content": f"LLM 调用失败：{str(e)}"
                }
                react_trace.append(assistant_entry)
                history_str = self._build_react_history_str(react_trace)
                system_prompt = self.prompt_template.format(
                    tools=tools_desc,
                    question=input_text,
                    history=history_str,
                    memory=memory
                )
                break

            # 把思考加入轨迹（用于下一轮执行历史）
            # 把思考 + 工具调用 一起加入轨迹（关键修复）
            assistant_entry = {
                "role": "assistant",
                "content": content
            }

            # 把 tool_calls 也存进去！
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ]

            react_trace.append(assistant_entry)

            # 无工具调用 = 结束
            if not tool_calls:
                final_answer = content
                history_str = self._build_react_history_str(react_trace)
                system_prompt = self.prompt_template.format(
                    tools=tools_desc,
                    question=input_text,
                    history=history_str,
                    memory=memory
                )
                break

            # 并行执行工具
            parallel_tasks = []
            tool_map = {}
            for idx, tc in enumerate(tool_calls):
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                if func_name != "send_qq_email":
                    args["user_id"] = kwargs.get("user_id")
                    args["session_id"] = kwargs.get("session_id")
                args = self._convert_params(func_name, args)

                parallel_tasks.append({"tool_name": func_name, "input_data": args})
                tool_map[idx] = tc

            from ..tools.async_executor import run_parallel_tools
            results = await run_parallel_tools(self.tool_registry, parallel_tasks, max_workers=4)

            # 工具结果加入轨迹（同时携带参数）
            for idx, res in enumerate(results):
                tc = tool_map[idx]
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    args = {}
                react_trace.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": res["tool_name"],
                    "arguments": tc.function.arguments,  # 👈 新增：把参数存在这里
                    "content": str(res["result"])
                })

        if not final_answer:
            final_answer = "抱歉，无法在限定步数内完成任务。"
            assistant_entry = {
                "role": "assistant",
                "content": final_answer
            }
            react_trace.append(assistant_entry)
            history_str = self._build_react_history_str(react_trace)
            system_prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str,
                memory=memory
            )

        # 异步保存记忆
        try:
           self._auto_save_all_memories(input_text, final_answer, system_prompt, **kwargs)
        except Exception:
            pass

        return final_answer

    # =========================================================================
    # 构建 ReAct 执行历史（企业级可审计）
    # =========================================================================
    def _build_react_history_str(self, react_trace: List[Dict[str, Any]]) -> str:
        lines = []
        step = 1
        for msg in react_trace:
            content = msg.get("content", "").strip()

            if msg["role"] == "assistant":
                # 思考过程
                if content:
                    lines.append(f"Step {step} - Thought: {content}")
                    step += 1

                # 如果有工具调用，在这里一起显示
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    func_name = func.get("name", "unknown_tool")
                    arguments = func.get("arguments", "{}")
                    # 格式化显示：工具名 + 参数
                    lines.append(f"Step {step} - Action: 调用工具 [{func_name}]，参数：{arguments}")
                    step += 1

            elif msg["role"] == "tool":
                # 工具返回结果
                tool_name = msg.get("name", "unknown_tool")
                arguments = msg.get("arguments", "{}")
                if content:
                    lines.append(f"Step {step} - Observation: 工具 [{tool_name}] 参数：{arguments} 返回：{content}")
                    step += 1

        return "\n".join(lines) if lines else "无历史记录"