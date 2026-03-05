"""企业级多用户RAG助手
集成RAG工具和记忆工具，支持四层隔离：
- 用户级隔离 (user_id): 独立知识库
- 功能级隔离 (agent_id): 独立工作记忆和情景记忆
- 会话级隔离 (session_id): 独立工作记忆
- 记忆类型隔离: 工作记忆、情景记忆、语义记忆、感知记忆
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry, global_registry
from ..tools.builtin.rag_tool import RAGTool
from ..tools.builtin.memory_tool import MemoryTool


class EnterpriseRagAgent(Agent):
    """企业级多用户RAG助手

    特点：
    1. 多用户隔离：基于user_id隔离知识库和记忆
    2. 多功能隔离：基于agent_id隔离不同助手功能
    3. 多会话隔离：基于session_id隔离会话上下文
    4. 记忆集成：自动记录对话，检索相关记忆
    5. 知识库集成：检索用户知识库内容
    6. 智能回答：结合记忆和知识库生成回答
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = None,
        default_agent_id: str = "rag_assistant"
    ):
        """
        初始化企业级RAG助手

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 工具注册表（默认使用全局注册表）
            default_agent_id: 默认助手ID
        """
        super().__init__(name, llm, system_prompt, config)

        # 使用全局工具注册表或传入的注册表
        self.tool_registry = tool_registry or global_registry

        # 默认助手ID
        self.default_agent_id = default_agent_id

        # 获取工具
        self.rag_tool: Optional[RAGTool] = None
        self.memory_tool: Optional[MemoryTool] = None

        if self.tool_registry:
            self.rag_tool = self.tool_registry.get_tool("rag")
            self.memory_tool = self.tool_registry.get_tool("memory")

    def run(
        self,
        input_text: str,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        enable_memory: bool = True,
        enable_rag: bool = True,
        max_context_length: int = 2000,
        **kwargs
    ) -> str:
        """
        运行企业级RAG助手

        Args:
            input_text: 用户输入
            user_id: 用户ID（必需）
            agent_id: 助手ID（默认为rag_assistant）
            session_id: 会话ID（可选，不传则自动生成）
            namespace: 知识库命名空间（默认为user_id）
            enable_memory: 是否启用记忆
            enable_rag: 是否启用RAG
            max_context_length: 最大上下文长度
            **kwargs: 其他参数

        Returns:
            Agent响应
        """
        # 参数处理
        agent_id = agent_id or self.default_agent_id
        session_id = session_id or self._generate_session_id()
        namespace = namespace or user_id

        print(f"🔍 企业级RAG助手启动: user={user_id}, agent={agent_id}, session={session_id}, namespace={namespace}")

        # 1. 记录用户输入到工作记忆
        if enable_memory and self.memory_tool:
            self._record_to_memory(
                content=f"用户: {input_text}",
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type="working"
            )

        # 2. 检索相关记忆
        memory_context = ""
        if enable_memory and self.memory_tool:
            memory_context = self._retrieve_relevant_memories(
                query=input_text,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                limit=5
            )

        # 3. 检索知识库内容
        rag_context = ""
        if enable_rag and self.rag_tool:
            rag_context = self._retrieve_rag_context(
                query=input_text,
                namespace=namespace,
                limit=5
            )

        # 4. 构建增强上下文
        enhanced_context = self._build_enhanced_context(
            user_input=input_text,
            memory_context=memory_context,
            rag_context=rag_context,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id
        )

        # 5. 构建消息列表
        messages = self._build_messages(
            input_text=input_text,
            enhanced_context=enhanced_context,
            max_context_length=max_context_length
        )

        # 6. 调用LLM生成回答
        try:
            response = self.llm.invoke(messages, **kwargs)
        except Exception as e:
            response = f"❌ 生成回答时出错: {str(e)}"

        # 7. 记录助手回答到工作记忆
        if enable_memory and self.memory_tool:
            self._record_to_memory(
                content=f"助手: {response}",
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type="working"
            )

        # 8. 保存到Agent历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

        return response

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        import uuid
        return f"ses_{uuid.uuid4().hex[:16]}"

    def _record_to_memory(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        memory_type: str = "working",
        importance: float = 0.5
    ):
        """记录内容到记忆"""
        if not self.memory_tool:
            return

        try:
            self.memory_tool.run({
                "action": "add",
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id
            })
        except Exception as e:
            print(f"⚠️ 记录到记忆失败: {e}")

    def _retrieve_relevant_memories(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        limit: int = 5
    ) -> str:
        """检索相关记忆"""
        if not self.memory_tool:
            return ""

        try:
            result = self.memory_tool.run({
                "action": "search",
                "query": query,
                "limit": limit,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id
            })

            # 解析记忆结果
            if "未找到" in result or "❌" in result:
                return ""

            return result
        except Exception as e:
            print(f"⚠️ 检索记忆失败: {e}")
            return ""

    def _retrieve_rag_context(
        self,
        query: str,
        namespace: str,
        limit: int = 5
    ) -> str:
        """检索RAG上下文"""
        if not self.rag_tool:
            return ""

        try:
            result = self.rag_tool.run({
                "action": "search",
                "query": query,
                "limit": limit,
                "namespace": namespace
            })

            # 解析RAG结果
            if "未找到" in result or "❌" in result:
                return ""

            return result
        except Exception as e:
            print(f"⚠️ 检索RAG上下文失败: {e}")
            return ""

    def _build_enhanced_context(
        self,
        user_input: str,
        memory_context: str,
        rag_context: str,
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> str:
        """构建增强上下文"""
        context_parts = []

        # 系统信息
        context_parts.append(f"# 企业级RAG助手上下文")
        context_parts.append(f"- 用户: {user_id}")
        context_parts.append(f"- 助手: {agent_id}")
        context_parts.append(f"- 会话: {session_id}")
        context_parts.append(f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_parts.append("")

        # 用户问题
        context_parts.append(f"## 用户问题")
        context_parts.append(f"{user_input}")
        context_parts.append("")

        # 相关记忆
        if memory_context:
            context_parts.append(f"## 相关记忆")
            context_parts.append(f"{memory_context}")
            context_parts.append("")

        # 知识库内容
        if rag_context:
            context_parts.append(f"## 知识库相关内容")
            context_parts.append(f"{rag_context}")
            context_parts.append("")

        # 回答要求
        context_parts.append(f"## 回答要求")
        context_parts.append("1. 基于以上信息提供准确、有帮助的回答")
        context_parts.append("2. 如果信息不足，请说明需要什么额外信息")
        context_parts.append("3. 引用相关来源时注明出处")
        context_parts.append("4. 保持专业、友好的语气")

        return "\n".join(context_parts)

    def _build_messages(
        self,
        input_text: str,
        enhanced_context: str,
        max_context_length: int = 2000
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []

        # 系统提示词
        system_prompt = self.system_prompt or self._get_default_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 历史消息（限制长度）
        history_context = self._get_recent_history(max_context_length // 2)
        if history_context:
            messages.append({"role": "system", "content": f"## 对话历史\n{history_context}"})

        # 增强上下文
        if enhanced_context:
            # 截断上下文以避免超出token限制
            if len(enhanced_context) > max_context_length:
                enhanced_context = enhanced_context[:max_context_length] + "...[内容被截断]"
            messages.append({"role": "user", "content": enhanced_context})
        else:
            # 如果没有增强上下文，直接使用用户输入
            messages.append({"role": "user", "content": input_text})

        return messages

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个企业级RAG助手，具备以下能力：
1. 多用户支持：为不同用户提供个性化服务
2. 记忆集成：能够回忆之前的对话内容
3. 知识库检索：能够从用户知识库中查找相关信息
4. 专业回答：提供准确、全面、有帮助的回答

请根据提供的上下文信息回答问题，如果信息不足请诚实地说明。"""

    def _get_recent_history(self, max_length: int = 1000) -> str:
        """获取最近的对话历史"""
        if not self._history:
            return ""

        history_parts = []
        current_length = 0

        # 从新到旧遍历历史
        for msg in reversed(self._history):
            content = f"{msg.role}: {msg.content}"
            if current_length + len(content) > max_length:
                break
            history_parts.append(content)
            current_length += len(content)

        # 反转回时间顺序
        history_parts.reverse()
        return "\n".join(history_parts) if history_parts else ""

    def get_memory_stats(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取记忆统计"""
        if not self.memory_tool:
            return {"error": "记忆工具未初始化"}

        try:
            result = self.memory_tool.run({
                "action": "stats",
                "user_id": user_id,
                "agent_id": agent_id or self.default_agent_id,
                "session_id": session_id
            })
            return {"success": True, "stats": result}
        except Exception as e:
            return {"error": str(e)}

    def clear_memories(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """清空记忆"""
        if not self.memory_tool:
            return False

        try:
            result = self.memory_tool.run({
                "action": "clear_all",
                "user_id": user_id,
                "agent_id": agent_id or self.default_agent_id,
                "session_id": session_id
            })
            return "已清空" in result
        except Exception as e:
            print(f"⚠️ 清空记忆失败: {e}")
            return False

    def consolidate_important_memories(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        importance_threshold: float = 0.7
    ) -> int:
        """整合重要记忆到长期记忆"""
        if not self.memory_tool:
            return 0

        try:
            result = self.memory_tool.run({
                "action": "consolidate",
                "from_type": "working",
                "to_type": "episodic",
                "importance_threshold": importance_threshold,
                "user_id": user_id,
                "agent_id": agent_id or self.default_agent_id,
                "session_id": session_id
            })

            # 解析结果中的数量
            import re
            match = re.search(r'已整合 (\d+) 条', result)
            if match:
                return int(match.group(1))
            return 0
        except Exception as e:
            print(f"⚠️ 整合记忆失败: {e}")
            return 0