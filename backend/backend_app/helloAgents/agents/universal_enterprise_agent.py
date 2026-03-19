from typing import Optional, Dict, Any, List, Tuple
import logging
import numpy as np
from datetime import datetime

from .plan_solve_agent import PlanAndSolveAgent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..tools.registry import ToolRegistry
from ..context import ContextBuilder, ContextConfig
from ..memory.embedding import get_text_embedder

logger = logging.getLogger(__name__)


class KnowledgeBaseAssistant(PlanAndSolveAgent):
    """
    企业知识库助手（休假/考勤/制度/报销专用）
    基于【LLM意图判断】做检索路由，无关键词、无向量、纯大模型决策
    """

    # 已删除：正负样本、关键词、嵌入向量相关全部内容

    def __init__(
        self,
        name: str = "knowledge_base_assistant",
        llm: Optional[HelloAgentsLLM] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True,
        default_tool_choice: str = "auto",
        max_tool_iterations: int = 1,
        workspace: str = ".",
        tool_call_listener: Optional[Any] = None,
        retrieval_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            **kwargs
        )

        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations
        self.workspace = workspace
        self.tool_call_listener = tool_call_listener
        self.llm_client = llm

        # 已删除：嵌入模型、关键词、样本向量初始化

        logger.info(f"知识库助手初始化完成 | 路由模式：纯LLM意图判断")

    # ======================= 【核心：纯 LLM 意图判断】 =======================
    def should_retrieve(self, query: str, agent_id: str = "default") -> Tuple[bool, float]:
        """
        return (是否需要检索, 置信度得分)
        企业级：纯LLM意图路由，无关键词、无向量
        """
        if not query or not isinstance(query, str):
            return False, 0.0

        try:
            # 企业级标准路由提示词（严格输出，无任何多余内容）
            prompt = f"""
你是企业内部知识库路由引擎，只做二分类判断，严格输出结果。

任务：判断用户问题是否需要查询【内部制度/考勤/休假/报销/开发文档】。

输出规则（必须严格遵守）：
- 需要查询 → 输出：NEED_QUERY
- 不需要查询（闲聊/问候/通用知识）→ 输出：NO_QUERY

禁止解释、禁止聊天、禁止加标点、只输出指令内容！

用户问题：{query}
            """

            # 调用你的LLM（与项目原有 llm_client 保持一致）
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.invoke(messages)
            resp_text = response.strip().upper()
            # 解析结果
            if "NEED_QUERY" in resp_text:
                logger.info(f"LLM判定需要检索：{query}")
                return True, 1.0
            else:
                logger.info(f"LLM判定无需检索：{query}")
                return False, 0.0

        except Exception as e:
            logger.error(f"LLM意图判断失败：{e}，默认不检索")
            return False, 0.0

    # ======================= 以下完全保持你原有逻辑不动 =======================
    def _get_context_builder(self) -> Optional[ContextBuilder]:
        if not self.tool_registry:
            return None
        memory_tool = self.tool_registry.get_tool("memory")
        rag_tool = self.tool_registry.get_tool("rag")
        if not memory_tool or not rag_tool:
            return None
        config = ContextConfig(max_tokens=8000, reserve_ratio=0.15)
        return ContextBuilder(memory_tool=memory_tool, rag_tool=rag_tool, config=config)

    def _build_context(self, user_query: str, **kwargs) -> str:
        builder = self._get_context_builder()
        if not builder:
            return user_query
        try:
            return builder.build(
                user_query=user_query,
                conversation_history=[],
                system_instructions=self.system_prompt,
                additional_packets=[],** kwargs
            )
        except:
            return user_query

    def list_tools(self) -> List[str]:
        return self.tool_registry.list_tools() if self.tool_registry else []

    def get_tool_statistics(self) -> Dict[str, Any]:
        return {
            "agent_name": self.name,
            "total_tools": len(self.list_tools()),
            "tools_list": self.list_tools(),
            "tool_registry": "global" if self.tool_registry else "custom",
            "enable_tool_calling": self.enable_tool_calling,
            "max_tool_iterations": self.max_tool_iterations,
            "timestamp": datetime.now().isoformat()
        }

    def run_with_context(
        self,
        input_text: str,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        enable_memory: bool = True,
        enable_rag: bool = True,
        auto_tool_selection: bool = True,** kwargs
    ) -> str:
        should, score = self.should_retrieve(input_text, agent_id or "default")
        if not enable_rag:
            should = False

        if not should:
            logger.info(f"无需检索，直接LLM回答 | score={score:.2f}")
            direct_answer_prompt = f"""
            你是企业内部智能助手，请友好、简洁、专业地回答用户问题。

            如果是闲聊、问候、通用知识，请直接回答。
            如果是不清楚的问题，请礼貌回复：“抱歉，我无法回答这个问题。”

            用户问题：{input_text}
            """
            messages = [{"role": "user", "content": direct_answer_prompt}]
            
            return self.llm_client.invoke(messages, **kwargs)
        else:
            logger.info(f"需要检索，执行RAG | score={score:.2f}")
            try:
                rag_tool = self.tool_registry.get_tool("rag")
                return rag_tool.run({
                    "action": "ask",
                    "query": input_text,
                    "limit": 5,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "session": session_id,
                    "namespace": namespace,
                    "min_score": 0.75
                })
            except Exception as e:
                logger.error(f"RAG调用失败: {e}")
                return super().run(input_text, **kwargs)

    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[str] = None,** kwargs,
    ) -> str:
        should, score = self.should_retrieve(input_text)
        if not should:
            return super().run(input_text,** kwargs)
        logger.info("run() 缺少上下文，直接LLM回答")
        return super().run(input_text, **kwargs)


def create_universal_assistant(
    name: str = "knowledge_base_assistant",
    llm: Optional[HelloAgentsLLM] = None,
    system_prompt: Optional[str] = None,
    workspace: str = ".",** kwargs
) -> KnowledgeBaseAssistant:
    return KnowledgeBaseAssistant(
        name=name,
        llm=llm,
        system_prompt=system_prompt,
        workspace=workspace,** kwargs
    )

UniversalEnterpriseAgent = KnowledgeBaseAssistant