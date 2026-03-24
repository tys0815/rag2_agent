from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from .react_agent import ReActAgent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..tools.registry import ToolRegistry
from ..context import ContextBuilder, ContextConfig
from helloAgents.core.exceptions import ToolException

logger = logging.getLogger(__name__)


class KnowledgeBaseAssistant(ReActAgent):
    """
    企业级知识库助手（休假/考勤/制度/报销专用）
    架构：纯 ReAct 推理闭环，无独立 LLM 判断，企业生产标准
    """

    def __init__(
        self,
        name: str = "knowledge_base_assistant",
        llm: Optional[HelloAgentsLLM] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        # 企业级标准：初始化 ReAct
        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            config=config,
            max_steps=4,
        )

    # ======================= 企业级 ReAct 上下文构建 =======================
    def _get_context_builder(self) -> Optional[ContextBuilder]:
        if not self.tool_registry:
            return None
        memory_tool = self.tool_registry.get_tool("memory")
        rag_tool = self.tool_registry.get_tool("rag")
        if not memory_tool or not rag_tool:
            return None
        config = ContextConfig(max_tokens=8000, reserve_ratio=0.15)
        return ContextBuilder(memory_tool=memory_tool, rag_tool=rag_tool, config=config)

    def _build_context(self, user_query: str,** kwargs) -> str:
        builder = self._get_context_builder()
        if not builder:
            return user_query
        try:
            return builder.build(
                user_query=user_query,
                conversation_history=[],
                system_instructions=self.system_prompt,** kwargs
            )
        except:
            return user_query

    # ======================= 工具统计（企业监控） =======================
    def list_tools(self) -> List[str]:
        return self.tool_registry.list_tools() if self.tool_registry else []

    def get_tool_statistics(self) -> Dict[str, Any]:
        return {
            "agent_name": self.name,
            "total_tools": len(self.list_tools()),
            "tools_list": self.list_tools(),
            "timestamp": datetime.now().isoformat()
        }

    # ======================= 企业业务入口（完全ReAct驱动） =======================
    def run(
        self,
        input_text: str,
        **kwargs
    ) -> str:
        # if not enable_rag:
        #     # 禁用RAG → 直接走ReAct通用回答
        #     return super().run(input_text,** kwargs)

        # ===================== 企业级核心：全部交给 ReAct 自动判断 =====================
        logger.info("业务交由 ReAct 统一推理：自动判断是否检索知识库")
        return super().run(input_text, **kwargs)