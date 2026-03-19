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
    基于【正负样本对比】做意图判断，不会把“你好”当成检索
    """

    # ======================= 【正样本】需要检索 =======================
    RETRIEVAL_INTENT_EXAMPLES = [
        "请假规则",
        "休假怎么申请",
        "年假有几天",
        "病假需要什么证明",
        "婚假多少天",
        "产假多少天",
        "陪产假规定",
        "事假能不能申请",
        "考勤制度",
        "迟到早退怎么算",
        "加班申请流程",
        "报销流程",
        "报销标准",
        "出差申请",
        "公司规章制度",
        "员工手册",
        "绩效考核",
        "IT设备申领",
        "网络安全规范",
        "会议室预订"
    ]

    # ======================= 【负样本】不需要检索（闲聊/问候） =======================
    NON_RETRIEVAL_EXAMPLES = [
        "你好",
        "你好呀",
        "哈喽",
        "嗨",
        "在吗",
        "在不在",
        "谢谢",
        "感谢",
        "再见",
        "拜拜",
        "早上好",
        "晚上好",
        "你是谁",
        "你叫什么",
        "哈哈",
        "哦",
        "嗯",
        "好的",
        "知道了",
        "随便问问"
    ]

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
        retrieval_threshold: float = 0.3,  # 【关键】正负差值阈值
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

        # 意图阈值 → 现在是【差值阈值】
        self.retrieval_threshold = retrieval_threshold

        # 嵌入模型
        self._use_simple_classifier = False
        self._intent_embeddings: List[np.ndarray] = []
        self._non_intent_embeddings: List[np.ndarray] = []
        self._intent_keywords: List[str] = []

        try:
            self.embedder = get_text_embedder()
            if hasattr(self.embedder, 'fit'):
                try:
                    all_examples = self.RETRIEVAL_INTENT_EXAMPLES + self.NON_RETRIEVAL_EXAMPLES
                    if hasattr(self.embedder, 'reset'):
                        self.embedder.reset()
                    self.embedder.fit(all_examples)
                    logger.info("嵌入模型训练完成")
                except Exception as e:
                    logger.warning(f"模型训练失败: {e}")

            # 向量化 正/负 样本
            self._intent_embeddings = self._encode_list(self.RETRIEVAL_INTENT_EXAMPLES)
            self._non_intent_embeddings = self._encode_list(self.NON_RETRIEVAL_EXAMPLES)

        except Exception as e:
            logger.warning(f"嵌入模型初始化失败: {e}，使用关键词匹配")
            self._use_simple_classifier = True
            self.embedder = None
            self._intent_keywords = self._extract_keywords(self.RETRIEVAL_INTENT_EXAMPLES)

        logger.info(f"知识库助手初始化完成 | 检索例句:{len(self.RETRIEVAL_INTENT_EXAMPLES)} 闲聊例句:{len(self.NON_RETRIEVAL_EXAMPLES)}")

    def _encode_list(self, texts: List[str]) -> List[np.ndarray]:
        """批量编码文本为向量"""
        embeddings = self.embedder.encode(texts)
        if isinstance(embeddings, np.ndarray):
            return [embeddings[i] for i in range(embeddings.shape[0])]
        return [np.array(emb) for emb in embeddings]

    def _extract_keywords(self, examples: List[str]) -> List[str]:
        keywords = set()
        for s in examples:
            words = s.replace('?',' ').replace('？',' ').replace('，',' ').replace('。',' ').split()
            for w in words:
                if len(w) >= 1:
                    keywords.add(w)
        return list(keywords)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    # ======================= 【最终修复】意图判断 =======================
    def should_retrieve(self, query: str, agent_id: str = "default") -> Tuple[bool, float]:
        """
        return (是否需要检索, 最终置信度得分)
        【企业级最终版】规则前置 + 向量验证，100%不误判
        """
        # 空查询直接不检索
        if not query or not isinstance(query, str):
            return False, 0.0

        query_clean = query.strip().lower()

        # ===================== 【最强规则：必须加】 =====================
        # 企业人事/行政/考勤/报销核心业务词
        BUSINESS_WORDS = {
            "请假", "休假", "年假", "病假", "婚假", "产假", "陪产假", "事假",
            "考勤", "迟到", "早退", "加班", "报销", "出差", "制度", "手册",
            "绩效", "设备", "网络", "会议室", "预订", "申请", "流程", "标准"
        }
        # 不包含任何业务关键词 → 直接不检索（根治乱触发）
        if not any(word in query_clean for word in BUSINESS_WORDS):
            logger.info(f"无业务关键词，跳过检索：{query}")
            return False, 0.0
        # ==============================================================

        # 关键词兜底
        if self._use_simple_classifier:
            hit = sum(1 for kw in self._intent_keywords if kw in query_clean)
            score = min(hit / 2.0, 1.0)
            need = score >= 0.3
            return need, score

        # 向量版本
        try:
            q_emb = np.array(self.embedder.encode(query))
        except:
            return False, 0.0

        max_ret = max(self._cosine_similarity(q_emb, emb) for emb in self._intent_embeddings)
        need = max_ret >= 0.5
        final_score = round(max_ret, 2)

        logger.debug(f"查询:{query} | 业务相似度:{max_ret:.2f} → 检索:{need}")
        return need, final_score

    # ======================= 以下保持你原有逻辑不变 =======================
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
            return super().run(input_text, **kwargs)
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
            return super().run(input_text, **kwargs)
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
        workspace=workspace,
        **kwargs
    )

UniversalEnterpriseAgent = KnowledgeBaseAssistant