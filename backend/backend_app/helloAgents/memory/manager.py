"""记忆管理器 - 记忆核心层的统一管理接口"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import logging

from .base import MemoryItem, MemoryConfig
from .types.working import WorkingMemory
from .types.episodic import EpisodicMemory
from .types.semantic import SemanticMemory
from .types.perceptual import PerceptualMemory

logger = logging.getLogger(__name__)

class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口

    负责：
    - 统一管理工作/情景/语义/感知记忆
    - 支持四层隔离：用户 → 助手 → 会话 → 记忆
    - 记忆生命周期、优先级、遗忘、整合机制
    - 多类型记忆统一读写入口
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
    ):
        self.config = config or MemoryConfig()

        # 初始化各类型记忆（已支持四层隔离）
        self.memory_types = {
            'working': WorkingMemory(self.config),
            'episodic': EpisodicMemory(self.config),
            'semantic': SemanticMemory(self.config),
            'perceptual': PerceptualMemory(self.config)
        }

        logger.info(f"MemoryManager 初始化完成，启用记忆类型: {list(self.memory_types.keys())}")

    # -------------------------------------------------------------------------
    # 添加记忆（四层标准：短期带全部，长期只带 user）
    # -------------------------------------------------------------------------
    def add_memory(
        self,
        content: str,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        memory_type: str = "working",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = True
    ) -> str:
        """添加记忆（四层隔离标准）

        标准规则：
        - working / episodic：带 user_id + agent_id + session_id
        - semantic / perceptual：只带 user_id
        """
        # 自动分类
        # if auto_classify:
        #     memory_type = self._classify_memory_type(content, metadata)

        # 计算重要性
        if importance is None:
            importance = self._calculate_importance(content, metadata)

        # 创建记忆项
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )

        # 添加到对应记忆库（严格按类型传参）
        if memory_type in self.memory_types:
            mem = self.memory_types[memory_type]

            # ====================== 核心标准 ======================
            if memory_type in ["working", "episodic"]:
                # 短期记忆：三层隔离
                return mem.add(memory_item, user_id=user_id, agent_id=agent_id, session_id=session_id)
            else:
                # 长期知识库：只按用户隔离
                return mem.add(memory_item)
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")

    # -------------------------------------------------------------------------
    # 检索记忆（标准：短期三层，长期一层）
    # -------------------------------------------------------------------------
    def retrieve_memories(
        self,
        query: str,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        time_range: Optional[tuple] = None
    ) -> List[MemoryItem]:
        """统一检索记忆（四层标准）"""
        if memory_types is None:
            memory_types = list(self.memory_types.keys())

        all_results = []
        per_type_limit = max(1, limit // len(memory_types))

        for mt in memory_types:
            if mt not in self.memory_types:
                continue

            mem = self.memory_types[mt]
            try:
                if mt in ["working", "episodic"]:
                    # 短期：三层过滤
                    res = mem.retrieve(
                        query=query,
                        limit=per_type_limit,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id
                    )
                else:
                    # 长期：只按用户
                    res = mem.retrieve(
                        query=query,
                        limit=per_type_limit,
                        user_id=user_id
                    )
                all_results.extend(res)
            except Exception as e:
                logger.warning(f"检索 {mt} 出错: {e}")

        all_results.sort(key=lambda x: x.importance, reverse=True)
        return all_results[:limit]

    # -------------------------------------------------------------------------
    # 更新记忆
    # -------------------------------------------------------------------------
    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新记忆（自动按类型处理）"""
        for mt, mem in self.memory_types.items():
            if mt in ["working", "episodic"]:
                if mem.has_memory(memory_id, user_id, agent_id, session_id):
                    return mem.update(memory_id, content, importance, metadata, user_id, agent_id, session_id)
            else:
                if mem.has_memory(memory_id):
                    return mem.update(memory_id, content, importance, metadata)
        logger.warning(f"未找到记忆 {memory_id}")
        return False

    # -------------------------------------------------------------------------
    # 删除记忆
    # -------------------------------------------------------------------------
    def remove_memory(
        self,
        memory_id: str,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str = "default_session"
    ) -> bool:
        """删除记忆"""
        for mt, mem in self.memory_types.items():
            if mt in ["working", "episodic"]:
                if mem.has_memory(memory_id, user_id, agent_id, session_id):
                    return mem.remove(memory_id, user_id, agent_id, session_id)
            else:
                if mem.has_memory(memory_id):
                    return mem.remove(memory_id)
        logger.warning(f"未找到记忆 {memory_id}")
        return False

    # -------------------------------------------------------------------------
    # 遗忘记忆
    # -------------------------------------------------------------------------
    def forget_memories(
        self,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30
    ) -> int:
        """全局遗忘（按标准隔离）"""
        total = 0
        for mt, mem in self.memory_types.items():
            if hasattr(mem, "forget"):
                if mt in ["working", "episodic"]:
                    total += mem.forget(strategy, threshold, max_age_days, user_id, agent_id, session_id)
                else:
                    total += mem.forget(strategy, threshold, max_age_days, user_id)
        logger.info(f"用户 {user_id} 遗忘完成，共删除 {total} 条")
        return total

    # -------------------------------------------------------------------------
    # 记忆整合（短期 → 长期）
    # -------------------------------------------------------------------------
    def consolidate_memories(
        self,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7
    ) -> int:
        """将高重要性工作记忆转为长期记忆"""
        if from_type not in self.memory_types or to_type not in self.memory_types:
            return 0

        src = self.memory_types[from_type]
        dst = self.memory_types[to_type]
        count = 0

        # 只从当前会话提取
        if from_type == "working":
            candidates = src.get_all(user_id, agent_id, session_id)
        else:
            candidates = src.get_all(user_id)

        for m in candidates:
            if m.importance >= importance_threshold:
                if from_type == "working":
                    src.remove(m.id, user_id, agent_id, session_id)
                else:
                    src.remove(m.id)

                m.memory_type = to_type
                m.importance = min(1.0, m.importance * 1.1)
                
                # 写入目标记忆（按标准）
                if to_type in ["working", "episodic"]:
                    dst.add(m, user_id=user_id, agent_id=agent_id, session_id=session_id)
                else:
                    dst.add(m)
                count += 1

        logger.info(f"记忆整合：{count} 条 {from_type} → {to_type}")
        return count

    # -------------------------------------------------------------------------
    # 获取统计
    # -------------------------------------------------------------------------
    def get_memory_stats(
        self,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str = "default_session"
    ) -> Dict[str, Any]:
        """获取记忆统计（支持四层维度）"""
        stats = {
            "user_id": user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "enabled_types": list(self.memory_types.keys()),
            "total_memories": 0,
            "memories_by_type": {}
        }

        for mt, mem in self.memory_types.items():
            if mt in ["working", "episodic"]:
                type_stats = mem.get_stats(user_id, agent_id, session_id)
            else:
                type_stats = mem.get_stats(user_id)

            stats["memories_by_type"][mt] = type_stats
            stats["total_memories"] += type_stats.get("count", 0)

        return stats

    # -------------------------------------------------------------------------
    # 清空记忆
    # -------------------------------------------------------------------------
    def clear_all_memories(
        self,
        user_id: str = None,
        agent_id: str = None,
        session_id: str = None
    ):
        """清空记忆（四层标准）"""
        for mt, mem in self.memory_types.items():
            if mt in ["working", "episodic"]:
                mem.clear(user_id, agent_id, session_id)
            else:
                mem.clear(user_id)
        logger.info(f"已清空记忆 user={user_id} agent={agent_id} session={session_id}")

    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------
    def _classify_memory_type(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        if metadata and metadata.get("type"):
            return metadata["type"]
        if self._is_episodic_content(content):
            return "episodic"
        elif self._is_semantic_content(content):
            return "semantic"
        return "working"

    def _is_episodic_content(self, content: str) -> bool:
        keywords = ["昨天", "今天", "明天", "上次", "记得", "发生", "经历", "刚才"]
        return any(k in content for k in keywords)

    def _is_semantic_content(self, content: str) -> bool:
        keywords = ["定义", "概念", "规则", "知识", "原理", "方法", "意思"]
        return any(k in content for k in keywords)

    def _calculate_importance(self, content: str, metadata: Optional[Dict[str, Any]]) -> float:
        importance = 0.5
        if len(content) > 100:
            importance += 0.1
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误", "记住"]
        if any(k in content for k in important_keywords):
            importance += 0.2
        if metadata:
            if metadata.get("priority") == "high":
                importance += 0.3
            elif metadata.get("priority") == "low":
                importance -= 0.2
        return max(0.0, min(1.0, importance))

    def __str__(self):
        return f"MemoryManager(types={list(self.memory_types.keys())})"