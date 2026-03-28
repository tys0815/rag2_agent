"""工作记忆实现（最终4层架构版）
支持：多用户 + 多助手（RAG/旅游） + 多会话 + 自动内存清理
存储结构：user_id → agent_id → session_id → memories
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import heapq
import uuid

from ..base import BaseMemory, MemoryItem, MemoryConfig


class WorkingMemory(BaseMemory):
    """
    工作记忆实现（四层隔离终极版）

    核心定位：
    - 短期上下文管理（纯内存，不落地）
    - 严格隔离：用户 → 助手类型 → 独立会话
    - 容量与Token双重限制，防止溢出
    - 自动遗忘机制，模拟人类短期记忆

    特点：
    - 容量有限（通常10-20条记忆）
    - 时效性强（会话级别，超时自动清理）
    - 优先级管理（重要性+时间衰减）
    - 自动清理过期记忆与闲置会话
    - 支持RAG助手、旅游助手等多角色完全隔离
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        """
        初始化工作记忆

        参数:
            config: 记忆配置（容量、Token上限、衰减系数等）
            storage_backend: 存储后端（工作记忆为纯内存，无需持久化）
        """
        super().__init__(config, storage_backend)

        # 工作记忆核心配置
        self.max_capacity = self.config.working_memory_capacity    # 单会话最大记忆条数
        self.max_tokens = self.config.working_memory_tokens        # 单会话最大Token数
        self.max_age_minutes = getattr(self.config, 'working_memory_ttl_minutes', 120)  # 单条记忆过期时间

        # ===================== 四层内存存储结构 =====================
        # user_id → session_id → List[MemoryItem]
        self.memories: Dict[str, Dict[str, List[MemoryItem]]] = {}  # 记忆主体
        self.memory_heaps: Dict[str, Dict[str, List]] = {}           # 优先级堆（快速淘汰低优先级）
        self.session_tokens: Dict[str, Dict[str, int]] = {}          # 各会话Token计数
        self.session_last_active: Dict[str, Dict[str, datetime]] = {} # 会话最后活跃时间（用于清理）

        # 内存保护机制（防止服务器长期运行OOM）
        self.max_inactive_minutes = 60       # 会话闲置60分钟自动销毁
        self.max_total_sessions = 2000       # 全局最大会话数上限

    # -------------------------------------------------------------------------
    # 核心内部工具方法
    # -------------------------------------------------------------------------

    def _get_session(self,
                    user_id: str = "default_user",
                    session_id: str = "default_session"):
        """
        【内部核心】获取或自动初始化指定用户+助手+会话的存储结构
        所有读写操作必须先调用此方法，确保层级存在

        返回:
            tuple: (当前会话记忆列表, 当前会话优先级堆, 当前会话Token数)
        """
        # 1. 初始化用户层级
        if user_id not in self.memories:
            self.memories[user_id] = {}
            self.memory_heaps[user_id] = {}
            self.session_tokens[user_id] = {}
            self.session_last_active[user_id] = {}


        # 2. 初始化会话层级
        if session_id not in self.memories[user_id]:
            self.memories[user_id][session_id] = []
            self.memory_heaps[user_id][session_id] = []
            self.session_tokens[user_id][session_id] = 0

        # 4. 刷新会话最后活跃时间
        self.session_last_active[user_id][session_id] = datetime.now()

        # 5. 触发自动清理（惰性清理，不阻塞业务）
        self._cleanup_inactive_sessions()

        return (
            self.memories[user_id][session_id],
            self.memory_heaps[user_id][session_id],
            self.session_tokens[user_id][session_id]
        )

    def _cleanup_inactive_sessions(self):
        """
        【内部】自动清理超时/超量会话，保护服务器内存
        逻辑：
            1. 清理超过60分钟未活动的会话
            2. 全局会话超限，清理最久未使用的会话
        """
        now = datetime.now()
        to_clean = []

        # 遍历所有会话，标记超时待清理
        for uid, agents in self.session_last_active.items():
            for aid, sessions in agents.items():
                for sid, last_active in sessions.items():
                    inactive_min = (now - last_active).total_seconds() / 60
                    if inactive_min > self.max_inactive_minutes:
                        to_clean.append((uid, aid, sid))

        # 统计总会话数，超限则按LRU清理
        total_sessions = 0
        for u in self.memories:
            for a in self.memories[u]:
                total_sessions += len(self.memories[u][a])

        if total_sessions > self.max_total_sessions:
            all_sessions = []
            for u, agents in self.session_last_active.items():
                for a, sessions in agents.items():
                    for s, lat in sessions.items():
                        all_sessions.append((lat, u, a, s))
            all_sessions.sort()  # 按时间升序，最久的排在前面
            excess = total_sessions - self.max_total_sessions
            for item in all_sessions[:excess]:
                to_clean.append((item[1], item[2], item[3]))

        # 执行清理
        for uid, aid, sid in set(to_clean):
            self.clear_session(uid, aid, sid)

    # -------------------------------------------------------------------------
    # 公共工具方法
    # -------------------------------------------------------------------------

    def generate_session_id(self) -> str:
        """生成全局唯一的会话ID"""
        return f"ses_{uuid.uuid4().hex[:16]}"

    def clear_session(self, user_id: str, session_id: str):
        """
        清空指定用户+助手下的某个会话
        用于：切换会话、手动删除会话、超时销毁
        """
        try:
            self.memories[user_id].pop(session_id, None)
            self.memory_heaps[user_id].pop(session_id, None)
            self.session_tokens[user_id].pop(session_id, None)
            self.session_last_active[user_id].pop(session_id, None)
        except KeyError:
            pass

    # -------------------------------------------------------------------------
    # 对外标准接口（方法名完全不变）
    # -------------------------------------------------------------------------

    def add(self,
            memory_item: MemoryItem,
            user_id: str = "default_user",
            session_id: str = "default_session") -> str:
        """
        添加一条记忆到工作记忆

        参数:
            memory_item: 记忆对象
            user_id: 用户ID
            session_id: 会话ID

        返回:
            记忆ID
        """
        # 获取会话存储
        memories, heap, _ = self._get_session(user_id, session_id)

        # 先清理本会话内过期记忆
        self._expire_old_memories(user_id, session_id)

        # 计算优先级并加入堆
        priority = self._calculate_priority(memory_item)
        heapq.heappush(heap, (-priority, memory_item.timestamp, memory_item))

        # 添加到记忆列表
        memories.append(memory_item)

        # 增加Token计数
        self.session_tokens[user_id][session_id] += len(memory_item.content.split())

        # 强制执行容量/Token限制
        self._enforce_capacity_limits(user_id, session_id)

        return memory_item.id

    def retrieve(self,
                 query: str,
                 limit: int = 5,
                 user_id: str = "default_user",
                 session_id: str = "default_session",** kwargs) -> List[MemoryItem]:
        """
        根据查询内容检索相关工作记忆（语义+关键词混合检索）

        参数:
            query: 查询语句
            limit: 返回最大条数
            user_id: 用户ID
            agent_id: 助手ID
            session_id: 会话ID

        返回:
            匹配的记忆列表
        """
        # 会话不存在则返回空
        try:
            memories = self.memories[user_id][session_id]
        except KeyError:
            return []

        # 清理过期记忆
        self._expire_old_memories(user_id, session_id)

        # 过滤未被遗忘的有效记忆
        active_memories = [m for m in memories if not m.metadata.get("forgotten", False)]
        if not active_memories:
            return []

        # 向量相似度检索
        vector_scores = {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            docs = [query] + [m.content for m in active_memories]
            vec = TfidfVectorizer(stop_words=None, lowercase=True)
            mtx = vec.fit_transform(docs)
            sim = cosine_similarity(mtx[0:1], mtx[1:]).flatten()
            for i, m in enumerate(active_memories):
                vector_scores[m.id] = sim[i]
        except Exception:
            vector_scores = {}

        # 计算最终得分并排序
        scored_memories = []
        query_lower = query.lower()
        for memory in active_memories:
            content_lower = memory.content.lower()
            vector_score = vector_scores.get(memory.id, 0.0)

            # 关键词匹配得分
            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                q_words = set(query_lower.split())
                c_words = set(content_lower.split())
                intersection = q_words.intersection(c_words)
                if intersection:
                    keyword_score = len(intersection) / len(q_words.union(c_words)) * 0.8

            # 混合分数
            if vector_score > 0:
                relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                relevance = keyword_score

            # 时间衰减 + 重要性加权
            relevance *= self._calculate_time_decay(memory.timestamp)
            final_score = relevance * (0.8 + memory.importance * 0.4)

            if final_score > 0:
                scored_memories.append((final_score, memory))

        # 按得分降序返回
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored_memories[:limit]]

    def update(self,
               memory_id: str,
               content: str = None,
               importance: float = None,
               metadata: Dict[str, Any] = None,
               user_id: str = "default_user",
               session_id: str = "default_session") -> bool:
        """
        更新指定记忆内容、重要性或元数据

        返回:
            bool: 是否更新成功
        """
        try:
            memories = self.memories[user_id][session_id]
        except KeyError:
            return False

        for memory in memories:
            if memory.id == memory_id:
                # 更新内容与Token
                if content:
                    old_tokens = len(memory.content.split())
                    memory.content = content
                    new_tokens = len(content.split())
                    self.session_tokens[user_id][session_id] += (new_tokens - old_tokens)

                # 更新重要性
                if importance:
                    memory.importance = importance

                # 更新元数据
                if metadata:
                    memory.metadata.update(metadata)

                # 重新计算堆优先级
                self._update_heap_priority(user_id, session_id)
                return True
        return False

    def remove(self,
               memory_id: str,
               user_id: str = "default_user",
               session_id: str = "default_session") -> bool:
        """
        从指定会话中删除一条记忆

        返回:
            bool: 是否删除成功
        """
        try:
            memories = self.memories[user_id][session_id]
            for i, m in enumerate(memories):
                if m.id == memory_id:
                    # 移除记忆
                    memories.pop(i)
                    # 扣除Token
                    self.session_tokens[user_id][session_id] -= len(m.content.split())
                    # 重建堆
                    self._update_heap_priority(user_id, session_id)
                    return True
        except KeyError:
            return False
        return False

    def has_memory(self,
                   memory_id: str,
                   user_id: str = None,
                   session_id: str = None) -> bool:
        """
        检查记忆是否存在
        可全局查找/按用户查找/按会话查找
        """
        # 指定范围检查
        if user_id and session_id:
            try:
                return any(m.id == memory_id for m in self.memories[user_id][session_id])
            except KeyError:
                return False

        # 全局遍历检查
        for users in self.memories.values():
            for agents in users.values():
                for sess in agents.values():
                    for m in sess:
                        if m.id == memory_id:
                            return True
        return False

    def clear(self, user_id: str = None, session_id: str = None):
        """
        清空记忆（支持三级粒度）
        - 不传参数：清空全局所有记忆
        - 传user_id：清空该用户所有记忆
        - 传user+agent：清空该用户下该助手的所有会话
        - 传全量：清空指定会话
        """
        if user_id and session_id:
            self.clear_session(user_id, session_id)
        elif user_id:
            self.memories.pop(user_id, None)
            self.memory_heaps.pop(user_id, None)
            self.session_tokens.pop(user_id, None)
            self.session_last_active.pop(user_id, None)
        else:
            self.memories.clear()
            self.memory_heaps.clear()
            self.session_tokens.clear()
            self.session_last_active.clear()

    def get_stats(self, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        获取记忆统计信息
        支持：全局统计 / 用户维度 / 助手维度 / 会话维度
        """
        # 单会话详情
        if user_id and session_id:
            try:
                mem_list = self.memories[user_id][session_id]
                token_cnt = self.session_tokens[user_id][session_id]
                return {
                    "count": len(mem_list),
                    "tokens": token_cnt,
                    "max_capacity": self.max_capacity,
                    "max_tokens": self.max_tokens
                }
            except KeyError:
                return {}

        # 全局汇总
        total_mem = 0
        total_tok = 0
        user_count = len(self.memories)
        session_count = 0

        for u in self.memories:
            for a in self.memories[u]:
                session_count += len(self.memories[u][a])
                for s in self.memories[u][a]:
                    total_mem += len(self.memories[u][a][s])
                    total_tok += self.session_tokens[u][a][s]

        return {
            "total_memories": total_mem,
            "total_tokens": total_tok,
            "user_count": user_count,
            "session_count": session_count,
            "memory_type": "working"
        }

    def get_recent(self,
                   limit: int = 10,
                   user_id: str = "default_user",
                   session_id: str = "default_session") -> List[MemoryItem]:
        """
        获取当前会话最近N条记忆（按时间倒序）
        """
        try:
            mem_list = self.memories[user_id][session_id]
            return sorted(mem_list, key=lambda x: x.timestamp, reverse=True)[:limit]
        except KeyError:
            return []

    def get_important(self,
                      limit: int = 10,
                      user_id: str = "default_user",
                      session_id: str = "default_session") -> List[MemoryItem]:
        """
        获取当前会话最重要的N条记忆（按重要性倒序）
        """
        try:
            mem_list = self.memories[user_id][session_id]
            return sorted(mem_list, key=lambda x: x.importance, reverse=True)[:limit]
        except KeyError:
            return []

    def get_all(self,
                user_id: str = "default_user",
                session_id: str = "default_session") -> List[MemoryItem]:
        """
        获取当前会话所有记忆（副本，防止外部篡改）
        """
        try:
            return self.memories[user_id][session_id].copy()
        except KeyError:
            return []

    def get_context_summary(self,
                            max_length: int = 500,
                            user_id: str = "default_user",
                            session_id: str = "default_session") -> str:
        """
        生成当前会话的上下文摘要（用于给LLM提供简短上下文）
        按重要性+时间排序，自动截断长度
        """
        try:
            mem_list = self.memories[user_id][session_id]
            if not mem_list:
                return "No working memories available."
        except KeyError:
            return "No working memories available."

        # 排序：重要性高 → 时间新
        sorted_memories = sorted(mem_list, key=lambda m: (m.importance, m.timestamp), reverse=True)
        summary_parts = []
        current_length = 0

        for memory in sorted_memories:
            content = memory.content
            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
            else:
                remaining = max_length - current_length
                if remaining > 50:
                    summary_parts.append(content[:remaining] + "...")
                break

        return "Working Memory Context:\n" + "\n".join(summary_parts)

    def forget(self,
               strategy: str = "importance_based",
               threshold: float = 0.1,
               max_age_days: int = 1,
               user_id: str = "default_user",
               session_id: str = "default_session") -> int:
        """
        主动遗忘机制（模拟人脑记忆消退）

        参数:
            strategy: 遗忘策略
                - importance_based: 基于重要性（默认）
                - time_based: 基于时间
                - capacity_based: 基于容量溢出
            threshold: 重要性阈值
            max_age_days: 最大保留天数

        返回:
            遗忘条数
        """
        try:
            memories = self.memories[user_id][session_id]
        except KeyError:
            return 0

        now = datetime.now()
        to_remove = []

        # 第一步：强制清理TTL过期记忆
        cutoff_time = now - timedelta(minutes=self.max_age_minutes)
        for m in memories:
            if m.timestamp < cutoff_time:
                to_remove.append(m.id)

        # 第二步：按策略遗忘
        if strategy == "importance_based":
            # 遗忘低重要性记忆
            for m in memories:
                if m.importance < threshold:
                    to_remove.append(m.id)

        elif strategy == "time_based":
            # 遗忘超期记忆
            old_cutoff = now - timedelta(hours=max_age_days * 24)
            for m in memories:
                if m.timestamp < old_cutoff:
                    to_remove.append(m.id)

        elif strategy == "capacity_based":
            # 容量超限，遗忘优先级最低
            if len(memories) > self.max_capacity:
                sorted_mem = sorted(memories, key=lambda x: self._calculate_priority(x))
                excess_count = len(memories) - self.max_capacity
                for m in sorted_mem[:excess_count]:
                    to_remove.append(m.id)

        # 执行删除并统计
        forgotten_count = 0
        for memory_id in list(set(to_remove)):
            if self.remove(memory_id, user_id, session_id):
                forgotten_count += 1

        return forgotten_count

    # -------------------------------------------------------------------------
    # 底层算法方法
    # -------------------------------------------------------------------------

    def _calculate_priority(self, memory: MemoryItem) -> float:
        """
        计算记忆优先级 = 重要性 × 时间衰减系数
        """
        time_decay = self._calculate_time_decay(memory.timestamp)
        return memory.importance * time_decay

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """
        计算时间衰减：越旧的记忆权重越低
        每6小时衰减一次，最低保留10%权重
        """
        hours_passed = (datetime.now() - timestamp).total_seconds() / 3600
        decay_factor = self.config.decay_factor ** (hours_passed / 6)
        return max(0.1, decay_factor)

    def _enforce_capacity_limits(self, uid, sid):
        """
        强制执行单会话容量限制：
        1. 不超过最大条数
        2. 不超过最大Token数
        超限则自动删除优先级最低的记忆
        """
        try:
            mem_list = self.memories[uid][sid]
            token_count = self.session_tokens[uid][sid]

            # 条数超限
            while len(mem_list) > self.max_capacity:
                self._remove_lowest_priority_memory(uid, sid)

            # Token超限
            while token_count > self.max_tokens:
                self._remove_lowest_priority_memory(uid, sid)
                token_count = self.session_tokens[uid][sid]
        except KeyError:
            pass

    def _expire_old_memories(self, uid, sid):
        """
        清理当前会话中TTL过期的记忆
        并同步更新Token计数与堆结构
        """
        try:
            memories = self.memories[uid][sid]
            cutoff = datetime.now() - timedelta(minutes=self.max_age_minutes)
            kept_memories = []
            removed_tokens = 0

            for m in memories:
                if m.timestamp >= cutoff:
                    kept_memories.append(m)
                else:
                    removed_tokens += len(m.content.split())

            # 无变化直接返回
            if len(kept_memories) == len(memories):
                return

            # 更新存储
            self.memories[uid][sid] = kept_memories
            self.session_tokens[uid][sid] = max(0, self.session_tokens[uid][sid] - removed_tokens)

            # 重建堆
            heap = self.memory_heaps[uid][sid]
            heap.clear()
            for m in kept_memories:
                priority = self._calculate_priority(m)
                heapq.heappush(heap, (-priority, m.timestamp, m))
        except KeyError:
            pass

    def _remove_lowest_priority_memory(self, uid, sid):
        """
        删除当前会话中优先级最低的记忆
        """
        try:
            mem_list = self.memories[uid][sid]
            if not mem_list:
                return

            lowest_priority = float('inf')
            lowest_memory = None
            for m in mem_list:
                p = self._calculate_priority(m)
                if p < lowest_priority:
                    lowest_priority = p
                    lowest_memory = m

            if lowest_memory:
                self.remove(lowest_memory.id, uid, sid)
        except KeyError:
            pass

    def _update_heap_priority(self, uid, sid):
        """
        完全重建堆（用于记忆更新/删除后）
        """
        try:
            mem_list = self.memories[uid][sid]
            heap = self.memory_heaps[uid][sid]
            heap.clear()
            for m in mem_list:
                p = self._calculate_priority(m)
                heapq.heappush(heap, (-p, m.timestamp, m))
        except KeyError:
            pass

    def _mark_deleted_in_heap(self, memory_id: str):
        """
        堆不支持直接删除，仅作占位兼容
        """
        pass