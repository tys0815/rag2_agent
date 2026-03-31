"""情景记忆实现（企业级四层架构版）
按照第8章架构设计的情景记忆，提供：
- 具体交互事件存储
- 时间序列组织
- 上下文丰富的记忆
- 模式识别能力
- 四层隔离：user_id → session_id → episode
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import math
import json
import logging

logger = logging.getLogger(__name__)

from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..storage import SQLiteDocumentStore, QdrantVectorStore
from ..embedding import get_text_embedder, get_dimension

class Episode:
    """情景记忆中的单个情景（企业级四层结构）"""
    
    def __init__(
        self,
        episode_id: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        content: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        importance: float = 0.5
    ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance

class EpisodicMemory(BaseMemory):
    """情景记忆实现（企业级四层架构）
    
    特点：
    - 存储具体的交互事件（对话历史、操作行为）
    - 四层完全隔离：用户 → 会话 → 事件
    - 按时间序列组织
    - 支持模式识别和回溯
    - 支持多租户、多应用、多助手隔离
    """
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 本地缓存（内存）
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, Dict[str, List[str]]] = {}  # agent_id → session_id -> episode_ids
        
        # 模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        # 权威文档存储（SQLite）
        db_dir = self.config.storage_path if hasattr(self.config, 'storage_path') else "./memory_data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)

        # 统一嵌入模型
        self.embedder = get_text_embedder()

        # 向量存储
        from ..storage.qdrant_store import QdrantConnectionManager
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_store = QdrantConnectionManager.get_instance(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=get_dimension(getattr(self.embedder, 'dimension', 512)),
            distance=os.getenv("QDRANT_DISTANCE", "cosine")
        )

    def add(self, memory_item: MemoryItem, user_id: str = None, session_id: str = None, **kwargs) -> str:
        """添加情景记忆（四层隔离）"""

        # 1) 权威存储（SQLite）
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            role=memory_item.role,
            content=memory_item.content,
            memory_type="episodic",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "session_id": session_id,
                **memory_item.metadata
            }
        )

        # 2) 向量索引（Qdrant
        try:
            embedding = self.embedder.encode(memory_item.content)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            self.vector_store.add_vectors(
                vectors=[embedding],
                metadata=[{
                    "memory_id": memory_item.id,
                    "user_id": memory_item.user_id,
                    "memory_type": "episodic",
                    "importance": memory_item.importance,
                    "content": memory_item.content,
                    "session_id": session_id,
                    "role": memory_item.role
                }],
                ids=[memory_item.id]
            )
        except Exception:
            pass

        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, user_id=None, session_id=None, **kwargs) -> List[MemoryItem]:
        """
        企业级情景记忆检索（四层过滤：向量+时间+重要度+会话+用户）
        返回排序：摘要 > 用户输入 > 助理回复 > 观察信息
        """
        time_range: Optional[Tuple[datetime, datetime]] = kwargs.get("time_range")
        importance_threshold = kwargs.get("min_importance", 0.0)

        # 向量检索
        try:
            query_vec = self.embedder.encode(query)
            if hasattr(query_vec, "tolist"):
                query_vec = query_vec.tolist()
                
            where = {"memory_type": "episodic"}
            if user_id:
                where["user_id"] = user_id
            if session_id:
                where["session_id"] = session_id

            hits = self.vector_store.search_similar(
                query_vector=query_vec,
                limit=max(limit * 5, 20),
                where=where
            )
        except Exception:
            hits = []

        now_ts = int(datetime.now().timestamp())
        results: List[Tuple[float, MemoryItem]] = []
        seen = set()

        # ==================== 企业级：角色权重定义（核心） ====================
        ROLE_WEIGHT = {
            "summary": 1.3,      # 摘要最高
            "user": 1.1,         # 用户问题次之
            "assistant": 1.0,    # 助理回答正常
            "observation": 0.8   # 观察信息最低（系统提示）
        }

        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = meta.get("memory_id")
            role = meta.get("role", "unknown")  # 取出 role（关键！）
            
            if not mem_id or mem_id in seen:
                continue

            # 从文档库获取真实数据
            doc = self.doc_store.get_memory(mem_id)
            if not doc:
                continue

            # ==================== 企业级：分数计算 ====================
            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(doc["timestamp"])) / 86400.0)
            recency_score = 1.0 / (1.0 + age_days)
            imp = float(doc.get("importance", 0.5))
            
            # 基础分数
            base_relevance = vec_score * 0.8 + recency_score * 0.2
            
            # 重要性加权
            importance_weight = 0.8 + (imp * 0.4)
            
            # 角色加权（企业级关键）
            role_weight = ROLE_WEIGHT.get(role, 1.0)
            
            # 最终综合分数
            combined = base_relevance * importance_weight * role_weight

            # ==================== 构建记忆项 ====================
            item = MemoryItem(
                id=doc["memory_id"],
                content=doc["content"],
                memory_type=doc["memory_type"],
                user_id=doc["user_id"],
                role=role,  # 必须带上 role
                timestamp=datetime.fromtimestamp(doc["timestamp"]),
                importance=doc.get("importance", 0.5),
                metadata={
                    **doc.get("properties", {}),
                    "relevance_score": round(combined, 4),
                    "role": role,
                    "session_id": session_id
                }
            )
            
            results.append((combined, item))
            seen.add(mem_id)

        if not results:
            return []

        # 按综合得分倒序（企业级：摘要优先，最新优先）
        results.sort(key=lambda x: x[0], reverse=True)
        final_items = [it for _, it in results[:limit]]

        # ==================== 企业级：返回前过滤 observation 数量（最多1条） ====================
        # 避免系统提示占满上下文
        obs_items = [it for it in final_items if it.metadata.get("role") == "observation"]
        normal_items = [it for it in final_items if it.metadata.get("role") != "observation"]
        
        # 最终组合：正常记忆 + 最多1条观察
        final_items = normal_items[:limit-1] + obs_items[:1]

        return final_items
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
        user_id: str = None,
        session_id: str = None
    ) -> bool:
        updated = False
        for episode in self.episodes:
            if episode.episode_id == memory_id:
                if user_id and episode.user_id != user_id:
                    continue
                if session_id and episode.session_id != session_id:
                    continue
                if content is not None:
                    episode.content = content
                if importance is not None:
                    episode.importance = importance
                if metadata is not None:
                    episode.context.update(metadata.get("context", {}))
                    if "outcome" in metadata:
                        episode.outcome = metadata["outcome"]
                updated = True
                break

        doc_updated = self.doc_store.update_memory(memory_id, content, importance, metadata)
        if content:
            try:
                emb = self.embedder.encode(content).tolist()
                doc = self.doc_store.get_memory(memory_id)
                payload = {
                    "memory_id": memory_id,
                    "user_id": doc["user_id"],
                    "memory_type": "episodic",
                    "session_id": doc["properties"]["session_id"],
                    "importance": doc.get("importance", 0.5),
                    "content": content
                }
                self.vector_store.add_vectors([emb], [payload], [memory_id])
            except Exception:
                pass
        return updated or doc_updated
    
    def remove(self, memory_id: str, user_id: str = None, session_id: str = None) -> bool:
        removed = False
        for i, episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                if user_id and episode.user_id != user_id:
                    continue
                if session_id and episode.session_id != session_id:
                    continue
                ep = self.episodes.pop(i)
                try:
                    self.sessions[ep.session_id].remove(memory_id)
                    if not self.sessions[ep.session_id]:
                        del self.sessions[ep.session_id]
                except:
                    pass
                removed = True
                break

        doc_del = self.doc_store.delete_memory(memory_id)
        try:
            self.vector_store.delete_memories([memory_id])
        except:
            pass
        return removed or doc_del
    
    def has_memory(self, memory_id: str, user_id=None, session_id=None) -> bool:
        for ep in self.episodes:
            if ep.episode_id == memory_id:
                if user_id and ep.user_id != user_id:
                    continue
                if session_id and ep.session_id != session_id:
                    continue
                return True
        return False
    
    def clear(self, user_id=None, agent_id=None, session_id=None):
        to_del = []
        for ep in self.episodes:
            if user_id and ep.user_id != user_id:
                continue
            if session_id and ep.session_id != session_id:
                continue
            to_del.append(ep.episode_id)

        for mid in to_del:
            self.remove(mid)

        if agent_id and session_id:
            self.sessions.get(agent_id, {}).pop(session_id, None)
        elif agent_id:
            self.sessions.pop(agent_id, None)
    
    def forget(self, strategy="importance_based", threshold=0.1, max_age_days=30, user_id=None, session_id=None) -> int:
        now = datetime.now()
        to_del = []
        for ep in self.episodes:
            if user_id and ep.user_id != user_id:
                continue
            if session_id and ep.session_id != session_id:
                continue

            forget = False
            if strategy == "importance_based" and ep.importance < threshold:
                forget = True
            if strategy == "time_based":
                if ep.timestamp < now - timedelta(days=max_age_days):
                    forget = True
            if strategy == "capacity_based":
                pass

            if forget:
                to_del.append(ep.episode_id)

        cnt = 0
        for mid in list(set(to_del)):
            if self.remove(mid, user_id=user_id, session_id=session_id):
                cnt += 1
        return cnt

    def get_all(self, user_id=None, session_id=None) -> List[MemoryItem]:
        items = []
        for ep in self._filter_episodes(user_id, session_id):
            items.append(MemoryItem(
                id=ep.episode_id,
                content=ep.content,
                memory_type="episodic",
                user_id=ep.user_id,
                timestamp=ep.timestamp,
                importance=ep.importance,
                metadata={
                    "session_id": ep.session_id,
                    "context": ep.context,
                    "outcome": ep.outcome
                }
            ))
        return items

    def get_stats(self, user_id=None, agent_id=None, session_id=None) -> Dict[str, Any]:
        filtered = self._filter_episodes(user_id, agent_id, session_id)
        return {
            "count": len(filtered),
            "sessions": len(self.sessions.get(agent_id, {})) if agent_id else sum(len(v) for v in self.sessions.values()),
            "avg_importance": sum(e.importance for e in filtered) / len(filtered) if filtered else 0,
            "memory_type": "episodic"
        }

    def _filter_episodes(self, user_id=None, session_id=None) -> List[Episode]:
        res = self.episodes
        if user_id:
            res = [e for e in res if e.user_id == user_id]
        if session_id:
            res = [e for e in res if e.session_id == session_id]
        return res