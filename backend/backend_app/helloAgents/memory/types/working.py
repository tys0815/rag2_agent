"""
企业级工作记忆（最简稳定版）
功能：
- 短期对话上下文管理
- Redis TTL 自动过期
- 按 user_id + session_id 隔离
- 自动保留最近 5 轮对话
"""

from typing import List, Dict, Any
import json
import redis
from datetime import datetime

from ..base import BaseMemory, MemoryItem, MemoryConfig

class WorkingMemory(BaseMemory):
    """
    企业级工作记忆（Redis + TTL）
    只做一件事：存最近5轮对话
    """

    def __init__(
        self,
        config: MemoryConfig,
        ttl_minutes: int = 20   # TTL 20分钟
    ):
        super().__init__(config, None)
        self.ttl_seconds = ttl_minutes * 60
        
        # Redis 连接（线上建议用连接池）
        try:
            self.redis = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True
            )
        except Exception as e:
            print(f"Redis连接失败: {e}")
        
        # 最多保留几轮对话（一轮=用户问+AI答）
        self.max_rounds = 5

    # ------------------------------
    # 核心：保存一轮对话（用户问题 + AI回答）
    # ------------------------------
    def add(self, item: MemoryItem, session_id: str = None, **kwargs):
        """保存一轮对话：用户问题 + AI回答"""
        history = []
        # 1. 读取历史
        historyList: List[MemoryItem] = self.retrieve(user_id=item.user_id, session_id=session_id)
        if historyList:
            for item in historyList:
                item_dict = {
                    "id": item.id,
                    "content": kwargs.get("user_content", ""),
                    "memory_type": "working",
                    "user_id": item.user_id,
                    "timestamp": item.timestamp.isoformat(),
                    "importance": item.importance,
                    "role": "user",
                    "session_id": session_id,
                    "metadata": item.metadata
                }
                history.append(item_dict) 
        # 2. 添加新对话
        user_item = {
            "id": item.id,
            "content": kwargs.get("user_content", ""),
            "memory_type": "working",
            "user_id": item.user_id,
            "timestamp": item.timestamp.isoformat(),
            "importance": item.importance,
            "role": "user",
            "session_id": session_id,
            "metadata": item.metadata
        }   
        history.append(user_item)

        assistant_item = {
            "id": item.id,
            "content": kwargs.get("assistant_content", ""),
            "memory_type": "working",
            "user_id": item.user_id,
            "timestamp": item.timestamp.isoformat(),
            "importance": item.importance,
            "role": "assistant",
            "session_id": session_id,
            "metadata": item.metadata
        }
        history.append(assistant_item)

        # 3. 只保留最近 N 轮
        keep_length = self.max_rounds * 2  # 一轮=2条消息
    
        # 只有当历史超过限制时才截断
        if len(history) > keep_length:
            history = history[-keep_length:]

        self.redis_key = f"working_memory:{item.user_id}:{session_id}"
        # 4. 写入 Redis + TTL
        try:
            self.redis.set(
                self.redis_key,
                json.dumps(history, ensure_ascii=False),
                ex=self.ttl_seconds
            )
        except Exception as e:
            print(f"Redis写入失败: {e}")

    # ------------------------------
    # 清空当前用户记忆
    # ------------------------------
    def clear(self):
        self.redis.delete(self.redis_key)

    # ------------------------------
    # 内部方法：从 Redis 读取历史
    # ------------------------------
    def retrieve(self, query: str = None, limit: int = 5, user_id: str = None, session_id: str = None, **kwargs) -> List[MemoryItem]:
        if not self.redis or not user_id or not session_id:
            return []

        redis_key = f"working_memory:{user_id}:{session_id}"
        data = self.redis.get(redis_key)

        if not data:
            return []

        try:
            history = json.loads(data)
            if len(history) > limit * 2:  # 一轮=2条消息
                history = history[-limit:]
        except:
            return []

        # 转为 MemoryItem
        items = []
        for msg in history:
            items.append(MemoryItem(
                id=msg["id"],
                content=msg["content"],
                memory_type="working",
                user_id=user_id,
                session_id=session_id,
                metadata=msg["metadata"],
                role=msg["role"],
                timestamp=datetime.fromisoformat(msg["timestamp"]) if "timestamp" in msg else None,
                importance=msg["importance"]
            ))
        return items
        

    def update(self, memory_id: str, **kwargs) -> bool:
        return False

    def remove(self, memory_id: str) -> bool:
        return False

    def has_memory(self, memory_id: str) -> bool:
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
        }

    def get_all(self) -> List[MemoryItem]:
        return self.retrieve()

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        return self.retrieve(limit=limit)

    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        return self.retrieve(limit=limit)
    
    def memory_item_to_dict(mem: MemoryItem) -> dict:
        return {
            "id": mem.id,
            "content": mem.content,
            "memory_type": mem.memory_type,
            "user_id": mem.user_id,
            "timestamp": mem.timestamp.isoformat() if isinstance(mem.timestamp, datetime) else str(mem.timestamp),
            "importance": mem.importance,
            "role": mem.role,
            "session_id": mem.session_id,
            "metadata": mem.metadata or {}
        }