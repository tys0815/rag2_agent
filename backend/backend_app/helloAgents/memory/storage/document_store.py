"""文档存储实现

支持多种文档数据库后端：
- SQLite: 轻量级关系型数据库
- PostgreSQL: 企业级关系型数据库（可扩展）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
import threading
import uuid
import time


class DocumentStore(ABC):
    """文档存储基类"""

    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        role: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Dict[str, Any] = None
    ) -> str:
        """添加记忆"""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单个记忆"""
        pass

    @abstractmethod
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索记忆"""
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        properties: Dict[str, Any] = None
    ) -> bool:
        """更新记忆"""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        pass

    @abstractmethod
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加文档"""
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        pass


class SQLiteDocumentStore(DocumentStore):
    """SQLite文档存储实现"""

    _instances = {}
    _initialized_dbs = set()

    def __new__(cls, db_path: str = "./memory.db"):
        abs_path = os.path.abspath(db_path)
        if abs_path not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[abs_path] = instance
        return cls._instances[abs_path]

    def __init__(self, db_path: str = "./memory.db"):
        if hasattr(self, '_initialized'):
            return

        self.db_path = db_path
        self.local = threading.local()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        abs_path = os.path.abspath(db_path)
        if abs_path not in self._initialized_dbs:
            self._init_database()
            self._initialized_dbs.add(abs_path)
            print(f"[OK] SQLite 文档存储初始化完成: {db_path}")

        self._initialized = True

    def _get_connection(self):
        """获取线程本地连接（修复多线程卡死）"""
        if not hasattr(self.local, 'connection'):
            # ✅ 关键修复：必须加 check_same_thread=False
            self.local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self.local.connection.row_factory = sqlite3.Row
        return self.local.connection

    def _init_database(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                importance REAL NOT NULL,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # 1. 文档表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,           
                metadata TEXT,        
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. 分块表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL, 
                chunk_index INTEGER,   
                metadata TEXT,         
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. ✅ 多对多关系表（核心）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunk (
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                PRIMARY KEY (document_id, chunk_id),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            )
        """)

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_role ON memories (role)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)"
        ]
        for idx in indexes:
            cursor.execute(idx)

        conn.commit()
        print("[OK] SQLite 数据库表和索引创建完成")

    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        role: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Dict[str, Any] = None
    ) -> str:
        """添加记忆（已完全修复）"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 确保用户存在
            cursor.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, user_id))

            # ✅ SQL 100% 正确
            cursor.execute("""
                INSERT OR REPLACE INTO memories 
                (id, user_id, role, content, memory_type, timestamp, importance, properties, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                memory_id,
                user_id,
                role,
                content,
                memory_type,
                timestamp,
                importance,
                json.dumps(properties) if properties else None
            ))

            conn.commit()
            return memory_id

        except Exception as e:
            print(f"[ERROR] add_memory 失败: {str(e)}")
            raise
        
    def add_document_chunk(self, document_id:str, chunk_ids:list[any]):
        if not chunk_ids:
            return False

        conn = self._get_connection()
        cursor = conn.cursor()

        # 构造多条 (doc_id, chunk_id)
        data = [(document_id, chunk_id["metadata"]["content_hash"]) for chunk_id in chunk_ids]

        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO document_chunk (document_id, chunk_id)
                VALUES (?, ?)
            """, data)
            conn.commit()
            return True
        except Exception as e:
            print(f"绑定失败: {e}")
            return False

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, user_id, role, content, memory_type, timestamp, importance, properties, created_at
            FROM memories WHERE id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "memory_id": row["id"],
            "user_id": row["user_id"],
            "role": row["role"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "timestamp": row["timestamp"],
            "importance": row["importance"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
            "created_at": row["created_at"]
        }

    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        where_conditions = []
        params = []

        if user_id:
            where_conditions.append("user_id = ?")
            params.append(user_id)
        if memory_type:
            where_conditions.append("memory_type = ?")
            params.append(memory_type)
        if start_time:
            where_conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where_conditions.append("timestamp <= ?")
            params.append(end_time)
        if importance_threshold:
            where_conditions.append("importance >= ?")
            params.append(importance_threshold)

        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        cursor.execute(f"""
            SELECT id, user_id, role, content, memory_type, timestamp, importance, properties, created_at
            FROM memories {where_clause}
            ORDER BY importance DESC, timestamp DESC LIMIT ?
        """, params + [limit])

        memories = []
        for row in cursor.fetchall():
            memories.append({
                "memory_id": row["id"],
                "user_id": row["user_id"],
                "role": row["role"],
                "content": row["content"],
                "memory_type": row["memory_type"],
                "timestamp": row["timestamp"],
                "importance": row["importance"],
                "properties": json.loads(row["properties"]) if row["properties"] else {},
                "created_at": row["created_at"]
            })
        return memories

    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        properties: Dict[str, Any] = None
    ) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        update_fields = []
        params = []

        if content is not None:
            update_fields.append("content = ?")
            params.append(content)
        if importance is not None:
            update_fields.append("importance = ?")
            params.append(importance)
        if properties is not None:
            update_fields.append("properties = ?")
            params.append(json.dumps(properties))

        if not update_fields:
            return False

        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(memory_id)

        cursor.execute(f"""
            UPDATE memories SET {', '.join(update_fields)} WHERE id = ?
        """, params)
        conn.commit()
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return cursor.rowcount > 0

    def get_database_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        stats = {}

        for table in ["users", "memories"]:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()["count"]

        cursor.execute("SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type")
        stats["memory_types"] = {row["memory_type"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT user_id, COUNT(*) as count FROM memories
            GROUP BY user_id ORDER BY count DESC LIMIT 10
        """)
        stats["top_users"] = {row["user_id"]: row["count"] for row in cursor.fetchall()}

        stats["store_type"] = "sqlite"
        stats["db_path"] = self.db_path
        return stats

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """✅ 关键修复：补充 role 参数"""
        doc_id = str(uuid.uuid4())
        user_id = metadata.get("user_id", "system") if metadata else "system"

        return self.add_memory(
            memory_id=doc_id,
            user_id=user_id,
            role="system",      # 必须加！
            content=content,
            memory_type="document",
            timestamp=int(time.time()),
            importance=0.5,
            properties=metadata or {}
        )

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.get_memory(document_id)

    def close(self):
        if hasattr(self.local, 'connection'):
            self.local.connection.close()
            delattr(self.local, 'connection')
            print("[OK] SQLite 连接已关闭")