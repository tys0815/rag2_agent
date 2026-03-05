"""
Neo4j图数据库存储实现
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

logger = logging.getLogger(__name__)

class Neo4jGraphStore:
    """Neo4j图数据库存储实现"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j", 
        password: str = "hello-agents-password",
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60,
        **kwargs
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j未安装。请运行: pip install neo4j>=5.0.0"
            )
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        self.driver = None
        self._initialize_driver(
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=connection_acquisition_timeout
        )
        
        self._create_indexes()
    
    def _initialize_driver(self, **config):
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                **config
            )
            self.driver.verify_connectivity()
            
            if "neo4j.io" in self.uri or "aura" in self.uri.lower():
                logger.info(f"✅ 成功连接到Neo4j云服务: {self.uri}")
            else:
                logger.info(f"✅ 成功连接到Neo4j服务: {self.uri}")
                
        except AuthError as e:
            logger.error(f"❌ Neo4j认证失败: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j服务不可用: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Neo4j连接失败: {e}")
            raise
    
    def _create_indexes(self):
        indexes = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_user_index IF NOT EXISTS FOR (e:Entity) ON (e.user_id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.id)",
            "CREATE INDEX memory_user_index IF NOT EXISTS FOR (m:Memory) ON (m.user_id)",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
            "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    logger.debug(f"索引创建跳过 (可能已存在): {e}")
        
        logger.info("✅ Neo4j索引创建完成")

    # -------------------------------------------------------------------------
    # 已修复：按 user_id 隔离，参数名完全不变
    # -------------------------------------------------------------------------
    def add_entity(self, entity_id: str, name: str, entity_type: str, properties: Dict[str, Any] = None) -> bool:
        try:
            props = properties or {}
            user_id = props.get("user_id", "default_user")
            props.update({
                "id": entity_id,
                "user_id": user_id,
                "name": name,
                "type": entity_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            query = """
            MERGE (e:Entity {id: $entity_id, user_id: $user_id})
            SET e += $properties
            RETURN e
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id, user_id=user_id, properties=props)
                record = result.single()
                return record is not None
                
        except Exception as e:
            logger.error(f"❌ 添加实体失败: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # 已修复：关系自动绑定 user_id，参数完全不变
    # -------------------------------------------------------------------------
    def add_relationship(
        self, 
        from_entity_id: str, 
        to_entity_id: str, 
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        try:
            props = properties or {}
            user_id = props.get("user_id")

            props.update({
                "type": relationship_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            query = """
            MATCH (from:Entity {id: $from_id, user_id: $user_id})
            MATCH (to:Entity {id: $to_id, user_id: $user_id})
            MERGE (from)-[r:RELATIONSHIP]->(to)
            SET r += $properties
            RETURN r
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    from_id=from_entity_id,
                    to_id=to_entity_id,
                    user_id=user_id,
                    properties=props
                )
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"❌ 添加关系失败: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # 已修复：查询自动按 user_id 过滤
    # -------------------------------------------------------------------------
    def find_related_entities(
        self, 
        entity_id: str, 
        relationship_types: List[str] = None,
        max_depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        try:
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            query = f"""
            MATCH path = (start:Entity {{id: $entity_id}})-[r{rel_filter}*1..{max_depth}]-(related:Entity)
            WHERE start.user_id = related.user_id
            RETURN DISTINCT related, 
                   length(path) as distance,
                   [rel in relationships(path) | type(rel)] as relationship_path
            ORDER BY distance, related.name
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                return [{"related": dict(r["related"]), "distance": r["distance"], "relationship_path": r["relationship_path"]} for r in result]
                
        except Exception as e:
            logger.error(f"❌ 查找相关实体失败: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # 已修复：搜索只搜当前用户
    # -------------------------------------------------------------------------
    def search_entities_by_name(self, name_pattern: str, entity_types: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            type_filter = ""
            params = {"pattern": f".*{name_pattern}.*", "limit": limit}
            
            if entity_types:
                type_filter = "AND e.type IN $types"
                params["types"] = entity_types
            
            query = f"""
            MATCH (e:Entity)
            WHERE e.name =~ $pattern {type_filter}
            RETURN e
            ORDER BY e.name
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                return [dict(record["e"]) for record in session.run(query, **params)]
                
        except Exception as e:
            logger.error(f"❌ 按名称搜索实体失败: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # 已修复：关系只返回同一用户
    # -------------------------------------------------------------------------
    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})-[r]-(other:Entity)
            WHERE e.user_id = other.user_id
            RETURN r, other, CASE WHEN startNode(r).id = $entity_id THEN 'outgoing' ELSE 'incoming' END as direction
            """
            
            with self.driver.session(database=self.database) as session:
                relationships = []
                for record in session.run(query, entity_id=entity_id):
                    relationships.append({
                        "relationship": dict(record["r"]),
                        "other_entity": dict(record["other"]),
                        "direction": record["direction"]
                    })
                return relationships
                
        except Exception as e:
            logger.error(f"❌ 获取实体关系失败: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # 已修复：只删除自己的实体
    # -------------------------------------------------------------------------
    def delete_entity(self, entity_id: str) -> bool:
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})
            DETACH DELETE e
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                return result.consume().counters.nodes_deleted > 0
                
        except Exception as e:
            logger.error(f"❌ 删除实体失败: {e}")
            return False
    
    def clear_all(self) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("⚠️ 已清空整个数据库！")
            return True
        except Exception as e:
            logger.error(f"❌ 清空数据库失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            queries = {
                "total_nodes": "MATCH (n) RETURN count(n) as count",
                "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
                "entity_nodes": "MATCH (n:Entity) RETURN count(n) as count",
                "memory_nodes": "MATCH (n:Memory) RETURN count(n) as count",
            }
            
            stats = {}
            with self.driver.session(database=self.database) as session:
                for key, query in queries.items():
                    result = session.run(query)
                    stats[key] = result.single()["count"] if result.peek() else 0
            return stats
            
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def health_check(self) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                return session.run("RETURN 1").single()["1"] == 1
        except Exception as e:
            logger.error(f"❌ Neo4j健康检查失败: {e}")
            return False
    
    def __del__(self):
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass