"""语义记忆实现

结合向量检索和知识图谱的混合语义记忆，使用：
- HuggingFace 中文预训练模型进行文本嵌入
- 向量相似度检索进行快速初筛
- 知识图谱进行实体关系推理
- 混合检索策略优化结果质量
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import logging
import math
import numpy as np

from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..embedding import get_text_embedder, get_dimension


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Entity:
    """实体类"""
    
    def __init__(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "MISC",
        description: str = "",
        properties: Dict[str, Any] = None
    ):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type  # PERSON, ORG, PRODUCT, SKILL, CONCEPT等
        self.description = description
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1  # 出现频率
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
            "frequency": self.frequency
        }


class Relation:
    """关系类"""
    
    def __init__(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        strength: float = 1.0,
        evidence: str = "",
        properties: Dict[str, Any] = None
    ):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence  # 支持该关系的原文本
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.frequency = 1  # 关系出现频率
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "properties": self.properties,
            "frequency": self.frequency
        }


class SemanticMemory(BaseMemory):
    """增强语义记忆实现（企业级知识库·只按用户隔离）
    
    特点：
    - 使用HuggingFace中文预训练模型进行文本嵌入
    - 向量检索进行快速相似度匹配
    - 知识图谱存储实体和关系
    - 混合检索策略：向量+图+语义推理
    - 知识库：只按 user_id 隔离，不绑定 agent_id / session_id
    """
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 嵌入模型（统一提供）
        self.embedding_model = None
        self._init_embedding_model()
        
        # 专业数据库存储
        self.vector_store = None
        self.graph_store = None
        self._init_databases()
        
        # 实体和关系缓存 (用于快速访问)
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        
        # 实体识别器
        self.nlp = None
        self._init_nlp()
        
        # 记忆存储
        self.semantic_memories: List[MemoryItem] = []
        self.memory_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info("增强语义记忆初始化完成（使用Qdrant+Neo4j专业数据库）")
    
    def _init_embedding_model(self):
        """初始化统一嵌入模型（由 embedding_provider 管理）。"""
        try:
            self.embedding_model = get_text_embedder()
            # 轻量健康检查与日志
            try:
                test_vec = self.embedding_model.encode("health_check")
                dim = getattr(self.embedding_model, "dimension", len(test_vec))
                logger.info(f"✅ 嵌入模型就绪，维度: {dim}")
            except Exception:
                logger.info("✅ 嵌入模型就绪")
        except Exception as e:
            logger.error(f"❌ 嵌入模型初始化失败: {e}")
            raise
    
    def _init_databases(self):
        """初始化专业数据库存储"""
        try:
            from ...core.database_config import get_database_config
            # 获取数据库配置
            db_config = get_database_config()
            
            # 初始化Qdrant向量数据库（使用连接管理器避免重复连接）
            from ..storage.qdrant_store import QdrantConnectionManager
            qdrant_config = db_config.get_qdrant_config() or {}
            qdrant_config["vector_size"] = get_dimension()
            self.vector_store = QdrantConnectionManager.get_instance(**qdrant_config)
            logger.info("✅ Qdrant向量数据库初始化完成")
            
            # 初始化Neo4j图数据库
            from ..storage.neo4j_store import Neo4jGraphStore
            neo4j_config = db_config.get_neo4j_config()
            self.graph_store = Neo4jGraphStore(**neo4j_config)
            logger.info("✅ Neo4j图数据库初始化完成")
            
            # 验证连接
            vector_health = self.vector_store.health_check()
            graph_health = self.graph_store.health_check()
            
            if not vector_health:
                logger.warning("⚠️ Qdrant连接异常，部分功能可能受限")
            if not graph_health:
                logger.warning("⚠️ Neo4j连接异常，图搜索功能可能受限")
            
            logger.info(f"🏥 数据库健康状态: Qdrant={'✅' if vector_health else '❌'}, Neo4j={'✅' if graph_health else '❌'}")
            
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
            logger.info("💡 请检查数据库配置和网络连接")
            logger.info("💡 参考 DATABASE_SETUP_GUIDE.md 进行配置")
            raise
    
    def _init_nlp(self):
        """初始化NLP处理器 - 智能多语言支持"""
        try:
            import spacy
            self.nlp_models = {}
            
            # 尝试加载多语言模型
            models_to_try = [
                ("zh_core_web_sm", "中文"),
                ("en_core_web_sm", "英文")
            ]
            
            loaded_models = []
            for model_name, lang_name in models_to_try:
                try:
                    nlp = spacy.load(model_name)
                    self.nlp_models[model_name] = nlp
                    loaded_models.append(lang_name)
                    logger.info(f"✅ 加载{lang_name}spaCy模型: {model_name}")
                except OSError:
                    logger.warning(f"⚠️ {lang_name}spaCy模型不可用: {model_name}")
            
            # 设置主要NLP处理器
            if "zh_core_web_sm" in self.nlp_models:
                self.nlp = self.nlp_models["zh_core_web_sm"]
                logger.info("🎯 主要使用中文spaCy模型")
            elif "en_core_web_sm" in self.nlp_models:
                self.nlp = self.nlp_models["en_core_web_sm"]
                logger.info("🎯 主要使用英文spaCy模型")
            else:
                self.nlp = None
                logger.warning("⚠️ 无可用spaCy模型，实体提取将受限")
            
            if loaded_models:
                logger.info(f"📚 可用语言模型: {', '.join(loaded_models)}")
                
        except ImportError:
            logger.warning("⚠️ spaCy不可用，实体提取将受限")
            self.nlp = None
            self.nlp_models = {}
    
    # =========================================================================
    # 🔹 核心：添加语义记忆（只存 user_id，不存 session_id）
    # =========================================================================
    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆（用户喜好）"""
        try:
            user_id = memory_item.user_id

            # 1. 生成文本嵌入
            embedding = self.embedding_model.encode(memory_item.content)
            self.memory_embeddings[memory_item.id] = embedding
            
            # 2. 提取实体和关系
            entities = self._extract_entities(memory_item.content, user_id)
            relations = self._extract_relations(memory_item.content, entities)
            
            # 3. 存储到Neo4j图数据库
            for entity in entities:
                self._add_entity_to_graph(entity, memory_item)
            
            for relation in relations:
                self._add_relation_to_graph(relation, memory_item)
            
            # 4. 存储到Qdrant向量数据库（只存 user_id）
            metadata = {
                "memory_id": memory_item.id,
                "user_id": user_id,
                "content": memory_item.content,
                "memory_type": "semantic",
                "timestamp": int(memory_item.timestamp.timestamp()),
                "importance": memory_item.importance,
                "entities": [e.entity_id for e in entities],
                "entity_count": len(entities),
                "relation_count": len(relations)
            }
            
            success = self.vector_store.add_vectors(
                vectors=[embedding.tolist()],
                metadata=[metadata],
                ids=[memory_item.id]
            )
            
            if not success:
                logger.warning("⚠️ 向量存储失败，但记忆已添加到图数据库")
            
            # 5. 添加实体信息到元数据（不存 session_id）
            memory_item.metadata["entities"] = [e.entity_id for e in entities]
            memory_item.metadata["relations"] = [
                f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations
            ]
            
            # 6. 存储记忆
            self.semantic_memories.append(memory_item)
            
            logger.info(f"✅ 添加语义记忆: {len(entities)}个实体, {len(relations)}个关系")
            return memory_item.id
        
        except Exception as e:
            logger.error(f"❌ 添加语义记忆失败: {e}")
            raise
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆（只按 user_id 过滤，忽略 session_id）"""
        try:
            user_id = kwargs.get("user_id")

            # 1. 向量检索
            vector_results = self._vector_search(query, limit * 2, user_id)
            
            # 2. 图检索
            graph_results = self._graph_search(query, limit * 2, user_id)
            
            # 3. 混合排序
            combined_results = self._combine_and_rank_results(
                vector_results, graph_results, query, limit
            )

            scores = [r.get("combined_score", r.get("vector_score", 0.0)) for r in combined_results]
            if scores:
                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                denom = sum(exps) or 1.0
                probs = [e / denom for e in exps]
            else:
                probs = []
            
            result_memories = []
            for idx, result in enumerate(combined_results):
                memory_id = result.get("memory_id")
                
                memory = next((m for m in self.semantic_memories if m.id == memory_id), None)
                if memory and memory.metadata.get("forgotten", False):
                    continue
                
                timestamp = result.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except ValueError:
                        timestamp = datetime.now()
                elif isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp)
                else:
                    timestamp = datetime.now()
                
                memory_item = MemoryItem(
                    id=result["memory_id"],
                    content=result["content"],
                    memory_type="semantic",
                    user_id=result.get("user_id", "default"),
                    timestamp=timestamp,
                    importance=result.get("importance", 0.5),
                    metadata={
                        **result.get("metadata", {}),
                        "combined_score": result.get("combined_score", 0.0),
                        "probability": probs[idx] if idx < len(probs) else 0.0,
                    }
                )
                result_memories.append(memory_item)
            
            logger.info(f"✅ 检索到 {len(result_memories)} 条相关记忆")
            return result_memories[:limit]
                
        except Exception as e:
            logger.error(f"❌ 检索语义记忆失败: {e}")
            return []
    
    def _vector_search(self, query: str, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedding_model.encode(query)
            where_filter = {"memory_type": "semantic"}
            if user_id:
                where_filter["user_id"] = user_id

            results = self.vector_store.search_similar(
                query_vector=query_embedding.tolist(),
                limit=limit,
                where=where_filter
            )

            formatted = []
            for r in results:
                formatted.append({
                    "id": r["id"],
                    "score": r["score"],
                    **r["metadata"]
                })
            return formatted
        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def _graph_search(self, query: str, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            query_entities = self._extract_entities(query,user_id)
            if not query_entities:
                return []

            related_memory_ids = set()
            for ent in query_entities:
                try:
                    related = self.graph_store.find_related_entities(
                        entity_id=ent.entity_id, max_depth=2, limit=20
                    )
                    for item in related:
                        if "memory_id" in item:
                            related_memory_ids.add(item["memory_id"])
                except:
                    continue

            results = []
            for mid in list(related_memory_ids)[:limit*2]:
                mem = self._find_memory_by_id(mid)
                if not mem: continue
                if user_id and mem.user_id != user_id: continue

                graph_score = self._calculate_graph_relevance_neo4j({
                    "entities": mem.metadata.get("entities", [])
                }, query_entities)

                results.append({
                    "memory_id": mid,
                    "content": mem.content,
                    "user_id": mem.user_id,
                    "similarity": graph_score,
                    "importance": mem.importance,
                    "timestamp": int(mem.timestamp.timestamp()),
                })
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"❌ 图搜索失败: {e}")
            return []

    def _combine_and_rank_results(self, vector_results, graph_results, query, limit):
        combined = {}
        content_seen = set()
        
        for r in vector_results:
            mid = r["memory_id"]
            c = r.get("content", "")
            h = hash(c.strip())
            if h in content_seen: continue
            content_seen.add(h)
            combined[mid] = {**r, "vector_score": r.get("score", 0), "graph_score": 0.0}
        
        for r in graph_results:
            mid = r["memory_id"]
            c = r.get("content", "")
            h = hash(c.strip())
            if mid in combined:
                combined[mid]["graph_score"] = r.get("similarity", 0)
            elif h not in content_seen:
                content_seen.add(h)
                combined[mid] = {**r, "vector_score": 0.0, "graph_score": r.get("similarity", 0)}
        
        for k, v in combined.items():
            vs = v["vector_score"]
            gs = v["graph_score"]
            imp = v.get("importance", 0.5)
            base = vs * 0.7 + gs * 0.3
            weight = 0.8 + imp * 0.4
            v["combined_score"] = base * weight
        
        filtered = [x for x in combined.values() if x["combined_score"] >= 0.1]
        filtered.sort(key=lambda x: x["combined_score"], reverse=True)
        return filtered[:limit]

    def _detect_language(self, text: str) -> str:
        chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        total = len(text.replace(' ', ''))
        if total == 0:
            return "en"
        return "zh" if (chinese_chars / total) > 0.3 else "en"

    def _extract_entities(self, text: str, user_id: str) -> List[Entity]:
        entities = []
        lang = self._detect_language(text)
        selected_nlp = None
        if lang == "zh" and "zh_core_web_sm" in self.nlp_models:
            selected_nlp = self.nlp_models["zh_core_web_sm"]
        elif lang == "en" and "en_core_web_sm" in self.nlp_models:
            selected_nlp = self.nlp_models["en_core_web_sm"]
        else:
            selected_nlp = self.nlp
        
        if selected_nlp:
            try:
                doc = selected_nlp(text)
                self._store_linguistic_analysis(doc, text, user_id)
                for ent in doc.ents:
                    entity = Entity(
                        entity_id=f"entity_{hash(ent.text)}",
                        name=ent.text,
                        entity_type=ent.label_,
                        description=f"从文本中识别的{ent.label_}实体"
                    )
                    entities.append(entity)
            except Exception as e:
                logger.warning(f"⚠️ spaCy实体识别失败: {e}")
        return entities

    def _store_linguistic_analysis(self, doc, text: str, user_id: str):
        if not self.graph_store:
            return
        try:
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                token_id = f"token_{hash(token.text + token.pos_)}"
                self.graph_store.add_entity(
                    entity_id=token_id,
                    name=token.text,
                    entity_type="TOKEN",
                    properties={
                        "pos": token.pos_,
                        "tag": token.tag_,
                        "lemma": token.lemma_,
                        "source_text": text[:50],
                        "language": self._detect_language(text),
                        "user_id": user_id
                    }
                )
                if token.pos_ in ["NOUN", "PROPN"]:
                    concept_id = f"concept_{hash(token.text)}"
                    self.graph_store.add_entity(
                        entity_id=concept_id,
                        name=token.text,
                        entity_type="CONCEPT",
                        properties={"category": token.pos_, "source_text": text[:50], "user_id": user_id}
                    )
                    self.graph_store.add_relationship(
                        from_entity_id=token_id,
                        to_entity_id=concept_id,
                        relationship_type="REPRESENTS",
                        properties={"confidence": 1.0, "user_id": user_id}
                    )
            for token in doc:
                if token.is_punct or token.is_space or token.head == token:
                    continue
                from_id = f"token_{hash(token.text + token.pos_)}"
                to_id = f"token_{hash(token.head.text + token.head.pos_)}"
                rel_type = token.dep_.upper().replace(":", "_")
                self.graph_store.add_relationship(
                    from_entity_id=from_id,
                    to_entity_id=to_id,
                    relationship_type=rel_type,
                    properties={"dependency": token.dep_, "source_text": text[:50], "user_id": user_id}
                )
        except Exception as e:
            logger.warning(f"⚠️ 存储词法分析失败: {e}")
    
    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        relations = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                relations.append(Relation(
                    from_entity=e1.entity_id,
                    to_entity=e2.entity_id,
                    relation_type="CO_OCCURS",
                    strength=0.5,
                    evidence=text[:100]
                ))
        return relations
    
    def _add_entity_to_graph(self, entity: Entity, memory_item: MemoryItem):
        try:
            props = {
                "name": entity.name,
                "description": entity.description,
                "frequency": entity.frequency,
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "importance": memory_item.importance,
                **entity.properties
            }
            success = self.graph_store.add_entity(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                properties=props
            )
            if success:
                if entity.entity_id in self.entities:
                    self.entities[entity.entity_id].frequency +=1
                else:
                    self.entities[entity.entity_id] = entity
            return success
        except:
            return False

    def _add_relation_to_graph(self, relation: Relation, memory_item: MemoryItem):
        try:
            props = {
                "strength": relation.strength,
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "importance": memory_item.importance,
                "evidence": relation.evidence
            }
            success = self.graph_store.add_relationship(
                from_entity_id=relation.from_entity,
                to_entity_id=relation.to_entity,
                relationship_type=relation.relation_type,
                properties=props
            )
            if success:
                self.relations.append(relation)
            return success
        except:
            return False

    def _calculate_graph_relevance_neo4j(self, memory_metadata: Dict[str, Any], query_entities: List[Entity]) -> float:
        """计算Neo4j图相关性分数"""
        try:
            memory_entities = memory_metadata.get("entities", [])
            if not memory_entities or not query_entities:
                return 0.0
            
            query_entity_ids = {e.entity_id for e in query_entities}
            matching_entities = len(set(memory_entities).intersection(query_entity_ids))
            entity_score = matching_entities / len(query_entity_ids) if query_entity_ids else 0
            
            entity_count = memory_metadata.get("entity_count", 0)
            entity_density = min(entity_count / 10, 1.0)
            
            relation_count = memory_metadata.get("relation_count", 0)
            relation_density = min(relation_count / 5, 1.0)
            
            relevance_score = (
                entity_score * 0.6 +
                entity_density * 0.2 +
                relation_density * 0.2
            )
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.debug(f"计算图相关性失败: {e}")
            return 0.0

    def _find_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        for m in self.semantic_memories:
            if m.id == memory_id:
                return m
        return None

    # =========================================================================
    # 🔹 所有操作：只校验 user_id，忽略 agent_id / session_id
    # =========================================================================
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        mem = self._find_memory_by_id(memory_id)
        if not mem: return False

        try:
            if content:
                emb = self.embedding_model.encode(content)
                self.memory_embeddings[memory_id] = emb
                new_ents = self._extract_entities(content)
                new_rels = self._extract_relations(content, new_ents)
                mem.content = content
                mem.metadata["entities"] = [e.entity_id for e in new_ents]
                mem.metadata["relations"] = [f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in new_rels]
            
            if importance is not None:
                mem.importance = importance
            if metadata:
                mem.metadata.update(metadata)
            return True
        except:
            return False

    def remove(self, memory_id: str) -> bool:
        mem = self._find_memory_by_id(memory_id)
        if not mem: return False

        try:
            self.vector_store.delete_memories([memory_id])
            self.semantic_memories.remove(mem)
            if memory_id in self.memory_embeddings:
                del self.memory_embeddings[memory_id]
            return True
        except:
            return False

    def has_memory(self, memory_id: str) -> bool:
        m = self._find_memory_by_id(memory_id)
        return m is not None

    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
        **kwargs
    ) -> int:
        user_id = kwargs.get("user_id")
        now = datetime.now()
        to_del = []
        for m in self.semantic_memories:
            if user_id and m.user_id != user_id:
                continue

            should_forget = False
            if strategy == "importance_based" and m.importance < threshold:
                should_forget = True
            elif strategy == "time_based":
                if m.timestamp < now - timedelta(days=max_age_days):
                    should_forget = True
            
            if should_forget:
                to_del.append(m.id)
        
        cnt = 0
        for mid in list(set(to_del)):
            if self.remove(mid):
                cnt += 1
        return cnt

    def clear(self, **kwargs):
        user_id = kwargs.get("user_id")
        to_del = [m.id for m in self.semantic_memories if not user_id or m.user_id == user_id]
        for mid in to_del:
            self.remove(mid)

    def get_all(self, **kwargs) -> List[MemoryItem]:
        user_id = kwargs.get("user_id")
        return [m for m in self.semantic_memories if not user_id or m.user_id == user_id]

    def get_stats(self, **kwargs) -> Dict[str, Any]:
        user_id = kwargs.get("user_id")
        filtered = self.get_all(user_id=user_id)
        return {
            "count": len(filtered),
            "entities_count": len(self.entities),
            "relations_count": len(self.relations),
            "avg_importance": sum(m.importance for m in filtered)/len(filtered) if filtered else 0.0,
            "memory_type": "semantic"
        }

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        query_lower = query.lower()
        scored_entities = []
        
        for entity in self.entities.values():
            score = 0.0
            
            if query_lower in entity.name.lower():
                score += 2.0
            if query_lower in entity.entity_type.lower():
                score += 1.0
            if query_lower in entity.description.lower():
                score += 0.5
            
            score *= math.log(1 + entity.frequency)
            
            if score > 0:
                scored_entities.append((score, entity))
        
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        return [entity for _, entity in scored_entities[:limit]]
    
    def get_related_entities(
        self,
        entity_id: str,
        relation_types: List[str] = None,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        return []