from typing import List
from neo4j import GraphDatabase

# ======================【配置】======================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123456"
# ====================================================

# ======================【全局公用驱动】======================
_NEO4J_DRIVER = None

def get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        _NEO4J_DRIVER = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _NEO4J_DRIVER

def close_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER:
        _NEO4J_DRIVER.close()
        _NEO4J_DRIVER = None

# ====================================================

# ======================【全局 UIE 模型单例】======================
# ✅ 全局只加载一次！！！
_UIE_MODEL = None

def get_uie_model():
    global _UIE_MODEL
    if _UIE_MODEL is None:
        from paddlenlp import Taskflow
        schema = [
            "姓名", "性别", "出生日期", "出生地", "民族", "国籍", "政治面貌",
            "学历", "学位", "毕业院校", "职务", "职称", "工作单位",
            "荣誉称号", "研究领域", "联系方式", "电子邮箱",
            {
                "人物": ["曾任职单位", "现任职务", "主要成就", "社会兼职", "家庭成员"],
                "工作经历": ["开始时间", "结束时间", "单位名称", "担任职务"],
                "教育经历": ["入学时间", "毕业时间", "学校名称", "专业", "学历"],
                "荣誉奖项": ["获得时间", "奖项名称", "颁发单位"]
            }
        ]
        print("🔸 加载 UIE 模型（仅第一次执行）...")
        _UIE_MODEL = Taskflow(
            "information_extraction",
            model="uie-base",
            schema=schema,
            is_static=False,
            device_id=-1
        )
    return _UIE_MODEL

# ================================================================




class Neo4jKGRAG_Enterprise:
    def __init__(self, user_id="default"):
        self.driver = get_neo4j_driver()
        self.user_id = user_id

    def extract(self, text):
        """🔥 第一次调用时才加载模型，多进程100%安全"""
        uie = get_uie_model()
        return uie(text)

    # -------------------------------------------------------------------------
    # ✅ 修复：解析 UIE 输出 → 实体 + 关系（修复关系解析逻辑）
    # -------------------------------------------------------------------------
    def parse_uie_result(self, uie_output, min_prob=0.5):
        entities = []
        relations = []
        seen_entities = set()  # 只追踪实体去重，不追踪关系

        if not uie_output:
            return {"entities": entities, "relations": relations}

        res = uie_output[0]

        for ent_type, item_list in res.items():
            if not isinstance(item_list, list):
                continue

            for item in item_list:
                val = item.get("text", "").strip()
                prob = item.get("probability", 1.0)

                # 过滤空值和低置信度
                if not val or prob < min_prob:
                    continue

                # 实体去重
                entity_key = (ent_type, val)
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities.append({
                        "name": val,
                        "type": ent_type,
                        "prob": round(float(prob), 4)
                    })

                # 抽取关系（修复：关系目标是实体，关系名不是实体类型）
                relations_dict = item.get("relations", {})
                subject = val  # 当前实体作为关系主体

                for rel_name, targets in relations_dict.items():
                    for tar in targets:
                        obj_val = tar.get("text", "").strip()
                        obj_prob = tar.get("probability", 1.0)
                        obj_type = tar.get("type", "")  # 目标实体类型

                        if not obj_val or obj_prob < min_prob:
                            continue

                        # 目标实体去重入库
                        obj_entity_key = (obj_type, obj_val)
                        if obj_entity_key not in seen_entities:
                            seen_entities.add(obj_entity_key)
                            entities.append({
                                "name": obj_val,
                                "type": obj_type,
                                "prob": round(float(obj_prob), 4)
                            })

                        # 关系（统一大写，避免重复）
                        relations.append({
                            "subject": subject,
                            "predicate": rel_name.upper(),
                            "object": obj_val
                        })

        return {"entities": entities, "relations": relations}

    # -------------------------------------------------------------------------
    # ✅ 修复：安全写入关系（无注入，Neo4j标准语法）
    # -------------------------------------------------------------------------
    def write_chunk_to_kg(self, chunk):
        try:
            chunk_id = chunk["id"]
            content = chunk["content"]
            meta = chunk["metadata"]
            doc_id = meta.get("doc_id", "unknown")
            source_path = meta.get("source_path", "")

            # UIE 信息抽取
            uie_out = self.uie(content)
            kg = self.parse_uie_result(uie_out, min_prob=0.5)

            with self.driver.session() as session:
                # 1. 写入文档节点
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.source_path = $source_path,
                        d.user_id = $user_id,
                        d.updated_at = timestamp()
                """, doc_id=doc_id, source_path=source_path, user_id=self.user_id)

                # 2. 写入文本分块
                session.run("""
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.content = left($content, 500),
                        c.doc_id = $doc_id,
                        c.user_id = $user_id,
                        c.updated_at = timestamp()
                """, chunk_id=chunk_id, content=content, doc_id=doc_id, user_id=self.user_id)

                # 3. 文档-分块关系
                session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, doc_id=doc_id, chunk_id=chunk_id)

                # 4. 写入实体
                for ent in kg["entities"]:
                    ent_type = ent["type"]
                    ent_name = ent["name"]
                    ent_prob = ent["prob"]

                    session.run("""
                        MERGE (e:Entity {type: $type, name: $name})
                        SET e.probability = $prob,
                            e.updated_at = timestamp()
                    """, type=ent_type, name=ent_name, prob=ent_prob)

                    # 分块-实体关系
                    session.run("""
                        MATCH (c:Chunk {chunk_id: $cid})
                        MATCH (e:Entity {type: $type, name: $name})
                        MERGE (c)-[:HAS_ENTITY]->(e)
                    """, cid=chunk_id, type=ent_type, name=ent_name)

                # 5. 修复：安全写入实体关系（无SQL注入，Neo4j官方推荐写法）
                for rel in kg["relations"]:
                    session.run("""
                        MATCH (s:Entity {name: $sub})
                        MATCH (o:Entity {name: $obj})
                        CALL apoc.create.relationship(s, $rel_type, {updated_at: timestamp()}, o)
                        YIELD rel
                        RETURN rel
                    """, sub=rel["subject"], obj=rel["object"], rel_type=rel["predicate"])

            print(f"✅ chunk={chunk_id} 实体:{len(kg['entities'])} 关系:{len(kg['relations'])}")
        
        except Exception as e:
            print(f"❌ 写入分块 {chunk.get('id','unknown')} 失败：{str(e)}")

    # -------------------------------------------------------------------------
    # ✅ 批量写入
    # -------------------------------------------------------------------------
    def add_neo4j_document(self, chunks: List[dict]):
        if not chunks:
            return False

        # 1. 全量抽取
        entities, relations = self.extract_all_chunks(chunks)

        # 2. 全局清洗（去重、消歧、融合）
        entities_clean, relations_clean = self.merge_kg_global(entities, relations)

        # 3. 批量写入
        self.batch_write_to_kg(chunks, entities_clean, relations_clean)

        return True
    
    # ========================= 【新增：全局知识合并与清洗】 =========================
    def extract_all_chunks(self, chunks):
        """批量抽取所有块，返回全局实体、关系列表"""
        global_entities = []
        global_relations = []

        for chunk in chunks:
            try:
                content = chunk["content"]
                uie_out = self.extract(content)
                kg = self.parse_uie_result(uie_out, min_prob=0.5)

                # 带上 chunk_id 用于溯源
                for e in kg["entities"]:
                    e["chunk_id"] = chunk["id"]
                for r in kg["relations"]:
                    r["chunk_id"] = chunk["id"]

                global_entities.extend(kg["entities"])
                global_relations.extend(kg["relations"])

            except Exception as e:
                print(f"⚠️  分块 {chunk.get('id')} 抽取失败：{e}")

        return global_entities, global_relations

    def merge_kg_global(self, entities, relations):
        """
        【核心】全局去重、消歧、合并
        1. 实体唯一：type + name 作为唯一键
        2. 保留最高置信度
        3. 关系去重
        """
        # -------- 实体去重 + 置信度融合 --------
        entity_map = {}
        for ent in entities:
            key = (ent["type"], ent["name"])
            if key not in entity_map or ent["prob"] > entity_map[key]["prob"]:
                entity_map[key] = ent

        # -------- 关系去重：主体-谓词-客体 --------
        rel_set = set()
        merged_rels = []
        for rel in relations:
            key = (rel["subject"], rel["predicate"], rel["object"])
            if key not in rel_set:
                rel_set.add(key)
                merged_rels.append(rel)

        return list(entity_map.values()), merged_rels

    # ========================= 【批量写入：超快、干净】 =========================
    def batch_write_to_kg(self, chunks, entities, relations):
        """一次性批量写入，Neo4j 最快方式"""
        try:
            with self.driver.session() as session:
                # -------------------- 1. 写入文档、分块（不变） --------------------
                if not chunks:
                    return
                meta = chunks[0]["metadata"]
                doc_id = meta.get("doc_id", "unknown")
                source_path = meta.get("source_path", "")

                # 文档
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.source_path = $sp, d.user_id = $uid, d.updated_at = timestamp()
                """, doc_id=doc_id, sp=source_path, uid=self.user_id)

                # 分块
                for chunk in chunks:
                    session.run("""
                        MERGE (c:Chunk {chunk_id: $cid})
                        SET c.content = left($content,500), c.doc_id=$did, c.user_id=$uid, c.updated_at=timestamp()
                    """, cid=chunk["id"], content=chunk["content"], did=doc_id, uid=self.user_id)

                    session.run("""
                        MATCH (d:Document {doc_id:$did}),(c:Chunk {chunk_id:$cid})
                        MERGE (d)-[:HAS_CHUNK]->(c)
                    """, did=doc_id, cid=chunk["id"])

                # -------------------- 2. 批量写入实体（全局唯一） --------------------
                for ent in entities:
                    session.run("""
                        MERGE (e:Entity {type:$t, name:$n})
                        SET e.probability=$p, e.updated_at=timestamp()
                    """, t=ent["type"], n=ent["name"], p=ent["prob"])

                    # 关联分块
                    session.run("""
                        MATCH (c:Chunk {chunk_id:$cid}), (e:Entity {type:$t,name:$n})
                        MERGE (c)-[:HAS_ENTITY]->(e)
                    """, cid=ent["chunk_id"], t=ent["type"], n=ent["name"])

                # -------------------- 3. 批量写入关系（全局去重后） --------------------
                for rel in relations:
                    session.run("""
                        MATCH (s:Entity {name:$sub}), (o:Entity {name:$obj})
                        CALL apoc.create.relationship(s, $rel, {updated_at:timestamp()}, o)
                        YIELD rel RETURN rel
                    """, sub=rel["subject"], obj=rel["object"], rel=rel["predicate"])

            print(f"✅ 全局入库完成：实体={len(entities)}, 关系={len(relations)}")

        except Exception as e:
            print(f"❌ 批量写入失败：{str(e)}")

    # -------------------------------------------------------------------------
    # ✅ 修复：关闭连接
    # -------------------------------------------------------------------------
    def close(self):
        close_neo4j_driver()