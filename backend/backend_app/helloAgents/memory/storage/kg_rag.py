from datetime import datetime
from typing import List
from neo4j import GraphDatabase
from helloAgents.core.llm import HelloAgentsLLM
import re
import hashlib
import warnings
warnings.filterwarnings("ignore")

# ======================【配置】======================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123456"
# ====================================================

# ======================【全局统一 Schema：存 + 取 完全一致】======================
# 1. 实体类型（UIE抽取 + 数据库type字段 + 查询过滤）
ENTITY_TYPES = ["人物", "门派", "武功", "兵器", "宝物", "地点", "事件"]

# 2. 支持的关系（UIE抽取 + 数据库r.type + 查询过滤）
RELATION_TYPES = [
    "师徒", "父子", "母子", "父女", "母女", "夫妻", "恋人", "爱慕",
    "兄弟", "义兄弟", "姐妹", "仇敌", "盟友", "上下级", "同门", "救命恩人",
    "创立", "掌门", "属于", "持有", "修炼", "精通", "自创",
    "出生地", "常驻", "发生于", "参与", "发起", "受害者",
    "敌对", "结盟", "铸造", "抢夺", "赠予", "毁坏", "成对", "互毁",
    "秘藏", "归属", "围攻", "兼并", "见证", "隐居地", "到访", "被困"
]

# 3. UIE 抽取专用 Schema（中文，用于抽取信息）
UIE_SCHEMA = [
    {"人物": RELATION_TYPES},
    {"门派": RELATION_TYPES},
    {"武功": RELATION_TYPES},
    {"兵器": RELATION_TYPES},
    {"宝物": RELATION_TYPES},
    {"地点": RELATION_TYPES},
    {"事件": RELATION_TYPES}
]

# 4. Neo4j 物理存储结构（固定，查询必须遵守）
NEO4J_STRUCTURE = {
    "NODE_LABEL": "Entity",          # 节点标签：Entity
    "PROP_NAME": "name",              # 实体名称：name
    "PROP_TYPE": "type",              # 实体类型：type
    "REL_TYPE": "RELATION",           # 关系类型：RELATION
    "REL_PROP_TYPE": "type"           # 关系名称存在属性：type
}
# ================================================================================

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

# ======================【全局 UIE 模型单例】======================
_UIE_MODEL = None

def get_uie_model():
    global _UIE_MODEL
    if _UIE_MODEL is None:
        from paddlenlp import Taskflow
        print("🔸 加载 UIE 模型（武侠知识抽取）...")
        _UIE_MODEL = Taskflow(
            "information_extraction",
            model="uie-nano",
            schema=UIE_SCHEMA,
            is_static=False,
            device_id=-1
        )
    return _UIE_MODEL

# ================================================================

class Neo4jKGRAG_Enterprise:
    def __init__(self, user_id="default"):
        self.driver = get_neo4j_driver()
        self.user_id = user_id
        self.STRUCT = NEO4J_STRUCTURE

    # ========================= 【实体关系抽取】 =========================
    def extract(self, text):
        return get_uie_model()(text)

    def parse_uie_result(self, uie_output, min_prob=0.5):
        entities = []
        relations = []
        seen_entities = set()

        if not uie_output:
            return {"entities": entities, "relations": relations}

        res = uie_output[0]
        for ent_type, item_list in res.items():
            if not isinstance(item_list, list):
                continue

            for item in item_list:
                val = item.get("text", "").strip()
                prob = item.get("probability", 1.0)
                if not val or prob < min_prob:
                    continue

                entity_key = (ent_type, val)
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities.append({
                        "name": val,
                        "type": ent_type,
                        "prob": round(float(prob), 4)
                    })

                relations_dict = item.get("relations", {})
                subject = val
                for rel_name, targets in relations_dict.items():
                    for tar in targets:
                        obj_val = tar.get("text", "").strip()
                        obj_prob = tar.get("probability", 1.0)
                        obj_type = tar.get("type", "")
                        if not obj_val or obj_prob < min_prob:
                            continue

                        obj_entity_key = (obj_type, obj_val)
                        if obj_entity_key not in seen_entities:
                            seen_entities.add(obj_entity_key)
                            entities.append({
                                "name": obj_val,
                                "type": obj_type,
                                "prob": round(float(obj_prob), 4)
                            })

                        relations.append({
                            "subject": subject,
                            "predicate": rel_name,
                            "object": obj_val
                        })

        return {"entities": entities, "relations": relations}

    # ========================= 【批量写入 KG】 =========================
    def batch_write_to_kg(self, chunks, entities, relations):
        try:
            with self.driver.session() as session:
                doc_id = "unknown"
                if chunks and "metadata" in chunks[0]:
                    doc_id = chunks[0]["metadata"].get("doc_id", "unknown")

                user_id = self.user_id
                now = datetime.now().isoformat()

                # 文档
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.user_id = $user_id, d.updated_at = $now
                """, doc_id=doc_id, user_id=user_id, now=now)

                # 分块
                for chunk in chunks:
                    chunk_id = chunk["id"]
                    content = chunk["content"]
                    session.run("""
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.content = $content, c.doc_id = $doc_id, c.user_id = $user_id, c.updated_at = $now
                    """, chunk_id=chunk_id, content=content, doc_id=doc_id, user_id=user_id, now=now)
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (d)-[:HAS_CHUNK]->(c)
                    """, doc_id=doc_id, chunk_id=chunk_id)

                # 实体
                for ent in entities:
                    name = ent["name"]
                    ent_type = ent["type"]
                    prob = ent.get("prob", 1.0)
                    chunk_id = ent.get("chunk_id")
                    entity_id = hashlib.md5(f"{ent_type}_{name}".encode()).hexdigest()[:16]

                    session.run(f"""
                        MERGE (e:{self.STRUCT['NODE_LABEL']} {{id: $entity_id}})
                        SET e.{self.STRUCT['PROP_NAME']} = $name,
                            e.{self.STRUCT['PROP_TYPE']} = $ent_type,
                            e.probability = $prob,
                            e.doc_id = $doc_id, e.user_id = $user_id, e.updated_at = $now
                    """, entity_id=entity_id, name=name, ent_type=ent_type, prob=prob,
                                doc_id=doc_id, user_id=user_id, now=now)
                    ent["id"] = entity_id

                    if chunk_id:
                        session.run(f"""
                            MATCH (c:Chunk {{chunk_id: $chunk_id}})
                            MATCH (e:{self.STRUCT['NODE_LABEL']} {{id: $entity_id}})
                            MERGE (c)-[:HAS_ENTITY]->(e)
                        """, chunk_id=chunk_id, entity_id=entity_id)

                # 关系
                for rel in relations:
                    sub = rel["subject"]
                    obj = rel["object"]
                    pred = rel["predicate"]
                    chunk_id = rel.get("chunk_id")

                    sub_id = next((e["id"] for e in entities if e["name"] == sub), None)
                    obj_id = next((e["id"] for e in entities if e["name"] == obj), None)
                    if not sub_id or not obj_id:
                        continue

                    session.run(f"""
                        MATCH (a:{self.STRUCT['NODE_LABEL']} {{id: $sub_id}})
                        MATCH (b:{self.STRUCT['NODE_LABEL']} {{id: $obj_id}})
                        MERGE (a)-[r:{self.STRUCT['REL_TYPE']}]->(b)
                        SET r.{self.STRUCT['REL_PROP_TYPE']} = $pred,
                            r.doc_id = $doc_id, r.user_id = $user_id, r.chunk_id = $chunk_id, r.updated_at = $now
                    """, sub_id=sub_id, obj_id=obj_id, pred=pred,
                                doc_id=doc_id, user_id=user_id, chunk_id=chunk_id, now=now)

            print(f"✅ KG 入库完成：文档={doc_id} 实体={len(entities)} 关系={len(relations)}")
        except Exception as e:
            print(f"❌ 写入失败：{str(e)}")

    # ========================= 【分块批量抽取】 =========================
    def extract_all_chunks(self, chunks):
        from tqdm import tqdm
        import gc
        gc.collect()
        uie = get_uie_model()
        entities_all, relations_all = [], []
        BATCH_SIZE = 10

        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="🔍 抽取实体关系"):
            batch = chunks[i:i+BATCH_SIZE]
            texts = [c["content"] for c in batch]
            outputs = uie(texts)

            for chunk, out in zip(batch, outputs):
                kg = self.parse_uie_result([out])
                for e in kg["entities"]: e["chunk_id"] = chunk["id"]
                for r in kg["relations"]: r["chunk_id"] = chunk["id"]
                entities_all.extend(kg["entities"])
                relations_all.extend(kg["relations"])
            gc.collect()
        return entities_all, relations_all

    # ========================= 【全局合并】 =========================
    def merge_kg_global(self, entities, relations):
        entity_map = {}
        for e in entities:
            key = (e["type"], e["name"])
            if key not in entity_map or e["prob"] > entity_map[key]["prob"]:
                entity_map[key] = e

        merged_entities = []
        for e in entity_map.values():
            e["id"] = hashlib.md5(f"{e['type']}_{e['name']}".encode()).hexdigest()[:16]
            merged_entities.append(e)

        rel_set = set()
        merged_rels = []
        for r in relations:
            key = (r["subject"], r["predicate"], r["object"])
            if key not in rel_set:
                rel_set.add(key)
                merged_rels.append(r)
        return merged_entities, merged_rels

    # ========================= 【文档入库入口】 =========================
    def add_neo4j_document(self, chunks: List[dict]):
        if not chunks: return False
        entities, relations = self.extract_all_chunks(chunks)
        entities_clean, relations_clean = self.merge_kg_global(entities, relations)
        self.batch_write_to_kg(chunks, entities_clean, relations_clean)
        return True

    # ========================= 【Cypher 生成（统一结构！）】 =========================
    def generate_cypher(self, question: str):
        prompt = f"""
你是严格的Neo4j Cypher生成器，**必须100%遵守以下结构，禁止自创任何内容**。

【真实库结构 不可更改】
- 节点标签：{self.STRUCT['NODE_LABEL']}
- 节点属性：
   {self.STRUCT['PROP_NAME']}  ：实体名称
   {self.STRUCT['PROP_TYPE']}  ：实体类型，只能是：{'、'.join(ENTITY_TYPES)}
- 关系：{self.STRUCT['REL_TYPE']}
- 关系属性：{self.STRUCT['REL_PROP_TYPE']} ，关系名称只能是：{'、'.join(RELATION_TYPES)}

【语法强制规则】
1. 固定格式：MATCH (a:{self.STRUCT['NODE_LABEL']})-[r:{self.STRUCT['REL_TYPE']}]->(b:{self.STRUCT['NODE_LABEL']})
2. 用 WHERE a.name = "XX" 匹配名称
3. 用 WHERE a.type = "XX" 过滤实体类型
4. 用 WHERE r.type = "XX" 过滤关系
5. 只返回 name 字段
6. 只输出纯Cypher，不要解释、不要markdown

问题：{question}
"""
        client = HelloAgentsLLM()
        res = client.invoke(messages=[{"role": "user", "content": prompt}], temperature=0)
        cypher = re.sub(r'```.*?```', '', res, flags=re.DOTALL).strip()
        return cypher

    # ========================= 【Neo4j 查询】 =========================
    def search_neo4j(self, query: str, top_k: int = 5, user_id: str = "default"):
        cypher = self.generate_cypher(query)
        print(f"🔍 生成 Cypher:\n{cypher}\n")

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                return result.data()[:top_k]
        except Exception as e:
            print(f"❌ 查询失败：{str(e)}")
            return []

    def close(self):
        close_neo4j_driver()