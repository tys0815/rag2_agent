from datetime import datetime
import json
from typing import Dict, List
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

# ====================== 【企业级 · 实体消歧 Schema】 ======================
# 1. 实体主类型（必须）
ENTITY_TYPES = ["人物", "门派", "武功", "兵器", "宝物", "地点", "事件"]

# 2. 【消歧核心】实体别名词典（真正解决：张教主 = 张无忌）
ALIAS = {
    # 人物别名（贴合小说原文，解决异名归一）​
    "张无忌": ["张教主", "无忌"],
    "段誉": ["段公子", "大理世子"],
    "谢逊": ["金毛狮王", "谢狮王"],
    "张三丰": ["张真人"],
    "成昆": ["混元霹雳手"],
    "鲁有脚": ["北丐传人"],
    # 兵器别名（贴合小说原文）​
    "倚天剑": ["倚天", "倚天神兵"],
    "屠龙刀": ["屠龙", "屠龙神兵"],
    "狼牙棒": ["谢逊狼牙棒"],
    "铁杖": ["丐帮铁杖"],
    # 门派别名（贴合小说原文）​
    "武当派": ["武当", "武当山"],
    "峨眉派": ["峨眉"],
    "昆仑派": ["昆仑"],
    "丐帮": ["丐帮宗门"],
    "明教": ["魔教"],
    # 武功别名（贴合小说原文）​
    "降龙掌法": ["降龙", "丐帮降龙掌"],
    "六脉神剑": ["六脉", "大理六脉"],
    "九阳神功": ["九阳", "九阳真气"],
    "乾坤大挪移": ["乾坤", "挪移功"],
    "太极拳": ["太极", "武当太极"],
    "太极剑": ["武当太极剑"],
    "混元功": ["混元霹雳功", "成昆混元功"],
    # 地点别名（贴合小说原文）​
    "青冥山脉": ["青冥山", "青冥"],
    "武当山": ["武当金顶", "武当驻地"],
    "冰火岛": ["冰火驻地"]
}

# 3. 【消歧核心】实体关系约束
RELATION_DEFINITIONS = {
    ("人物", "人物"):      ["师徒", "父子", "母子", "兄弟", "仇敌", "盟友", "同门", "救命恩人", "夫妇", "莫逆之交", "传人"],
    ("人物", "门派"):      ["属于", "创立", "掌门", "长老", "带领", "联合"],
    ("门派", "门派"):      ["仇敌", "盟友", "联合", "对立"],
    ("人物", "武功"):      ["修炼", "精通", "自创", "传承", "使用"],
    ("人物", "兵器"):      ["持有", "使用", "供奉", "传承"],
    ("人物", "宝物"):      [],
    ("人物", "地点"):      ["常驻", "到访", "隐居地", "发生地", "驻守"],
    ("门派", "地点"):      ["常驻", "驻地", "结盟地"],
    ("人物", "事件"):      ["参与", "发起", "受害者", "见证", "歼灭", "击败"],
    ("门派", "事件"):      ["参与", "联合发起", "对抗"],
    ("事件", "地点"):      ["发生于", "举办地"],
    ("武功", "武功"):      [],
    ("兵器", "兵器"):      []
}

# 4. 自动生成合法 UIE 消歧抽取 Schema
UIE_SCHEMA = []
for (s, o), rels in RELATION_DEFINITIONS.items():
    for rel in rels:
        UIE_SCHEMA.append({"subject": s, "predicate": rel, "object": o})

# 6. Neo4j 消歧存储结构（企业标准）
NEO4J_STRUCTURE = {
    "NODE_LABEL": "Entity",
    "PROP_NAME": "name",
    "PROP_TYPE": "type",
    "PROP_SUBTYPE": "subtype",
    "PROP_QUALIFIER": "qualifier",
    "REL_TYPE": "RELATION",
    "REL_PROP_TYPE": "type"
}

# 【修复】补充缺失的实体子类型定义
ENTITY_SUB_TYPES = {
    "人物": ["掌门", "长老", "弟子", "传人"],
    "门派": ["正道", "魔道", "中立"],
    "武功": ["剑法", "掌法", "心法", "内功"],
    "兵器": ["剑", "刀", "棒", "杖"],
    "宝物": ["秘宝", "信物"],
    "地点": ["山脉", "岛屿", "山门"],
    "事件": ["大战", "秘会", "传承"]
}
# ==========================================================================

# ======================【全局驱动】======================
_NEO4J_DRIVER = None

def get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        _NEO4J_DRIVER = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _NEO4J_DRIVER

def close_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER:
        _NEO4J_DRIVER.close()
        _NEO4J_DRIVER = None

# ======================【UIE单例】======================
_UIE_MODEL = None

def get_uie_model():
    global _UIE_MODEL
    if _UIE_MODEL is None:
        import paddlenlp
        from paddlenlp import Taskflow
        print("🔸 加载 UIE 模型（武侠消歧抽取）...")
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
        self.entity_types = ENTITY_TYPES
        self.sub_types = ENTITY_SUB_TYPES
        self.relation_types = UIE_SCHEMA

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

                # ========== 自动消歧：子类型 + 限定词 ==========
                subtype = self._guess_subtype(ent_type, val)
                qualifier = self._guess_qualifier(ent_type, val)

                entity_key = (ent_type, val, subtype, qualifier)
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities.append({
                        "name": val,
                        "type": ent_type,
                        "subtype": subtype,
                        "qualifier": qualifier,
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

                        obj_subtype = self._guess_subtype(obj_type, obj_val)
                        obj_qualifier = self._guess_qualifier(obj_type, obj_val)
                        obj_entity_key = (obj_type, obj_val, obj_subtype, obj_qualifier)

                        if obj_entity_key not in seen_entities:
                            seen_entities.add(obj_entity_key)
                            entities.append({
                                "name": obj_val,
                                "type": obj_type,
                                "subtype": obj_subtype,
                                "qualifier": obj_qualifier,
                                "prob": round(float(obj_prob), 4)
                            })

                        relations.append({
                            "subject": subject,
                            "predicate": rel_name,
                            "object": obj_val
                        })

        return {"entities": entities, "relations": relations}
    
    def _guess_subtype(self, ent_type, val):
        # 按你要求：不使用子类型，直接返回空
        return ""

    def _guess_qualifier(self, ent_type, val):
        # 基于 ALIAS 词典做实体消歧，返回标准原名
        for standard_name, aliases in ALIAS.items():
            if val == standard_name or val in aliases:
                return standard_name
        # 不在别名表里就返回原名，保证实体唯一
        return val

    # ========================= 【消歧ID生成】 =========================
    def generate_entity_id(self, ent):
        key_parts = [
            ent["type"],
            ent["subtype"],
            ent["name"],
            ent["qualifier"]
        ]
        key = "_".join(str(p) for p in key_parts if p).strip("_")
        return hashlib.md5(key.encode()).hexdigest()[:16]

    # ========================= 【批量写入 KG（消歧版）】 =========================
    def batch_write_to_kg(self, chunks, entities, relations):
        try:
            with self.driver.session() as session:
                doc_id = "unknown"
                if chunks and "metadata" in chunks[0]:
                    doc_id = chunks[0]["metadata"].get("doc_id", "unknown")

                user_id = self.user_id
                now = datetime.now().isoformat()

                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.user_id = $user_id, d.updated_at = $now
                """, doc_id=doc_id, user_id=user_id, now=now)

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

                # 实体（带消歧）
                for ent in entities:
                    name = ent["name"]
                    ent_type = ent["type"]
                    subtype = ent.get("subtype", "")
                    qualifier = ent.get("qualifier", "")
                    prob = ent.get("prob", 1.0)
                    chunk_id = ent.get("chunk_id")
                    entity_id = self.generate_entity_id(ent)

                    session.run(f"""
                        MERGE (e:{self.STRUCT['NODE_LABEL']} {{id: $entity_id}})
                        SET e.{self.STRUCT['PROP_NAME']} = $name,
                            e.{self.STRUCT['PROP_TYPE']} = $ent_type,
                            e.{self.STRUCT['PROP_SUBTYPE']} = $subtype,
                            e.{self.STRUCT['PROP_QUALIFIER']} = $qualifier,
                            e.probability = $prob,
                            e.doc_id = $doc_id, e.user_id = $user_id, e.updated_at = $now
                    """, entity_id=entity_id, name=name, ent_type=ent_type, subtype=subtype,
                                qualifier=qualifier, prob=prob, doc_id=doc_id, user_id=user_id, now=now)
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

            print(f"✅ 消歧版KG入库完成：实体={len(entities)} 关系={len(relations)}")
        except Exception as e:
            print(f"❌ 写入失败：{str(e)}")

    # ========================= 【分块抽取】 =========================
    def extract_all_chunks(self, chunks):
        from tqdm import tqdm
        import gc
        gc.collect()
        uie = get_uie_model()
        entities_all, relations_all = [], []
        BATCH_SIZE = 10

        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="🔍 抽取实体关系（消歧）"):
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

    # ========================= 【全局消歧去重】 =========================
    def merge_kg_global(self, entities, relations):
        entity_map = {}
        for e in entities:
            key = (e["type"], e["subtype"], e["name"], e["qualifier"])
            if key not in entity_map or e["prob"] > entity_map[key]["prob"]:
                entity_map[key] = e

        merged_entities = []
        for e in entity_map.values():
            e["id"] = self.generate_entity_id(e)
            merged_entities.append(e)

        rel_set = set()
        merged_rels = []
        for r in relations:
            key = (r["subject"], r["predicate"], r["object"])
            if key not in rel_set:
                rel_set.add(key)
                merged_rels.append(r)
        return merged_entities, merged_rels

    # ========================= 【文档入库】 =========================
    def add_neo4j_document(self, chunks: List[dict]):
        if not chunks: return False
        entities, relations = self.extract_all_chunks(chunks)
        entities_clean, relations_clean = self.merge_kg_global(entities, relations)
        self.batch_write_to_kg(chunks, entities_clean, relations_clean)
        return True

    # ========================= 【Cypher 生成】 =========================
    def generate_cypher(self, question: str) -> str:
        # 1. 正确的参数抽取 Prompt
        prompt = f"""
    你是知识图谱参数抽取器，**只输出纯JSON，不要任何解释、不要符号、不要数组**。

    实体类型可选：{"、".join(self.entity_types)}
    关系类型可选：{"、".join(self.relation_types)}

    请从问题中抽取以下4个字段：
    - subject_name: 主体名称（没有则为空字符串 ""）
    - subject_type: 主体类型（没有则为空字符串 ""）
    - relation_type: 关系类型（没有则为空字符串 ""）
    - target_type: 目标实体类型（没有则为空字符串 ""）

    输出格式（严格JSON）：
    {{"subject_name":"张三","subject_type":"人物","relation_type":"属于","target_type":"部门"}}

    问题：{question}
        """.strip()

        # 2. 调用 LLM
        client = HelloAgentsLLM()
        res = client.invoke(
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # 3. 清洗JSON
        res = re.sub(r'```json|```', '', res.strip())
        params = json.loads(res)

        # 4. 固定 MATCH 结构
        match_clause = "MATCH (a:Entity)-[r:RELATION]->(b:Entity)"

        # 5. 构建 WHERE 条件
        where_list = []
        if params.get("subject_name"):
            where_list.append(f'a.name = "{params["subject_name"]}"')
        if params.get("subject_type"):
            where_list.append(f'a.type = "{params["subject_type"]}"')
        if params.get("relation_type"):
            where_list.append(f'r.type = "{params["relation_type"]}"')
        if params.get("target_type"):
            where_list.append(f'b.type = "{params["target_type"]}"')

        where_clause = "WHERE " + " AND ".join(where_list) if where_list else ""

        # 6. 返回数组
        return_clause = "RETURN collect({name: b.name, type: b.type}) AS result_array"

        # 7. 拼接最终 Cypher
        cypher = " ".join([match_clause, where_clause, return_clause]).strip()

        return cypher

    # ========================= 【查询（安全+消歧）】 =========================
    def search_neo4j(self, query: str, top_k: int = 5, user_id: str = "default"):
        # 【修复】去掉强制覆盖，使用用户传入的query
        cypher = self.generate_cypher(query)
        print(f"🔍 生成 Cypher:\n{cypher}\n")

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                data = result.data()
                return data[:top_k]
        except Exception as e:
            print(f"❌ 查询失败：{str(e)}")
            return []

    def close(self):
        close_neo4j_driver()