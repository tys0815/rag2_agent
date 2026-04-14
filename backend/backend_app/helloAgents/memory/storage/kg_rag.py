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

ENTITY_TYPES = ["人物", "门派", "武功", "兵器", "宝物", "地点", "事件"]

ALIAS = {
    "张无忌": ["张教主", "无忌"],
    "段誉": ["段公子", "大理世子"],
    "谢逊": ["金毛狮王", "谢狮王"],
    "张三丰": ["张真人"],
    "成昆": ["混元霹雳手"],
    "鲁有脚": ["北丐传人"],
    "倚天剑": ["倚天", "倚天神兵"],
    "屠龙刀": ["屠龙", "屠龙神兵"],
    "狼牙棒": ["谢逊狼牙棒"],
    "铁杖": ["丐帮铁杖"],
    "武当派": ["武当", "武当山"],
    "峨眉派": ["峨眉"],
    "昆仑派": ["昆仑"],
    "丐帮": ["丐帮宗门"],
    "明教": ["魔教"],
    "降龙掌法": ["降龙", "丐帮降龙掌"],
    "六脉神剑": ["六脉", "大理六脉"],
    "九阳神功": ["九阳", "九阳真气"],
    "乾坤大挪移": ["乾坤", "挪移功"],
    "太极拳": ["太极", "武当太极"],
    "太极剑": ["武当太极剑"],
    "混元功": ["混元霹雳功", "成昆混元功"],
    "青冥山脉": ["青冥山", "青冥"],
    "武当山": ["武当金顶", "武当驻地"],
    "冰火岛": ["冰火驻地"]
}

# ====================== ✅ 正确 UIE 三元组 Schema ======================
UIE_SCHEMA = [
    {"人物": ["属于", "使用", "掌握", "持有", "居住于", "敌对", "交好", "创立", "传承", "师从", "击败", "参与"]},
    {"门派": ["包含", "位于", "敌对", "交好", "创立"]},
    {"武功": ["属于", "创造者", "使用者"]},
    {"兵器": ["属于", "威力", "出处"]},
    {"宝物": ["持有者", "作用"]},
    {"地点": ["属于", "包含", "发生"]},
    {"事件": ["发生在", "参与者", "结果"]}
]
# ======================================================================

NEO4J_STRUCTURE = {
    "NODE_LABEL": "Entity",
    "PROP_NAME": "name",
    "PROP_TYPE": "type",
    "PROP_SUBTYPE": "subtype",
    "PROP_QUALIFIER": "qualifier",
    "REL_TYPE": "RELATION",
    "REL_PROP_TYPE": "type"
}

ENTITY_SUB_TYPES = {
    "人物": ["掌门", "长老", "弟子", "传人"],
    "门派": ["正道", "魔道", "中立"],
    "武功": ["剑法", "掌法", "心法", "内功"],
    "兵器": ["剑", "刀", "棒", "杖"],
    "宝物": ["秘宝", "信物"],
    "地点": ["山脉", "岛屿", "山门"],
    "事件": ["大战", "秘会", "传承"]
}

# ======================【全局驱动】======================
_NEO4J_DRIVER = None

def get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        _NEO4J_DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
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
        print("🔸 加载 UIE 模型（支持三元组抽取）...")
        # ✅ 修复 1：必须用 base 模型，nano 不支持关系
        _UIE_MODEL = Taskflow(
            "information_extraction",
            model="uie-base",
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

    # ========================= 【实体消歧工具】 =========================
    def _disambiguate(self, val):
        for standard, aliases in ALIAS.items():
            if val == standard or val in aliases:
                return standard
        return val

    def parse_uie_result(self, uie_output, min_prob=0.5):
        entities = []
        relations = []
        seen_entities = set()

        if not uie_output:
            return {"entities": entities, "relations": relations}

        # 遍历 UIE 输出结果
        for res in uie_output:
            if not isinstance(res, dict):
                continue
                
            for ent_type, item_list in res.items():
                # 只处理定义好的实体类型
                if ent_type not in self.entity_types:
                    continue
                if not isinstance(item_list, list):
                    continue

                for item in item_list:
                    val = item.get("text", "").strip()
                    prob = item.get("probability", 1.0)
                    if not val or prob < min_prob:
                        continue

                    # ====================== ✅ 修复：自动推断 subtype 和 qualifier ======================
                    # 实体别名归一化
                    val_standard = self._disambiguate(val)
                    # 子类型（取第一个默认值）
                    subtype = ENTITY_SUB_TYPES.get(ent_type, ["其他"])[0]
                    # 限定词（空）
                    qualifier = ""

                    # 去重存储
                    entity_key = (ent_type, val_standard, subtype, qualifier)
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        entities.append({
                            "name": val_standard,
                            "type": ent_type,
                            "subtype": subtype,
                            "qualifier": qualifier,
                            "prob": round(float(prob), 4)
                        })

                    # ====================== ✅ 关系解析 ======================
                    relations_dict = item.get("relations", {})
                    subject = val_standard  # 主语归一化
                    
                    for rel_name, targets in relations_dict.items():
                        if not isinstance(targets, list):
                            continue
                            
                        for tar in targets:
                            obj_val = tar.get("text", "").strip()
                            obj_prob = tar.get("probability", 1.0)
                            obj_type = tar.get("type", "")
                            if not obj_val or obj_prob < min_prob or obj_type not in self.entity_types:
                                continue

                            # 宾语归一化 + 自动推断 subtype
                            obj_val_standard = self._disambiguate(obj_val)
                            obj_subtype = ENTITY_SUB_TYPES.get(obj_type, ["其他"])[0]
                            obj_qualifier = ""

                            # 存储目标实体
                            obj_entity_key = (obj_type, obj_val_standard, obj_subtype, obj_qualifier)
                            if obj_entity_key not in seen_entities:
                                seen_entities.add(obj_entity_key)
                                entities.append({
                                    "name": obj_val_standard,
                                    "type": obj_type,
                                    "subtype": obj_subtype,
                                    "qualifier": obj_qualifier,
                                    "prob": round(float(obj_prob), 4)
                                })

                            # 存储关系三元组
                            relations.append({
                                "subject": subject,
                                "predicate": rel_name,
                                "object": obj_val_standard
                            })

        return {"entities": entities, "relations": relations}

    # ========================= 【ID 生成】 =========================
    def generate_entity_id(self, ent):
        key = f"{ent['type']}_{ent['name']}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    # ========================= 【批量写入】 =========================
    def batch_write_to_kg(self, chunks, entities, relations):
        try:
            with self.driver.session() as session:
                doc_id = chunks[0]["metadata"]["doc_id"] if chunks and "metadata" in chunks[0] else "test"
                now = datetime.now().isoformat()

                # 写入实体
                for ent in entities:
                    eid = self.generate_entity_id(ent)
                    session.run(f"""
                        MERGE (e:{self.STRUCT['NODE_LABEL']} {{id: $eid}})
                        SET e.name=$name, e.type=$type, e.subtype=$subtype, e.updated_at=$now
                    """, eid=eid, name=ent["name"], type=ent["type"], subtype=ent["subtype"], now=now)

                # 写入关系（修复版）
                for rel in relations:
                    sub_ent = next((e for e in entities if e["name"] == rel["subject"]), None)
                    obj_ent = next((e for e in entities if e["name"] == rel["object"]), None)
                    if not sub_ent or not obj_ent:
                        continue
                        
                    sub_id = self.generate_entity_id(sub_ent)
                    obj_id = self.generate_entity_id(obj_ent)
                    session.run(f"""
                        MATCH (a:Entity {{id: $sub}}), (b:Entity {{id: $obj}})
                        MERGE (a)-[r:RELATION]->(b) 
                        SET r.type=$pred, r.updated_at=$now
                    """, sub=sub_id, obj=obj_id, pred=rel["predicate"], now=now)

            print(f"✅ 入库完成：实体 {len(entities)} 个，三元组 {len(relations)} 个")
        except Exception as e:
            print(f"❌ 写入失败：{str(e)}")

    # ========================= 【抽取所有文本块】 =========================
    def extract_all_chunks(self, chunks):
        import gc
        uie = get_uie_model()
        entities_all, relations_all = [], []
        for chunk in chunks:
            out = uie(chunk["content"])
            kg = self.parse_uie_result(out)
            entities_all.extend(kg["entities"])
            relations_all.extend(kg["relations"])
        gc.collect()
        return entities_all, relations_all

    # ========================= 【去重】 =========================
    def merge_kg_global(self, entities, relations):
        entity_dict = {(e["type"], e["name"]): e for e in entities}
        rel_dict = {(r["subject"], r["predicate"], r["object"]): r for r in relations}
        return list(entity_dict.values()), list(rel_dict.values())

    # ========================= 【入库】 =========================
    def add_neo4j_document(self, chunks: List[dict]):
        if not chunks: return []
        entities, relations = self.extract_all_chunks(chunks)
        entities_clean, relations_clean = self.merge_kg_global(entities, relations)
        self.batch_write_to_kg(chunks, entities_clean, relations_clean)
        return relations_clean  # ✅ 直接返回三元组！

    # ========================= 【查询输出三元组】 =========================
    def search_neo4j(self, keyword):
        try:
            with self.driver.session() as session:
                res = session.run("""
                    MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                    WHERE a.name CONTAINS $kw OR b.name CONTAINS $kw
                    RETURN a.name AS s, r.type AS p, b.name AS o
                """, kw=keyword)
                triples = [{"subject": r["s"], "predicate": r["p"], "object": r["o"]} for r in res]
                print("\n✅ 查询到的三元组：")
                for t in triples:
                    print(f"• {t['subject']} → {t['predicate']} → {t['object']}")
                return triples
        except Exception as e:
            print(f"查询失败：{e}")
            return []

    def close(self):
        close_neo4j_driver()