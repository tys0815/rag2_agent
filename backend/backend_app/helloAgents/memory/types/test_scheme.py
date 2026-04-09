import spacy
from typing import List, Dict
from dataclasses import dataclass

# ======================
# 1. 【先定义】公司制度文档 Schema
# ======================
COMPANY_POLICY_SCHEMA = {
    "entity_types": [
        "PERSON",
        "DEPARTMENT",
        "POSITION",
        "POLICY",
        "POLICY_CLAUSE",
        "RULE",
        "DUTY"
    ],
    "relations": [
        {
            "subject": "PERSON",
            "relation": "BELONGS_TO_DEPT",
            "object": "DEPARTMENT",
            "triggers": ["属于", "隶属于", "所在部门"]
        },
        {
            "subject": "PERSON",
            "relation": "HOLDS_POSITION",
            "object": "POSITION",
            "triggers": ["担任", "任职", "岗位为"]
        },
        {
            "subject": "DEPARTMENT",
            "relation": "RESPONSIBLE_FOR_POLICY",
            "object": "POLICY",
            "triggers": ["负责", "制定", "执行"]
        },
        {
            "subject": "POLICY",
            "relation": "CONTAINS_CLAUSE",
            "object": "POLICY_CLAUSE",
            "triggers": ["包含", "规定"]
        },
        {
            "subject": "POSITION",
            "relation": "UNDERTAKE_DUTY",
            "object": "DUTY",
            "triggers": ["负责", "承担", "履行"]
        },
        {
            "subject": "PERSON",
            "relation": "MUST_COMPLY_WITH",
            "object": "POLICY",
            "triggers": ["遵守", "执行", "按照"]
        }
    ]
}

# ======================
# 2. 实体、关系结构（简化版）
# ======================
@dataclass
class Entity:
    entity_id: str
    name: str
    entity_type: str

@dataclass
class Relation:
    from_entity: str
    to_entity: str
    relation_type: str
    evidence: str

# ======================
# 3. 实体抽取（按Schema）
# ======================
def extract_entities(nlp, text: str) -> List[Entity]:
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        # 只保留 Schema 里的实体类型
        if ent.label_ not in COMPANY_POLICY_SCHEMA["entity_types"]:
            continue

        entity = Entity(
            entity_id=f"entity_{hash(ent.text)}",
            name=ent.text,
            entity_type=ent.label_
        )
        entities.append(entity)
    return entities

# ======================
# 4. 关系抽取（按Schema）
# ======================
def extract_relations(nlp, text: str, entities: List[Entity]) -> List[Relation]:
    doc = nlp(text)
    relations = []
    ent_map = {e.name: e for e in entities}

    for sent in doc.sents:
        sent_ents = list(sent.ents)
        sent_text = sent.text

        for rule in COMPANY_POLICY_SCHEMA["relations"]:
            subj_type = rule["subject"]
            rel_type = rule["relation"]
            obj_type = rule["object"]
            triggers = rule["triggers"]

            if not any(t in sent_text for t in triggers):
                continue

            subjects = [e for e in sent_ents if e.label_ == subj_type]
            objects = [e for e in sent_ents if e.label_ == obj_type]

            for subj in subjects:
                for obj in objects:
                    if subj.text in ent_map and obj.text in ent_map:
                        relations.append(Relation(
                            from_entity=ent_map[subj.text].entity_id,
                            to_entity=ent_map[obj.text].entity_id,
                            relation_type=rel_type,
                            evidence=sent_text
                        ))
    return relations

# ======================
# 5. 测试主程序
# ======================
if __name__ == "__main__":
    # 加载中文模型
    nlp = spacy.load("zh_core_web_sm")

    # 测试文本（公司制度句子）
    test_text = """
    张三隶属于技术部，担任后端工程师岗位，负责系统开发工作。
    技术部负责制定《员工考勤管理制度》，该制度包含打卡规定。
    所有员工必须遵守考勤制度。
    """

    print("=" * 60)
    print("📄 测试文本：")
    print(test_text.strip())
    print("=" * 60)

    # 1. 抽取实体（按Schema）
    entities = extract_entities(nlp, test_text)
    print("\n🔎 提取到的实体：")
    for e in entities:
        print(f"实体：{e.name}  类型：{e.entity_type}")

    # 2. 抽取关系（按Schema）
    relations = extract_relations(nlp, test_text, entities)
    print("\n🔗 提取到的三元组：")
    for r in relations:
        print(f"({r.from_entity[-8:]}, {r.relation_type}, {r.to_entity[-8:]})")

    print("\n✅ 测试完成！")