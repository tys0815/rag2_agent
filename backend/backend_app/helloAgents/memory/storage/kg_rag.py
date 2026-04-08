from neo4j import GraphDatabase
import json
import re
from helloAgents.core.llm import HelloAgentsLLM

# ======================【配置】======================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123456"
# ====================================================

# ======================【Schema 已修复完整】======================
SCHEMA = {
    "node_labels": ["Document", "Chunk", "Entity", "Person", "Department", "Project", "Term", "TextContent", "Keyword"],
    "relation_types": [
        "BELONGS_TO",
        "HAS",
        "RELATED_TO",
        "CREATED_BY",
        "PART_OF",
        "REFERS_TO",
        "HAS_CHUNK",    # 新增
        "HAS_ENTITY"    # 新增
    ],
    "properties": ["name", "type", "content", "create_time", "source"]
}

ALLOWED_ENTITY_TYPES = [
    "Document",
    "TextContent",
    "Keyword",
    "Person",
    "Department",
    "Project"
]

class Neo4jKGRAG_Enterprise:
    def __init__(self, user_id="default"):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        self.user_id = user_id
        self.llm = HelloAgentsLLM(temperature=0)
        # self._create_user_graph_if_not_exists()

    # ========== 修复 2：创建图谱使用参数化，无注入 ==========
    # def _create_user_graph_if_not_exists(self):
    #     try:
    #         with self.driver.session(database="system") as session:
    #             session.run("CREATE DATABASE $graph IF NOT EXISTS", graph=self.graph_name)
    #         print(f"✅ 用户图谱就绪：{self.graph_name}")
    #     except Exception as e:
    #         print(f"❌ 创建用户图谱失败：{e}")

    # ========== 修复 3：LLM 异常捕获 ==========
    def extract_entities_relations(self, text):
        prompt = f"""
你是严格知识图谱抽取器，只抽取以下允许内容，禁止编造。

允许实体类型：{', '.join(ALLOWED_ENTITY_TYPES)}
允许关系：{', '.join(SCHEMA['relation_types'])}

输出严格JSON，无其他内容：
{{
    "entities": [{{"name":"实体名", "type":"类型"}}],
    "relations": [{{"subject":"主体", "predicate":"关系", "object":"客体"}}]
}}

文本：{text}
"""
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}],temperature=0)
            data = json.loads(result)
            data["entities"] = [e for e in data.get("entities", []) if e.get("type") in ALLOWED_ENTITY_TYPES]
            return data
        except Exception as e:
            print("❌ LLM 抽取失败：", e)
            return {"entities": [], "relations": []}

    # ========== 修复 4：移除外部 user_id，使用当前图谱用户 ==========
    def write_chunk_to_kg(self, chunk):
        chunk_id = chunk["id"]
        content = chunk["content"]
        meta = chunk["metadata"]
        doc_id = meta.get("doc_id", "unknown_doc")
        source = meta.get("source", "rag")
        source_path = meta.get("source_path", "")

        kg_data = self.extract_entities_relations(content)

        with self.driver.session() as session:
            # Document
            session.run("""
                MERGE (d:Document {doc_id: $doc_id})
                SET d.source_path = $source_path,
                    d.user_id = $user_id,
                    d.source = $source,
                    d.updated_at = timestamp()
            """, doc_id=doc_id, source_path=source_path, user_id=self.user_id, source=source)

            # Chunk
            session.run("""
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.content = substring($content, 0, 100),
                    c.doc_id = $doc_id,
                    c.user_id = $user_id,
                    c.updated_at = timestamp()
            """, chunk_id=chunk_id, content=content, doc_id=doc_id, user_id=self.user_id)

            # Document -[:HAS_CHUNK]-> Chunk
            session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, doc_id=doc_id, chunk_id=chunk_id)

            # 实体
            for ent in kg_data.get("entities", []):
                name = ent["name"]
                type = ent["type"]
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type, e.updated_at = timestamp()
                """, name=name, type=type)

                session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MATCH (e:Entity {name: $name})
                    MERGE (c)-[:HAS_ENTITY]->(e)
                """, chunk_id=chunk_id, name=name)

            # 关系
            for rel in kg_data.get("relations", []):
                predicate = rel.get("predicate")
                if predicate not in SCHEMA["relation_types"]:
                    continue

                session.run(f"""
                    MATCH (s:Entity {{name: $subject}})
                    MATCH (o:Entity {{name: $object}})
                    MERGE (s)-[:{predicate}]->(o)
                """, subject=rel["subject"], object=rel["object"])

    def get_schema(self):
        return f"""
Node labels: {', '.join(SCHEMA['node_labels'])}
Relationship types: {', '.join(SCHEMA['relation_types'])}
Properties: name, type, content
"""

    def generate_cypher(self, question, schema):
        prompt = f"""
你是Neo4j Cypher生成器，严格遵守：
1. 只使用以下结构：
{schema}
2. 最多2跳，禁止深度遍历
3. 只返回纯Cypher，不要解释、不要```
4. 关系名称必须大写

问题：{question}
"""
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}],temperature=0)
            cypher = result.strip()
            cypher = re.sub(r'```.*?```', '', cypher, flags=re.DOTALL).strip()
            return cypher
        except:
            return ""

    def run_cypher(self, cypher):
        if not cypher:
            return []
        try:
            with self.driver.session() as session:
                return session.run(cypher).data()
        except Exception as e:
            print("Cypher 执行错误：", e)
            return []

    # ========== 修复 5：显示当前用户名 ==========
    def generate_answer(self, context):
        if not context:
            return f"用户【{self.user_id}】暂无相关知识。"

        data = context[0]
        info_text = "\n".join([f"{key}: {value}" for key, value in data.items()])
        return f"【用户 {self.user_id}】知识图谱：\n{info_text}"

    def query(self, question):
        schema = self.get_schema()
        cypher = self.generate_cypher(question, schema)
        print("✅ 生成 Cypher：\n", cypher, "\n")
        results = self.run_cypher(cypher)
        answer = self.generate_answer(results)
        return answer, cypher, results

    def close(self):
        try:
            self.driver.close()
        except:
            pass