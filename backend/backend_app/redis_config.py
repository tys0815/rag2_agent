import redis

# ====================== 全局队列名 统一管理 ======================
QUEUE_RAG_QDRANT = "rag_qdrant_queue"
QUEUE_RAG_NEO4J = "rag_neo4j_queue"

def get_redis():
    return redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True,
        socket_timeout=5,
        retry_on_timeout=True
    )