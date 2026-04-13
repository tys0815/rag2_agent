import redis

# ====================== 全局队列名 统一管理 ======================
QUEUE_RAG_QDRANT = "rag_qdrant_queue"
QUEUE_RAG_NEO4J = "rag_neo4j_queue"
QUEUE_MEMORY = "memory_queue"

redis_pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    socket_timeout=5,
    retry_on_timeout=True
)

def get_redis():
    # 每次都从池子拿，安全、不重复建连接
    return redis.Redis(connection_pool=redis_pool)