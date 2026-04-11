import json
import time
import logging
from redis_config import get_redis, QUEUE_RAG_QDRANT, QUEUE_RAG_NEO4J

# 你自己的 RAGTool
from helloAgents.tools.builtin.rag_tool import RAGTool

logger = logging.getLogger(__name__)

def run_consumer(queue_name):
    # 每个进程独立创建 Redis 连接（关键！多进程必须这样）
    redis = get_redis()
    logger.info("🚀 消费者启动，监听队列：%s", queue_name)
    while True:
        try:
            # 从队列左侧取任务（非阻塞）
            msg = redis.lpop(queue_name)
            if not msg:
                time.sleep(0.2)
                continue

            # JSON 解析任务
            try:
                task = json.loads(msg)
            except json.JSONDecodeError:
                logger.error("❌ 任务格式不是合法 JSON")
                continue

            # 执行文档处理
            run(queue_name,task)

            logger.info("✅ 处理完成")

        except Exception as e:
            logger.exception("❌ 消费任务时发生异常")  # 自动打印堆栈
            time.sleep(1)

def run(queue_name,task):
    rag_tool = RAGTool()
    # 这里根据 queue_name 来区分处理逻辑
    if queue_name == QUEUE_RAG_QDRANT:
        logger.info("📥 处理 Qdrant 相关任务")
        rag_tool.run(task)
    elif queue_name == QUEUE_RAG_NEO4J:
        logger.info("📥 处理 Neo4j 相关任务")
        rag_tool.run(task)