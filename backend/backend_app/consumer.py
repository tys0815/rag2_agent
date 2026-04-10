import json
import time
import logging
from redis_config import get_redis, QUEUE_RAG

# 你自己的 RAGTool
from helloAgents.tools.builtin.rag_tool import RAGTool

logger = logging.getLogger(__name__)

def run_consumer():
    # 每个进程独立创建 Redis 连接（关键！多进程必须这样）
    redis = get_redis()
    logger.info("🚀 RAG 消费者启动，监听队列：%s", QUEUE_RAG)

    # 每个进程独立初始化 RAG 工具（避免多进程资源冲突）
    rag_tool = RAGTool()

    while True:
        try:
            # 从队列左侧取任务（非阻塞）
            msg = redis.lpop(QUEUE_RAG)
            if not msg:
                time.sleep(0.2)
                continue

            # JSON 解析任务
            try:
                task = json.loads(msg)
            except json.JSONDecodeError:
                logger.error("❌ 任务格式不是合法 JSON")
                continue

            logger.info("📥 开始处理 RAG 任务")

            # 执行文档处理
            rag_tool.run(task)

            logger.info("✅ RAG 处理完成")

        except Exception as e:
            logger.exception("❌ 消费任务时发生异常")  # 自动打印堆栈
            time.sleep(1)