import multiprocessing
import logging
import uvicorn

from consumer import run_consumer
from redis_config import QUEUE_RAG_QDRANT, QUEUE_RAG_NEO4J

import sys
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ------------------- 启动 API -------------------
def run_api():
    uvicorn.run(
        "backend_app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1
    )

# ------------------- 【核心】启动指定队列的 N 个消费者 -------------------
def start_consumers(queue_name, count=1):
    """
    启动同一个队列的多个消费者进程
    :param queue_name: 队列名
    :param count: 启动几个消费者
    :return: 进程列表
    """
    processes = []
    for i in range(count):
        p = multiprocessing.Process(
            target=run_consumer,
            args=(queue_name,),
            name=f"Consumer-{queue_name}-{i+1}"
        )
        p.start()
        processes.append(p)
        logging.info(f"✅ 启动消费者：{p.name}")
    return processes

# ------------------- 主入口：自由配置数量 -------------------
if __name__ == "__main__":
    # 1. 启动 API
    p_api = multiprocessing.Process(target=run_api, name="API-Server")
    p_api.start()

    # 2. 启动多种消费者
    all_processes = []

    # 启动 Qdrant 消费者 N 个
    qdrant_procs = start_consumers(QUEUE_RAG_QDRANT, 1)
    all_processes.extend(qdrant_procs)

    # 启动 Neo4j 消费者 N 个
    neo4j_procs = start_consumers(QUEUE_RAG_NEO4J, 1)
    all_processes.extend(neo4j_procs)

    # 等待所有
    p_api.join()
    for p in all_processes:
        p.join()