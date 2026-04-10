import multiprocessing
import logging
import uvicorn

from consumer import run_consumer

import sys
from pathlib import Path
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent  # 自动定位到 backend 目录
sys.path.append(str(ROOT_DIR))

# 日志基础配置
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

# ------------------- 启动多个 RAG 消费者 -------------------
def run_rag_consumers(count=3):
    processes = []
    for i in range(count):
        p = multiprocessing.Process(target=run_consumer, name=f"RAG-Consumer-{i+1}")
        p.start()
        processes.append(p)
        logging.info(f"✅ RAG 消费者 %d 启动成功", i+1)
    return processes

# ------------------- 主入口 -------------------
if __name__ == "__main__":
    logging.info("🚀 启动整个系统：API + 3 个 RAG 消费者")

    # 1. 启动 API
    p_api = multiprocessing.Process(target=run_api, name="API-Server")
    p_api.start()

    # 2. 启动消费者
    p_consumers = run_rag_consumers(count=3)

    # 等待所有进程运行
    p_api.join()
    for p in p_consumers:
        p.join()