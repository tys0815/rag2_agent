import json
import time
import logging
from redis_config import get_redis, QUEUE_RAG_QDRANT, QUEUE_RAG_NEO4J, QUEUE_MEMORY

# 你自己的 RAGTool
from helloAgents.tools.builtin.rag_tool import RAGTool
from helloAgents.tools.builtin.memory_tool import MemoryTool

logger = logging.getLogger(__name__)

# ====================== 【全局单例：只创建一次】 ======================
# 全局只初始化1次，所有任务共用
_RAG_TOOL = None
_MEMORY_TOOL = None

def get_rag_tool():
    """获取RAG工具单例"""
    global _RAG_TOOL
    if _RAG_TOOL is None:
        _RAG_TOOL = RAGTool()
    return _RAG_TOOL

def get_memory_tool():
    """获取记忆工具单例"""
    global _MEMORY_TOOL
    if _MEMORY_TOOL is None:
        _MEMORY_TOOL = MemoryTool()
    return _MEMORY_TOOL
# ======================================================================

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
            run(queue_name, task)
            logger.info("✅ 处理完成")

        except Exception as e:
            logger.exception("❌ 消费任务时发生异常")
            time.sleep(1)

def run(queue_name, task):
    logger.info(f"📥 处理 {queue_name} 相关任务")
    
    # ====================== 【使用单例，不重复创建】 ======================
    if queue_name == QUEUE_RAG_QDRANT:
        rag_tool = get_rag_tool()  # 拿单例，不新建
        rag_tool.run(task)
        
    elif queue_name == QUEUE_RAG_NEO4J:
        rag_tool = get_rag_tool()  # 拿单例，不新建
        rag_tool.run(task)
        
    elif queue_name == QUEUE_MEMORY:
        memory_tool = get_memory_tool()  # 拿单例，不新建
        memory_tool.run(task)