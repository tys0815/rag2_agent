import os
import sys
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

# 配置日志（增强企业级日志格式）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 路径配置
BACKEND_APP_DIR = os.path.dirname(os.path.abspath(__file__))
# 将 backend_app 加入 Python 搜索路径最前端（优先识别）
if BACKEND_APP_DIR not in sys.path:
    sys.path.insert(0, BACKEND_APP_DIR)
    logger.info(f"✅ 已将 backend_app 加入搜索路径: {BACKEND_APP_DIR}")

# FastAPI 核心导入
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from helloAgents.tools.builtin.rag_tool import RAGTool  # RAG工具类
from helloAgents.tools.builtin.memory_tool import MemoryTool  # RAG工具类
from helloAgents.tools.registry import global_registry


import torch
print(torch.cuda.is_available())  # True = 有GPU可用
print(torch.cuda.device_count())   # GPU数量
# ---------------------- 应用生命周期管理 ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段：初始化并注册 RAG 工具
    logger.info("🚀 应用启动中...")
    try:
        # 1. 初始化 RAGTool（从环境变量读取配置，保证灵活性）
        rag_tool = RAGTool()

        memory_tool = MemoryTool()
        
        # 2. 注册到全局工具注册表
        if not rag_tool.initialized:
            raise RuntimeError(f"RAG工具初始化失败: {getattr(rag_tool, 'init_error', '未知错误')}")
        
        global_registry.register_tool(rag_tool)
        global_registry.register_tool(memory_tool)  
        logger.info(f"✅ RAG工具已成功注册到全局注册表{global_registry.get_tools_description()}")
        
    except Exception as e:
        logger.critical(f"❌ RAG工具初始化失败，应用启动终止: {e}")
        raise  # 核心工具初始化失败，终止应用启动
    
    yield  # 应用运行阶段
    
    # 关闭阶段：清理资源
    logger.info("🔌 应用关闭中...")
    try:
        # 可选：清理RAG资源（根据业务需求）
        rag_tool = global_registry.get_tool("rag")
        if rag_tool:
            # 示例：清空过期命名空间/关闭数据库连接
            # rag_tool.clear_all_namespaces()
            logger.info("✅ RAG工具资源已清理")
    except Exception as e:
        logger.error(f"⚠️ 清理RAG资源时发生警告: {e}")

# ---------------------- 创建 FastAPI 应用实例 ----------------------
app = FastAPI(
    title="Llama3.2 RAG 服务",
    description="集成RAGTool的检索增强生成服务",
    version="1.0.0",
    lifespan=lifespan,  # 绑定生命周期钩子
    reload=False  # 开发模式自动重载
)

# ---------------------- 导入并注册已有路由 ----------------------
# 你的上传/查询接口都在这个路由中，保持原有逻辑不变
try:
    from backend_app.api.api_router import api_router
    app.include_router(router=api_router, prefix='/api/v1')
    logger.info("✅ 已注册API路由: /api/v1")
except ImportError as e:
    logger.error(f"❌ 导入API路由失败: {e}")
    raise

# ---------------------- 辅助接口：工具管理 ----------------------
@app.get("/")
async def root():
    """根路由：健康检查"""
    # 获取RAG工具状态
    rag_tool = global_registry.get_tool("rag")
    rag_status = "✅ 已初始化" if getattr(rag_tool, "initialized", False) else "❌ 未初始化"
    
    return JSONResponse(
        content={
            "message": "Llama3.2 服务启动成功！",
            "rag_tool_status": rag_status,
            "registered_tools": global_registry.list_tools(),
            "api_prefix": "/api/v1"
        }
    )