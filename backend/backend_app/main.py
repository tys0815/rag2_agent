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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from helloAgents.tools.registry import global_registry
from helloAgents.tools.builtin.rag_tool import RAGTool  # RAG工具类
from helloAgents.tools.builtin.memory_tool  import MemoryTool  # 记忆工具类
from helloAgents.tools.builtin.terminal_tool import TerminalTool  # 命令行工具类
from helloAgents.tools.builtin.search_tool import SearchTool  # 搜索工具类
from helloAgents.tools.builtin.calculator import CalculatorTool  # 计算器工具类
from helloAgents.tools.builtin.note_tool import NoteTool  # 笔记工具类
from helloAgents.tools.builtin.protocol_tools import MCPTool, A2ATool, ANPTool  # 协议工具类
from helloAgents.core.exceptions import (
    HelloAgentsException,
    http_exception_handler,
    hello_agents_exception_handler,
    general_exception_handler
)

import logging

# 关闭 pdfminer 烦人的警告
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdffont").setLevel(logging.ERROR)


import torch
print(torch.cuda.is_available())  # True = 有GPU可用
print(torch.cuda.device_count())   # GPU数量
# ---------------------- 工具注册函数 ----------------------
def register_all_core_tools():
    """注册所有核心工具到全局注册表"""
    tools_registered = []

    try:
        # 1. RAG工具
        logger.info("🔄 初始化RAG工具...")
        rag_tool = RAGTool()
        if not rag_tool.initialized:
            raise RuntimeError(f"RAG工具初始化失败: {getattr(rag_tool, 'init_error', '未知错误')}")
        global_registry.register_tool(rag_tool)
        tools_registered.append(("rag", "✅"))
        logger.info("✅ RAG工具已注册")
    except Exception as e:
        logger.error(f"❌ RAG工具注册失败: {e}")
        tools_registered.append(("rag", "❌"))

    try:
        # 2. 记忆工具
        logger.info("🔄 初始化记忆工具...")
        memory_tool = MemoryTool()
        global_registry.register_tool(memory_tool)
        tools_registered.append(("memory", "✅"))
        logger.info("✅ 记忆工具已注册")
    except Exception as e:
        logger.error(f"❌ 记忆工具注册失败: {e}")
        tools_registered.append(("memory", "❌"))

    
    # try:
    #     # 3. 命令行工具
    #     logger.info("🔄 初始化命令行工具...")
    #     terminal_tool = TerminalTool(workspace="./workspace")
    #     global_registry.register_tool(terminal_tool)
    #     tools_registered.append(("terminal", "✅"))
    #     logger.info("✅ 命令行工具已注册")
    # except Exception as e:
    #     logger.error(f"❌ 命令行工具注册失败: {e}")
    #     tools_registered.append(("terminal", "❌"))

    # try:
    #     # 4. 搜索工具
    #     logger.info("🔄 初始化搜索工具...")
    #     search_tool = SearchTool(backend="hybrid")
    #     global_registry.register_tool(search_tool)
    #     tools_registered.append(("search", "✅"))
    #     logger.info("✅ 搜索工具已注册")
    # except Exception as e:
    #     logger.error(f"❌ 搜索工具注册失败: {e}")
    #     tools_registered.append(("search", "❌"))

    # try:
    #     # 5. 计算器工具
    #     logger.info("🔄 初始化计算器工具...")
    #     calculator_tool = CalculatorTool()
    #     global_registry.register_tool(calculator_tool)
    #     tools_registered.append(("calculator", "✅"))
    #     logger.info("✅ 计算器工具已注册")
    # except Exception as e:
    #     logger.error(f"❌ 计算器工具注册失败: {e}")
    #     tools_registered.append(("calculator", "❌"))

    # try:
    #     # 6. 笔记工具
    #     logger.info("🔄 初始化笔记工具...")
    #     note_tool = NoteTool()
    #     global_registry.register_tool(note_tool)
    #     tools_registered.append(("note", "✅"))
    #     logger.info("✅ 笔记工具已注册")
    # except Exception as e:
    #     logger.error(f"❌ 笔记工具注册失败: {e}")
    #     tools_registered.append(("note", "❌"))

    # try:
    #     # 7. MCP协议工具
    #     logger.info("🔄 初始化MCP协议工具...")

    #     # 7.1 基础MCP工具（内置演示服务器）
    #     mcp_tool = MCPTool(
    #         name="mcp",
    #         description="连接到MCP服务器，调用工具、读取资源和获取提示词。支持内置服务器和外部服务器。",
    #         auto_expand=True
    #     )
    #     global_registry.register_tool(mcp_tool)
    #     tools_registered.append(("mcp", "✅"))
    #     logger.info("✅ 基础MCP工具已注册")

    #     # 7.2 GitHub MCP工具（如果环境变量GITHUB_PERSONAL_ACCESS_TOKEN存在）
    #     github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    #     if github_token:
    #         try:
    #             github_mcp_tool = MCPTool(
    #                 name="github",
    #                 server_command=["npx", "-y", "@modelcontextprotocol/server-github"],
    #                 env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
    #                 auto_expand=True
    #             )
    #             global_registry.register_tool(github_mcp_tool)
    #             tools_registered.append(("github_mcp", "✅"))
    #             logger.info("✅ GitHub MCP工具已注册")
    #         except Exception as e:
    #             logger.warning(f"⚠️ GitHub MCP工具注册失败: {e}")
    #             tools_registered.append(("github_mcp", "⚠️"))
    #     else:
    #         logger.info("ℹ️  未设置GITHUB_PERSONAL_ACCESS_TOKEN，跳过GitHub MCP工具注册")
    #         tools_registered.append(("github_mcp", "⏭️"))

    #     # 7.3 A2A工具（需要a2a-sdk）
    #     try:
    #         a2a_tool = A2ATool(
    #             name="a2a",
    #             agent_url="http://localhost:5000",  # 默认URL，实际使用时需要配置
    #             description="连接到A2A Agent，支持提问和获取信息。需要安装官方a2a-sdk库。"
    #         )
    #         global_registry.register_tool(a2a_tool)
    #         tools_registered.append(("a2a", "✅"))
    #         logger.info("✅ A2A工具已注册")
    #     except Exception as e:
    #         logger.warning(f"⚠️ A2A工具注册失败（可能需要安装a2a-sdk）: {e}")
    #         tools_registered.append(("a2a", "⚠️"))

    #     # 7.4 ANP工具（概念性实现）
    #     try:
    #         anp_tool = ANPTool(
    #             name="anp",
    #             description="智能体网络管理工具，支持服务发现、节点管理和消息路由。概念性实现。"
    #         )
    #         global_registry.register_tool(anp_tool)
    #         tools_registered.append(("anp", "✅"))
    #         logger.info("✅ ANP工具已注册")
    #     except Exception as e:
    #         logger.error(f"❌ ANP工具注册失败: {e}")
    #         tools_registered.append(("anp", "❌"))

    # except Exception as e:
    #     logger.error(f"❌ MCP协议工具注册失败: {e}")
    #     tools_registered.append(("mcp_protocols", "❌"))

    # 总结注册结果
    logger.info("📋 核心工具注册完成:")
    for tool_name, status in tools_registered:
        logger.info(f"  {status} {tool_name}")

    # 列出所有可用工具
    all_tools = global_registry.list_tools()

    return all_tools




# ---------------------- 应用生命周期管理 ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段：初始化并注册所有核心工具
    logger.info("🚀 应用启动中...")

    try:
        # 注册所有核心工具
        register_all_core_tools()


        logger.info("✅ 所有核心工具已成功注册到全局注册表")

    except Exception as e:
        logger.critical(f"❌ 工具初始化失败，应用启动终止: {e}")
        raise  # 核心工具初始化失败，终止应用启动

    yield  # 应用运行阶段

    # 关闭阶段：清理资源
    logger.info("🔌 应用关闭中...")
    try:
        # 可选：清理工具资源
        rag_tool = global_registry.get_tool("rag")
        if rag_tool:
            # 示例：清空过期命名空间/关闭数据库连接
            # rag_tool.clear_all_namespaces()
            logger.info("✅ RAG工具资源已清理")

        # 清理其他工具资源（如有需要）
        logger.info("✅ 所有工具资源已清理")
    except Exception as e:
        logger.error(f"⚠️ 清理工具资源时发生警告: {e}")

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

# ---------------------- 注册全局异常处理器 ----------------------
# 统一处理HTTP异常、业务异常和未知异常
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(HelloAgentsException, hello_agents_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
logger.info("✅ 全局异常处理器已注册")

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