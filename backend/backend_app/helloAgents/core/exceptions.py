"""异常体系
包含基础异常类和FastAPI全局异常处理器
"""

import uuid
import logging
from datetime import datetime
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ==================== 统一的业务异常基类 ====================
class HelloAgentsException(Exception):
    """HelloAgents基础异常类（统一携带错误码+消息）"""
    code: int = 500
    message: str = "服务异常"

    def __init__(self, message: str = None, code: int = None):
        if message:
            self.message = message
        if code:
            self.code = code


# 业务异常子类（统一继承、统一格式）
class LLMException(HelloAgentsException):
    """LLM相关异常"""
    code = 400
    message = "LLM调用异常"


class AgentException(HelloAgentsException):
    """Agent相关异常"""
    code = 400
    message = "Agent执行异常"


class ConfigException(HelloAgentsException):
    """配置相关异常"""
    code = 400
    message = "配置异常"


class ToolException(HelloAgentsException):
    """工具相关异常"""
    code = 400
    message = "工具调用异常"


# ==================== FastAPI全局异常处理器（完全统一） ====================
class ExceptionHandler:
    """FastAPI全局异常处理器类"""

    # 1. 处理 FastAPI 原生 HTTPException
    @staticmethod
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "request_id": getattr(request.state, "request_id", "unknown")
                },
                "timestamp": datetime.now().isoformat()
            }
        )

    # 2. 处理 自定义业务异常 HelloAgentsException
    @staticmethod
    async def hello_agents_exception_handler(request: Request, exc: HelloAgentsException) -> JSONResponse:
        """统一处理所有业务异常（LLM/Agent/Config/Tool）"""
        return JSONResponse(
            status_code=exc.code,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": getattr(request.state, "request_id", "unknown")
                },
                "timestamp": datetime.now().isoformat()
            }
        )

    # 3. 处理 所有未捕获的未知异常
    @staticmethod
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"未捕获异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": 500,
                    "message": "服务器内部错误",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "trace_id": str(uuid.uuid4())
                },
                "timestamp": datetime.now().isoformat()
            }
        )


# ==================== 向后兼容函数 ====================
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return await ExceptionHandler.http_exception_handler(request, exc)


async def hello_agents_exception_handler(request: Request, exc: HelloAgentsException) -> JSONResponse:
    return await ExceptionHandler.hello_agents_exception_handler(request, exc)


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return await ExceptionHandler.general_exception_handler(request, exc)
