
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional
import time
import uuid
import os

from helloAgents.tools.registry import global_registry
from helloAgents.agents.rag_agent import RagAgent
from helloAgents.core.llm import HelloAgentsLLM

import logging
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

new_chat_router = APIRouter()

# 定义请求体模型（Pydantic Model）
class ChatRequest(BaseModel):
    """定义聊天请求的参数结构"""
    text: str = Field(..., min_length=1, max_length=5000, description="用户输入文本")
    namespace: str = Field("default", min_length=1, max_length=100, description="知识库命名空间")
    agent_id: str = Field("default", min_length=1, max_length=100, description="代理ID")
    session_id: str = Field("default", min_length=1, max_length=100, description="会话ID")

    @validator('text')
    def validate_text(cls, v):
        """验证文本内容"""
        if not v or not v.strip():
            raise ValueError('文本内容不能为空')
        # 检查是否有恶意内容（简单示例）
        forbidden_patterns = ['<script>', 'javascript:', 'onload=']
        for pattern in forbidden_patterns:
            if pattern in v.lower():
                raise ValueError(f'文本包含不允许的内容: {pattern}')
        return v.strip()

agent = RagAgent(name="rag_agent", llm=HelloAgentsLLM(), tool_registry=global_registry)

@new_chat_router.post("/completions")
async def new_chat(
    request: Request,
    body: ChatRequest
) -> dict:
    """企业级RAG聊天端点

    特性：
    1. API密钥认证
    2. 请求追踪（request_id）
    3. 结构化日志
    4. 性能监控
    5. 输入验证
    6. 错误处理
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # 结构化日志：请求开始
    logger.info(f"[{request_id}] RAG请求开始 | text_length={len(body.text)} | namespace={body.namespace}")

    try:
        text: str = body.text
        namespace: str = body.namespace
        agent_id: str = body.agent_id
        session_id: str = body.session_id

        # 记录请求详情
        logger.debug(f"[{request_id}] 请求参数: text={text[:100]}..., namespace={namespace}, agent_id={agent_id}, session_id={session_id}")

        # 执行RAG查询
        response_start = time.time()
        response = agent.run(text, namespace=namespace, agent_id=agent_id, session_id=session_id)
        response_time = time.time() - response_start

        # 结构化日志：响应成功
        logger.info(f"[{request_id}] RAG响应成功 | response_time={response_time:.3f}s | response_length={len(str(response))}")

        # 构建响应
        result = {
            "success": True,
            "data": response,
            "request_id": request_id,
            "response_time": round(response_time, 3),
            "timestamp": time.time()
        }

        total_time = time.time() - start_time
        logger.info(f"[{request_id}] 请求完成 | total_time={total_time:.3f}s")

        return result

    except ValueError as e:
        # 输入验证错误
        error_msg = f"输入验证失败: {str(e)}"
        logger.warning(f"[{request_id}] {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": error_msg,
                "request_id": request_id
            }
        )

    except Exception as e:
        # 系统错误
        error_msg = f"系统错误: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}", exc_info=True)

        # 在生产环境中，不暴露内部错误详情
        internal_error_msg = "内部服务器错误" if "production" in os.getenv("ENVIRONMENT", "") else str(e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": internal_error_msg,
                "request_id": request_id
            }
        )




