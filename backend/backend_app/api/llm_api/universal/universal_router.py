"""全能企业级助手API路由
提供全能企业级助手功能，集成RAG、记忆、命令行、MCP等工具
"""

import re
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
import logging

from helloAgents.agents.universal_enterprise_agent import KnowledgeBaseAssistant
from helloAgents.core.llm import HelloAgentsLLM
from helloAgents.tools.registry import global_registry
from helloAgents.core.exceptions import (
    HelloAgentsException,
    ToolException
)

logger = logging.getLogger(__name__)

# 创建路由器
universal_router = APIRouter()

# ==================== 请求/响应模型 ====================

class UniversalChatRequest(BaseModel):
    """全能聊天请求"""
    text: str = Field(..., description="用户输入的文本", max_length=5000)
    user_id: str = Field(..., description="用户ID（必需）")
    agent_id: Optional[str] = Field("universal_assistant", description="助手ID（默认为universal_assistant）")
    session_id: Optional[str] = Field(None, description="会话ID（可选，不传则自动生成）")
    namespace: Optional[str] = Field(None, description="知识库命名空间（默认为user_id）")
    enable_memory: Optional[bool] = Field(True, description="是否启用记忆")
    enable_rag: Optional[bool] = Field(True, description="是否启用RAG检索")
    max_context_length: Optional[int] = Field(2000, description="最大上下文长度", ge=100, le=8000)
    tool_choice: Optional[str] = Field("auto", description="工具选择策略")

    @field_validator("user_id")
    def validate_user_id(cls, v):
        """验证用户ID格式"""
        if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", v):
            raise ValueError("用户ID只能包含字母、数字、下划线和短横线，长度3-50")
        return v

    @field_validator("namespace")
    def validate_namespace(cls, v):
        """验证命名空间格式"""
        if v and not re.match(r"^[a-zA-Z0-9_-]{1,100}$", v):
            raise ValueError("命名空间只能包含字母、数字、下划线和短横线，长度1-100")
        return v

    class Config:
        schema_extra = {
            "example": {
                "text": "请搜索最新的AI发展动态，并总结成报告",
                "user_id": "user_12345",
                "agent_id": "universal_assistant",
                "session_id": "session_abc123",
                "namespace": "company_knowledge",
                "enable_memory": True,
                "enable_rag": True,
                "max_context_length": 2000,
                "tool_choice": "auto"
            }
        }


class UniversalChatResponse(BaseModel):
    """全能聊天响应"""
    success: bool = Field(..., description="是否成功")
    data: str = Field(..., description="助手回答")
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    agent_id: str = Field(..., description="助手ID")
    namespace: str = Field(..., description="知识库命名空间")
    tool_calls: Optional[int] = Field(None, description="工具调用次数")
    timestamp: str = Field(..., description="响应时间戳")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": "根据最新搜索，AI发展动态如下...",
                "session_id": "session_abc123",
                "user_id": "user_12345",
                "agent_id": "universal_assistant",
                "namespace": "company_knowledge",
                "tool_calls": 2,
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class ToolStatisticsResponse(BaseModel):
    """工具统计响应"""
    success: bool = Field(..., description="是否成功")
    statistics: Dict[str, Any] = Field(..., description="统计信息")
    timestamp: str = Field(..., description="响应时间戳")






# ==================== API端点 ====================

@universal_router.post("/chat", response_model=UniversalChatResponse)
async def universal_chat(
    request: Request,
    body: UniversalChatRequest
) -> UniversalChatResponse:
    """
    全能聊天接口

    支持：
    - 多用户隔离
    - 记忆管理
    - 知识库检索
    - 命令行操作
    - 实时搜索
    - 多工具自动调用
    """
    try:
        agent = KnowledgeBaseAssistant(
            name="通用助手",
            llm=HelloAgentsLLM(),
            tool_registry=global_registry
        )
        # 调用助手（通过依赖注入获取）
        response = agent.run(
            input_text=body.text,
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id
        )

        # 使用传入的session_id或生成新的
        session_id = body.session_id or f"universal_session_{datetime.now().timestamp():.0f}"

        return UniversalChatResponse(
            success=True,
            data=response,
            session_id=session_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            namespace=body.namespace or body.user_id,
            tool_calls=0,  # 暂时不统计，后续可以增强
            timestamp=datetime.now().isoformat()
        )

    except HelloAgentsException as e:
        logger.error(f"全能聊天业务异常: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"全能聊天失败: {e}", exc_info=True)
        raise