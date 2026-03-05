"""企业级RAG助手API路由
提供企业级多用户RAG助手功能，支持用户隔离、记忆管理和知识库检索
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

from helloAgents.agents.enterprise_rag_agent import EnterpriseRagAgent
from helloAgents.core.llm import HelloAgentsLLM
from helloAgents.tools.registry import global_registry

logger = logging.getLogger(__name__)

# 创建路由器
enterprise_router = APIRouter()

# 全局Agent实例（单例模式）
_enterprise_agent = None

def get_enterprise_agent() -> EnterpriseRagAgent:
    """获取企业级RAG助手实例（单例）"""
    global _enterprise_agent
    if _enterprise_agent is None:
        llm = HelloAgentsLLM()
        _enterprise_agent = EnterpriseRagAgent(
            name="enterprise_rag_assistant",
            llm=llm,
            system_prompt="""你是一个企业级RAG助手，专门为企业用户提供智能问答服务。
你具备以下能力：
1. 多用户隔离：为不同用户提供独立的知识库和记忆
2. 记忆管理：能够记住用户的偏好和历史对话
3. 知识库检索：能够从用户上传的文档中查找相关信息
4. 专业回答：提供准确、全面、专业的回答

请根据用户的身份、历史对话和知识库内容提供个性化的服务。""",
            tool_registry=global_registry,
            default_agent_id="enterprise_rag"
        )
        logger.info("✅ 企业级RAG助手初始化完成")
    return _enterprise_agent


# ==================== 请求/响应模型 ====================

class EnterpriseChatRequest(BaseModel):
    """企业级聊天请求"""
    text: str = Field(..., description="用户输入的文本")
    user_id: str = Field(..., description="用户ID（必需）")
    agent_id: Optional[str] = Field("enterprise_rag", description="助手ID（默认为enterprise_rag）")
    session_id: Optional[str] = Field(None, description="会话ID（可选，不传则自动生成）")
    namespace: Optional[str] = Field(None, description="知识库命名空间（默认为user_id）")
    enable_memory: Optional[bool] = Field(True, description="是否启用记忆")
    enable_rag: Optional[bool] = Field(True, description="是否启用RAG检索")
    max_context_length: Optional[int] = Field(2000, description="最大上下文长度")

    class Config:
        schema_extra = {
            "example": {
                "text": "请介绍一下我们公司的产品",
                "user_id": "user_12345",
                "agent_id": "enterprise_rag",
                "session_id": "session_abc123",
                "namespace": "company_knowledge",
                "enable_memory": True,
                "enable_rag": True,
                "max_context_length": 2000
            }
        }


class EnterpriseChatResponse(BaseModel):
    """企业级聊天响应"""
    success: bool = Field(..., description="是否成功")
    data: str = Field(..., description="助手回答")
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    agent_id: str = Field(..., description="助手ID")
    namespace: str = Field(..., description="知识库命名空间")
    timestamp: str = Field(..., description="响应时间戳")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": "根据您的知识库，公司的主要产品包括...",
                "session_id": "session_abc123",
                "user_id": "user_12345",
                "agent_id": "enterprise_rag",
                "namespace": "company_knowledge",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class MemoryStatsRequest(BaseModel):
    """记忆统计请求"""
    user_id: str = Field(..., description="用户ID")
    agent_id: Optional[str] = Field("enterprise_rag", description="助手ID")
    session_id: Optional[str] = Field(None, description="会话ID")


class MemoryStatsResponse(BaseModel):
    """记忆统计响应"""
    success: bool = Field(..., description="是否成功")
    stats: dict = Field(..., description="统计信息")
    timestamp: str = Field(..., description="响应时间戳")


class ClearMemoriesRequest(BaseModel):
    """清空记忆请求"""
    user_id: str = Field(..., description="用户ID")
    agent_id: Optional[str] = Field("enterprise_rag", description="助手ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    confirm: bool = Field(False, description="确认执行")


class ClearMemoriesResponse(BaseModel):
    """清空记忆响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="结果消息")
    timestamp: str = Field(..., description="响应时间戳")


# ==================== API端点 ====================

@enterprise_router.post("/chat", response_model=EnterpriseChatResponse)
async def enterprise_chat(request: Request, body: EnterpriseChatRequest) -> EnterpriseChatResponse:
    """
    企业级聊天接口

    支持：
    - 多用户隔离
    - 记忆管理
    - 知识库检索
    - 会话管理
    """
    try:
        # 获取助手实例
        agent = get_enterprise_agent()

        # 调用助手
        response = agent.run(
            input_text=body.text,
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id,
            namespace=body.namespace,
            enable_memory=body.enable_memory,
            enable_rag=body.enable_rag,
            max_context_length=body.max_context_length
        )

        # 使用传入的session_id或生成新的
        session_id = body.session_id or "auto_generated"

        return EnterpriseChatResponse(
            success=True,
            data=response,
            session_id=session_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            namespace=body.namespace or body.user_id,
            timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
        )

    except Exception as e:
        logger.error(f"企业级聊天失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"企业级聊天失败: {str(e)}"
        )


@enterprise_router.post("/memory/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(request: Request, body: MemoryStatsRequest) -> MemoryStatsResponse:
    """获取记忆统计信息"""
    try:
        agent = get_enterprise_agent()
        stats = agent.get_memory_stats(
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id
        )

        return MemoryStatsResponse(
            success=True,
            stats=stats,
            timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
        )

    except Exception as e:
        logger.error(f"获取记忆统计失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"获取记忆统计失败: {str(e)}"
        )


@enterprise_router.post("/memory/clear", response_model=ClearMemoriesResponse)
async def clear_memories(request: Request, body: ClearMemoriesRequest) -> ClearMemoriesResponse:
    """清空记忆"""
    try:
        # 安全检查
        if not body.confirm:
            return ClearMemoriesResponse(
                success=False,
                message="请设置confirm=true以确认清空记忆",
                timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
            )

        agent = get_enterprise_agent()
        success = agent.clear_memories(
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id
        )

        if success:
            return ClearMemoriesResponse(
                success=True,
                message="记忆已成功清空",
                timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
            )
        else:
            return ClearMemoriesResponse(
                success=False,
                message="清空记忆失败",
                timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
            )

    except Exception as e:
        logger.error(f"清空记忆失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"清空记忆失败: {str(e)}"
        )


@enterprise_router.post("/memory/consolidate")
async def consolidate_memories(request: Request, body: MemoryStatsRequest):
    """整合重要记忆到长期记忆"""
    try:
        agent = get_enterprise_agent()
        count = agent.consolidate_important_memories(
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id
        )

        return {
            "success": True,
            "message": f"已整合 {count} 条重要记忆到长期记忆",
            "count": count,
            "timestamp": request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
        }

    except Exception as e:
        logger.error(f"整合记忆失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"整合记忆失败: {str(e)}"
        )


@enterprise_router.get("/health")
async def health_check():
    """健康检查"""
    try:
        agent = get_enterprise_agent()
        return {
            "status": "healthy",
            "agent": agent.name,
            "tools_available": {
                "rag": agent.rag_tool is not None,
                "memory": agent.memory_tool is not None
            },
            "timestamp": "2024-01-01T12:00:00"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"健康检查失败: {str(e)}"
        )