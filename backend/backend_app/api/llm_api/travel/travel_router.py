"""旅游路线规划助手API路由
提供旅游路线规划功能，集成MCP服务和记忆管理
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

from helloAgents.agents.travel_planner_agent import TravelPlannerAgent
from helloAgents.core.llm import HelloAgentsLLM
from helloAgents.tools.registry import global_registry

logger = logging.getLogger(__name__)

# 创建路由器
travel_router = APIRouter()

# 全局Agent实例（单例模式）
_travel_agent = None

def get_travel_agent() -> TravelPlannerAgent:
    """获取旅游路线规划助手实例（单例）"""
    global _travel_agent
    if _travel_agent is None:
        llm = HelloAgentsLLM()
        _travel_agent = TravelPlannerAgent(
            name="travel_planner_assistant",
            llm=llm,
            system_prompt="""你是一个专业的旅游路线规划助手，专门为用户提供个性化的旅游路线规划服务。
你具备以下能力：
1. 🧳 信息提取：能从用户需求中提取地点、时间、天数、预算、兴趣等关键信息
2. 🌐 数据整合：能整合天气、酒店、景点、交通等外部服务信息
3. 📅 路线规划：能制定详细、可行的每日行程安排
4. 💡 实用建议：能提供游览建议、注意事项、必备物品等实用信息
5. 💰 预算管理：能根据用户预算推荐合适的住宿、餐饮和活动
6. 🎯 个性化：能根据用户兴趣（历史文化、自然风光、美食购物等）定制路线
7. 🧠 记忆管理：能记住用户的偏好和历史对话，提供更个性化的服务

请为用户生成专业、详细、个性化的旅游路线方案。""",
            tool_registry=global_registry,
            default_agent_id="travel_planner"
        )
        logger.info("✅ 旅游路线规划助手初始化完成")
    return _travel_agent


# ==================== 请求/响应模型 ====================

class TravelChatRequest(BaseModel):
    """旅游路线规划聊天请求"""
    text: str = Field(..., description="用户输入的旅游需求文本")
    user_id: str = Field(..., description="用户ID（必需）")
    agent_id: Optional[str] = Field("travel_planner", description="助手ID（默认为travel_planner）")
    session_id: Optional[str] = Field(None, description="会话ID（可选，不传则自动生成）")
    enable_memory: Optional[bool] = Field(True, description="是否启用记忆")
    max_context_length: Optional[int] = Field(2000, description="最大上下文长度")

    class Config:
        schema_extra = {
            "example": {
                "text": "我想去北京旅游3天，预算5000元，喜欢历史文化和美食",
                "user_id": "user_12345",
                "agent_id": "travel_planner",
                "session_id": "session_abc123",
                "enable_memory": True,
                "max_context_length": 2000
            }
        }


class TravelChatResponse(BaseModel):
    """旅游路线规划聊天响应"""
    success: bool = Field(..., description="是否成功")
    data: str = Field(..., description="助手回答（旅游路线方案）")
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    agent_id: str = Field(..., description="助手ID")
    timestamp: str = Field(..., description="响应时间戳")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": "以下是您的北京3日游路线方案：\n\n**第一天：历史文化探索**\n- 上午：参观故宫...",
                "session_id": "session_abc123",
                "user_id": "user_12345",
                "agent_id": "travel_planner",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class TravelInfoRequest(BaseModel):
    """旅游信息提取请求"""
    text: str = Field(..., description="用户输入的旅游需求文本")


class TravelInfoResponse(BaseModel):
    """旅游信息提取响应"""
    success: bool = Field(..., description="是否成功")
    extracted_info: dict = Field(..., description="提取的旅游信息")
    timestamp: str = Field(..., description="响应时间戳")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "extracted_info": {
                    "locations": ["北京"],
                    "duration_days": 3,
                    "budget": "5000元",
                    "interests": ["历史文化", "美食"],
                    "travelers": None
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class MemoryStatsRequest(BaseModel):
    """记忆统计请求"""
    user_id: str = Field(..., description="用户ID")
    agent_id: Optional[str] = Field("travel_planner", description="助手ID")
    session_id: Optional[str] = Field(None, description="会话ID")


class MemoryStatsResponse(BaseModel):
    """记忆统计响应"""
    success: bool = Field(..., description="是否成功")
    stats: dict = Field(..., description="统计信息")
    timestamp: str = Field(..., description="响应时间戳")


class ClearMemoriesRequest(BaseModel):
    """清空记忆请求"""
    user_id: str = Field(..., description="用户ID")
    agent_id: Optional[str] = Field("travel_planner", description="助手ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    confirm: bool = Field(False, description="确认执行")


class ClearMemoriesResponse(BaseModel):
    """清空记忆响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="结果消息")
    timestamp: str = Field(..., description="响应时间戳")


# ==================== API端点 ====================

@travel_router.post("/chat", response_model=TravelChatResponse)
async def travel_chat(request: Request, body: TravelChatRequest) -> TravelChatResponse:
    """
    旅游路线规划聊天接口

    支持：
    - 旅游信息提取
    - MCP服务集成（天气、酒店、景点等）
    - 记忆管理
    - 个性化路线规划
    """
    try:
        # 获取助手实例
        agent = get_travel_agent()

        # 调用助手
        response = agent.run(
            input_text=body.text,
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id,
            enable_memory=body.enable_memory,
            max_context_length=body.max_context_length
        )

        # 使用传入的session_id或生成新的
        session_id = body.session_id or "auto_generated"

        return TravelChatResponse(
            success=True,
            data=response,
            session_id=session_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
        )

    except Exception as e:
        logger.error(f"旅游路线规划失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"旅游路线规划失败: {str(e)}"
        )


@travel_router.post("/extract_info", response_model=TravelInfoResponse)
async def extract_travel_info(request: Request, body: TravelInfoRequest) -> TravelInfoResponse:
    """提取旅游信息"""
    try:
        agent = get_travel_agent()

        # 调用信息提取方法
        extracted_info = agent._extract_travel_info(body.text)

        return TravelInfoResponse(
            success=True,
            extracted_info=extracted_info,
            timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
        )

    except Exception as e:
        logger.error(f"提取旅游信息失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"提取旅游信息失败: {str(e)}"
        )


@travel_router.post("/memory/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(request: Request, body: MemoryStatsRequest) -> MemoryStatsResponse:
    """获取记忆统计信息"""
    try:
        agent = get_travel_agent()
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


@travel_router.post("/memory/clear", response_model=ClearMemoriesResponse)
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

        agent = get_travel_agent()
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


@travel_router.get("/health")
async def health_check():
    """健康检查"""
    try:
        agent = get_travel_agent()
        return {
            "status": "healthy",
            "agent": agent.name,
            "tools_available": {
                "mcp": agent.mcp_tool is not None,
                "memory": agent.memory_tool is not None
            },
            "timestamp": "2024-01-01T12:00:00"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"健康检查失败: {str(e)}"
        )