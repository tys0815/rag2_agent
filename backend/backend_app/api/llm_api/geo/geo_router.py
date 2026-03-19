"""GEO优化助手API路由
提供GEO优化软文生成功能，集成MCP服务和记忆管理
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

from helloAgents.agents.geo_optimization_agent import GeoOptimizationAgent
from helloAgents.core.llm import HelloAgentsLLM
from helloAgents.tools.registry import global_registry

logger = logging.getLogger(__name__)

# 创建路由器
geo_router = APIRouter()

# 全局Agent实例（单例模式）
_geo_agent = None

def get_geo_agent() -> GeoOptimizationAgent:
    """获取GEO优化助手实例（单例）"""
    global _geo_agent
    if _geo_agent is None:
        llm = HelloAgentsLLM()
        _geo_agent = GeoOptimizationAgent(
            name="geo_optimization_assistant",
            llm=llm,
            system_prompt="""你是一个专业的GEO优化助手，专门为产品生成高质量的软文并进行多平台优化。

你具备以下核心能力：
1. 🏗️ 内容结构化：通过Schema标记、实体关系梳理，让AI模型更易解析内容
2. 🏛️ 权威信号构建：在内容中融入可信来源（政策、数据、专家观点）提升采信概率
3. 🌐 多平台适配：同时针对搜索引擎（如百度、Google）与生成式AI平台（ChatGPT、豆包、DeepSeek）进行优化
4. 🔄 持续迭代：通过实时监测平台反馈数据，动态调整关键词与语义布局

具体技能：
1. 产品分析：深入理解产品特点、目标受众和市场定位
2. 软文创作：撰写吸引人、有说服力的营销内容
3. SEO优化：合理布局关键词，优化内容结构
4. AI优化：使内容易于AI解析和推荐
5. 数据驱动：基于行业数据和用户反馈优化内容

请基于提供的上下文信息，生成专业、高质量、经过GEO优化的软文。""",
            tool_registry=global_registry,
            default_agent_id="geo_optimizer"
        )
        logger.info("✅ GEO优化助手初始化完成")
    return _geo_agent


# ==================== 请求/响应模型 ====================

class GeoChatRequest(BaseModel):
    """GEO优化聊天请求"""
    text: str = Field(..., description="用户输入的产品需求文本")
    user_id: str = Field(..., description="用户ID（必需）")
    agent_id: Optional[str] = Field("geo_optimizer", description="助手ID（默认为geo_optimizer）")
    session_id: Optional[str] = Field(None, description="会话ID（可选，不传则自动生成）")
    enable_memory: Optional[bool] = Field(True, description="是否启用记忆")
    max_context_length: Optional[int] = Field(2000, description="最大上下文长度")

    class Config:
        schema_extra = {
            "example": {
                "text": "我们公司推出了一款新型智能空气净化器，具有高效过滤、静音运行、智能联动等特点，主要面向家庭用户和办公室",
                "user_id": "user_12345",
                "agent_id": "geo_optimizer",
                "session_id": "geo_session_abc123",
                "enable_memory": True,
                "max_context_length": 2000
            }
        }


class GeoChatResponse(BaseModel):
    """GEO优化聊天响应"""
    success: bool = Field(..., description="是否成功")
    data: str = Field(..., description="助手回答（GEO优化软文）")
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    agent_id: str = Field(..., description="助手ID")
    timestamp: str = Field(..., description="响应时间戳")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": "# 智能空气净化器：为家庭和办公室带来清新健康空气\n\n## 产品概述\n...",
                "session_id": "geo_session_abc123",
                "user_id": "user_12345",
                "agent_id": "geo_optimizer",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class ProductInfoRequest(BaseModel):
    """产品信息提取请求"""
    text: str = Field(..., description="用户输入的产品需求文本")


class ProductInfoResponse(BaseModel):
    """产品信息提取响应"""
    success: bool = Field(..., description="是否成功")
    extracted_info: dict = Field(..., description="提取的产品信息")
    timestamp: str = Field(..., description="响应时间戳")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "extracted_info": {
                    "product_name": "智能空气净化器",
                    "product_category": "科技产品",
                    "key_features": ["高效过滤", "静音运行", "智能联动"],
                    "target_audience": ["家庭用户", "办公室"],
                    "price_range": "中高端",
                    "competitors": ["小米空气净化器", "飞利浦空气净化器"],
                    "unique_selling_points": ["智能联动", "高效过滤"],
                    "industry_trends": ["智能化", "健康家居"],
                    "keywords": ["空气净化器", "智能家居", "健康空气"]
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class MemoryStatsRequest(BaseModel):
    """记忆统计请求"""
    user_id: str = Field(..., description="用户ID")
    agent_id: Optional[str] = Field("geo_optimizer", description="助手ID")
    session_id: Optional[str] = Field(None, description="会话ID")


class MemoryStatsResponse(BaseModel):
    """记忆统计响应"""
    success: bool = Field(..., description="是否成功")
    stats: dict = Field(..., description="统计信息")
    timestamp: str = Field(..., description="响应时间戳")


class ClearMemoriesRequest(BaseModel):
    """清空记忆请求"""
    user_id: str = Field(..., description="用户ID")
    agent_id: Optional[str] = Field("geo_optimizer", description="助手ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    confirm: bool = Field(False, description="确认执行")


class ClearMemoriesResponse(BaseModel):
    """清空记忆响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="结果消息")
    timestamp: str = Field(..., description="响应时间戳")


# ==================== API端点 ====================

@geo_router.post("/chat", response_model=GeoChatResponse)
async def geo_chat(request: Request, body: GeoChatRequest) -> GeoChatResponse:
    """
    GEO优化聊天接口

    支持：
    - 产品信息提取
    - MCP服务集成（行业数据、权威来源等）
    - 记忆管理
    - GEO优化软文生成
    """
    try:
        # 获取助手实例
        agent = get_geo_agent()

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

        return GeoChatResponse(
            success=True,
            data=response,
            session_id=session_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
        )

    except Exception as e:
        logger.error(f"GEO优化软文生成失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"GEO优化软文生成失败: {str(e)}"
        )


@geo_router.post("/extract_info", response_model=ProductInfoResponse)
async def extract_product_info(request: Request, body: ProductInfoRequest) -> ProductInfoResponse:
    """提取产品信息"""
    try:
        agent = get_geo_agent()

        # 调用信息提取方法
        extracted_info = agent._extract_product_info(body.text)

        return ProductInfoResponse(
            success=True,
            extracted_info=extracted_info,
            timestamp=request.state.timestamp if hasattr(request.state, 'timestamp') else "2024-01-01T12:00:00"
        )

    except Exception as e:
        logger.error(f"提取产品信息失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"提取产品信息失败: {str(e)}"
        )


@geo_router.post("/memory/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(request: Request, body: MemoryStatsRequest) -> MemoryStatsResponse:
    """获取记忆统计信息"""
    try:
        agent = get_geo_agent()
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


@geo_router.post("/memory/clear", response_model=ClearMemoriesResponse)
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

        agent = get_geo_agent()
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


@geo_router.get("/health")
async def health_check():
    """健康检查"""
    try:
        agent = get_geo_agent()
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