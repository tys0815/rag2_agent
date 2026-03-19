"""全能企业级助手API路由
提供全能企业级助手功能，集成RAG、记忆、命令行、MCP等工具
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from helloAgents.agents.universal_enterprise_agent import UniversalEnterpriseAgent, create_universal_assistant
from helloAgents.core.llm import HelloAgentsLLM
from helloAgents.tools.registry import global_registry

logger = logging.getLogger(__name__)

# 创建路由器
universal_router = APIRouter()

# 全局Agent实例（单例模式）
_universal_agent = None


def get_universal_agent() -> UniversalEnterpriseAgent:
    """获取全能企业级助手实例（单例）"""
    global _universal_agent
    if _universal_agent is None:
        _universal_agent = create_universal_assistant(
            name="universal_enterprise_assistant",
            llm=HelloAgentsLLM(),
            system_prompt="""你是一个全能企业级AI助手，

## 输出要求
1. 提供清晰、准确的回答
2. 保持专业、友好的语气
3. 对复杂任务分步骤解释

现在开始为用户提供全面的AI助手服务！""",
            workspace="./workspace",
            tool_registry=global_registry
        )
        logger.info("✅ 全能企业级助手初始化完成")
        logger.info(f"📋 可用工具: {_universal_agent.list_tools()}")
    return _universal_agent


# ==================== 请求/响应模型 ====================

class UniversalChatRequest(BaseModel):
    """全能聊天请求"""
    text: str = Field(..., description="用户输入的文本")
    user_id: str = Field(..., description="用户ID（必需）")
    agent_id: Optional[str] = Field("universal_assistant", description="助手ID（默认为universal_assistant）")
    session_id: Optional[str] = Field(None, description="会话ID（可选，不传则自动生成）")
    namespace: Optional[str] = Field(None, description="知识库命名空间（默认为user_id）")
    enable_memory: Optional[bool] = Field(True, description="是否启用记忆")
    enable_rag: Optional[bool] = Field(True, description="是否启用RAG检索")
    max_context_length: Optional[int] = Field(2000, description="最大上下文长度")
    tool_choice: Optional[str] = Field("auto", description="工具选择策略")

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
async def universal_chat(request: Request, body: UniversalChatRequest) -> UniversalChatResponse:
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
        # 获取助手实例
        agent = get_universal_agent()

        # 调用助手
        response = agent.run_with_context(
            input_text=body.text,
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id,
            namespace=body.namespace,
            enable_memory=body.enable_memory,
            enable_rag=body.enable_rag,
            tool_choice=body.tool_choice
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

    except Exception as e:
        logger.error(f"全能聊天失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"全能聊天失败: {str(e)}"
        )


@universal_router.get("/tools", response_model=ToolStatisticsResponse)
async def get_tool_statistics():
    """获取工具统计信息"""
    try:
        agent = get_universal_agent()
        stats = agent.get_tool_statistics()

        return ToolStatisticsResponse(
            success=True,
            statistics=stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"获取工具统计失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"获取工具统计失败: {str(e)}"
        )






@universal_router.get("/health")
async def health_check():
    """健康检查"""
    try:
        agent = get_universal_agent()
        tools = agent.list_tools()

        return {
            "status": "healthy",
            "agent": agent.name,
            "total_tools": len(tools),
            "core_tools": ["rag", "memory", "terminal", "search", "calculator", "note"],
            "available_tools": tools[:10],  # 只显示前10个
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"健康检查失败: {str(e)}"
        )


@universal_router.get("/tools/list")
async def list_tools():
    """列出所有可用工具"""
    try:
        agent = get_universal_agent()
        tools = agent.list_tools()

        return {
            "success": True,
            "total_tools": len(tools),
            "tools": tools,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"列出工具失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"列出工具失败: {str(e)}"
        )