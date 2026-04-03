import re
import asyncio
import nest_asyncio
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
nest_asyncio.apply()

# 评估工具
from helloAgents.tools.builtin.bfcl_evaluation_tool import BFCLEvaluationTool

logger = logging.getLogger(__name__)
universal_router = APIRouter()

# ==================== 请求/响应模型 ====================
class UniversalChatRequest(BaseModel):
    text: str = Field(..., description="用户输入的文本", max_length=5000)
    user_id: str = Field(..., description="用户ID（必需）")
    agent_id: Optional[str] = Field("universal_assistant")
    session_id: Optional[str] = Field(None)
    enable_memory: Optional[bool] = Field(True)
    enable_rag: Optional[bool] = Field(True)
    max_context_length: Optional[int] = Field(2000, ge=100, le=8000)
    tool_choice: Optional[str] = Field("auto")

    @field_validator("user_id")
    def validate_user_id(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", v):
            raise ValueError("用户ID格式错误")
        return v

class UniversalChatResponse(BaseModel):
    success: bool
    data: str
    session_id: str
    user_id: str
    agent_id: str
    tool_calls: Optional[int]
    timestamp: str

class ToolStatisticsResponse(BaseModel):
    success: bool
    statistics: Dict[str, Any]
    timestamp: str

# ==================== 【终极正确】异步Agent → 同步适配（FastAPI安全版）====================
class SyncAgentWrapper:
    def __init__(self, async_agent):
        self.agent = async_agent
        self.name = async_agent.name

    def run(self, prompt, **kwargs):
        # ✅ 唯一在 FastAPI 中安全运行异步 Agent 的方式
        return asyncio.run(self.agent.run(prompt,** kwargs))

# ==================== 正常聊天接口 ====================
@universal_router.post("/chat", response_model=UniversalChatResponse)
async def universal_chat(
    request: Request,
    body: UniversalChatRequest
) -> UniversalChatResponse:
    try:
        agent = KnowledgeBaseAssistant(
            name="通用助手",
            llm=HelloAgentsLLM(),
            tool_registry=global_registry
        )

        response = await agent.run(
            input_text=body.text,
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id
        )

        session_id = body.session_id or f"ses_{int(datetime.now().timestamp())}"

        return UniversalChatResponse(
            success=True,
            data=response,
            session_id=session_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            tool_calls=0,
            timestamp=datetime.now().isoformat()
        )

    except HelloAgentsException as e:
        logger.error(f"业务异常: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"聊天失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务内部错误")

# ==================== 【独立】BFCL评估接口（不影响正常业务） ====================
@universal_router.get("/bfcl-eval")
def run_bfcl_evaluation():  # 注意：这里必须是 同步函数！
    try:
        # 初始化
        llm = HelloAgentsLLM()
        agent = KnowledgeBaseAssistant(name="BFCL-Eval-Agent", llm=llm)
        sync_agent = SyncAgentWrapper(agent)

        # 执行评估
        bfcl_tool = BFCLEvaluationTool()
        result = bfcl_tool.run(
            agent=sync_agent,
            category="simple_python",
            max_samples=0,
            run_official_eval=True
        )

        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}