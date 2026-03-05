
from fastapi import APIRouter, Request

from helloAgents.tools.registry import global_registry
from helloAgents.agents.rag_agent import RagAgent
from helloAgents.core.llm import HelloAgentsLLM



import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

new_chat_router = APIRouter()

# 定义请求体模型（Pydantic Model）
class ChatRequest(BaseModel):
    """定义聊天请求的参数结构"""
    text: str  # 对应前端发送的 text 字段
    namespace: str = "default"  # 可选的命名空间参数，默认为 "default"
    agent_id: str = "default"  # 可选的代理ID参数，默认为 "default"
    session_id: str = "default"  # 可选的会话ID参数，默认为 "default"

agent = RagAgent(name="rag_agent", llm=HelloAgentsLLM(), tool_registry=global_registry)

@new_chat_router.post("/completions")
def new_chat(request: Request, body: ChatRequest) -> dict:
    try:
        text: str = body.text
        namespace: str = body.namespace
        agent_id: str = body.agent_id
        session_id: str = body.session_id
        response = agent.run(text, namespace=namespace, agent_id=agent_id, session_id=session_id)
        print(f"✅ RAG Agent response: {response}")
        return {"success": True, "data": response}
    except Exception as e:
        print(f"❌ Error processing chat request: {e}")
        return {"success": False, "error": str(e)}




