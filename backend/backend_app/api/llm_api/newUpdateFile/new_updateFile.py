
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends

from helloAgents.utils.helpers import get_file_path_from_upload_async

import logging

from helloAgents.tools.builtin.rag_tool import RAGTool
from helloAgents.tools.builtin.memory_tool import MemoryTool

from helloAgents.tools.registry import global_registry

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

new_updateFile_router = APIRouter()

from fastapi import UploadFile
from datetime import datetime



@new_updateFile_router.post("/new_update_file")
async def new_update_file(
    request: Request, 
    file: UploadFile = File(..., description="要上传的文件"),
    # 关键2：namespace 标注为 Form 类型（匹配前端 FormData 传参），并设置必填
    namespace: str = Form(..., description="知识库命名空间")
    
) -> dict:
    # rag_tool.run({
    #     "action": 'clear',
    #     "confirm": True, 
    #     "namespace": namespace
    # })
    rag_tool: RAGTool = global_registry.get_tool("rag")
    memory_tool: MemoryTool = global_registry.get_tool("memory")
    #
    memory_tool.run({
        "action": "stats",
        "user_id": namespace
    })

    #
    file_path = await get_file_path_from_upload_async(file, namespace=namespace, save_dir="./knowledge_base")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # result = rag_tool.add_document(file_path, namespace=namespace)
        memory_tool.run({
            "action": "add",
            "user_id": namespace,
            "memory_type": "semantic",
            "content": f"{now} 用户{namespace}已上传文件：{file.filename}，存储路径：{file_path}"
        })
        # print(f"✅ 文件上传结果: {"result"}")
        return {"success": True, "data": "result"}
    except Exception as e:
        memory_tool.run({
            "action": "add",
            "user_id": namespace,
            "memory_type": "perceptual",
            "content": f"{now} 用户{namespace}已上传文件：{file.filename}失败，转为感知记忆存储，存储路径：{file_path}",
            "file_path": file_path
        })
        return {"success": False, "data": "文件上传失败，已转为感知记忆存储", "error": str(e)}


    
    
