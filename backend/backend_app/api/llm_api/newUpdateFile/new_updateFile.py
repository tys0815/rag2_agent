
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends
from typing import List

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
    file: List[UploadFile] = File(..., description="要上传的文件列表（支持多文件或文件夹）"),
    # 关键2：namespace 标注为 Form 类型（匹配前端 FormData 传参），并设置必填
    namespace: str = Form(..., description="知识库命名空间")

) -> dict:

    rag_tool: RAGTool = global_registry.get_tool("rag")
    # rag_tool.run({
    #     "action": 'clear',
    #     "confirm": True,
    #     "namespace": namespace
    # })
    memory_tool: MemoryTool = global_registry.get_tool("memory")
    #
    # memory_tool.run({
    #     "action": "stats",
    #     "user_id": namespace
    # })

    results = []
    errors = []

    for upload_file in file:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            file_path = await get_file_path_from_upload_async(upload_file, namespace=namespace, save_dir="./knowledge_base")
            result = rag_tool.add_document(file_path, namespace=namespace)
            memory_tool.run({
                "action": "add",
                "user_id": namespace,
                "memory_type": "semantic",
                "content": f"{now} 用户{namespace}已上传文件：{upload_file.filename}，存储路径：{file_path}"
            })
            results.append({
                "filename": upload_file.filename,
                "success": True,
                "data": result,
                "file_path": file_path
            })
        except Exception as e:
            # 如果文件保存失败，file_path 可能不存在
            file_path = ""
            try:
                # 尝试保存文件获取路径，用于记忆
                file_path = await get_file_path_from_upload_async(upload_file, namespace=namespace, save_dir="./knowledge_base")
            except Exception:
                pass
            memory_tool.run({
                "action": "add",
                "user_id": namespace,
                "memory_type": "perceptual",
                "content": f"{now} 用户{namespace}已上传文件：{upload_file.filename}失败，转为感知记忆存储，存储路径：{file_path}",
                "file_path": file_path
            })
            errors.append({
                "filename": upload_file.filename,
                "error": str(e)
            })

    if errors:
        return {
            "success": False,
            "data": f"部分文件上传失败，成功 {len(results)} 个，失败 {len(errors)} 个",
            "results": results,
            "errors": errors
        }
    else:
        return {
            "success": True,
            "data": f"所有文件上传成功，共 {len(results)} 个",
            "results": results
        }


    
    
