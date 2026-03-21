from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from typing import List
from datetime import datetime
import logging
import os
import asyncio
import json  # 必须加

from helloAgents.tools.async_executor import AsyncToolExecutor
from helloAgents.utils.helpers import get_file_path_from_upload_async
from helloAgents.tools.builtin.rag_tool import RAGTool
from helloAgents.tools.builtin.memory_tool import MemoryTool
from helloAgents.tools.registry import global_registry

logger = logging.getLogger(__name__)
new_updateFile_router = APIRouter()

# 并发配置（根据你的机器调整）
MAX_PARALLEL_FILES = 4


@new_updateFile_router.post("/new_update_file")
async def new_update_file(
    request: Request,
    files: List[UploadFile] = File(..., description="上传文件列表"),
    namespace: str = Form(..., description="命名空间")
) -> dict:
    """
    🔥 最终企业版：
    异步多文件上传 + 并发保存 + 多线程并行存入向量库 + 结构化返回
    """
    # rag_tool: RAGTool = global_registry.get_tool("rag")
    # rag_tool._clear_knowledge_base(confirm=True)  # 🔥 开发阶段先清库，正式环境请去掉
    memory_tool: MemoryTool = global_registry.get_tool("memory")

    # ======================
    # 1. 基础校验
    # ======================
    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个文件")
    if not namespace.strip():
        raise HTTPException(status_code=400, detail="命名空间不能为空")

    # ======================
    # 2. 异步并发保存所有文件
    # ======================
    async def save_single_file(upload_file: UploadFile):
        try:
            file_path = await get_file_path_from_upload_async(
                upload_file, namespace=namespace, save_dir="./knowledge_base"
            )
            return {
                "filename": upload_file.filename,
                "file_path": file_path,
                "success": True
            }
        except Exception as e:
            logger.error(f"文件保存失败 {upload_file.filename}: {str(e)}")
            return {
                "filename": upload_file.filename,
                "error": str(e),
                "success": False
            }

    # 并发保存
    save_tasks = [save_single_file(f) for f in files]
    save_results = await asyncio.gather(*save_tasks)

    # 分类成功/失败
    saved_files = [r for r in save_results if r["success"]]
    save_errors = [r for r in save_results if not r["success"]]

    # 记录保存失败记忆
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for err in save_errors:
        memory_tool.run({
            "action": "add",
            "user_id": namespace,
            "memory_type": "perceptual",
            "content": f"{now} 用户{namespace}上传文件 {err['filename']} 保存失败：{err['error']}",
            "file_path": ""
        })

    if not saved_files:
        return {
            "success": False,
            "msg": "所有文件保存失败",
            "results": [],
            "errors": save_errors
        }

    # ======================
    # 3. 🔥 核心：使用你的 AsyncToolExecutor 并行入库向量库
    # ======================
    file_paths = [f["file_path"] for f in saved_files]

    # 构造批量任务（每个文件 = 1个RAG入库任务）
    rag_tasks = [
        {
            "tool_name": "rag",
            "input_data": {
                "action": "add_document",
                "file_path": fp,
                "namespace": namespace
            }
        }
        for fp in file_paths
    ]

    # 并行执行（真正多文件同时入库）
    executor = AsyncToolExecutor(global_registry, max_workers=MAX_PARALLEL_FILES)
    execute_results = await executor.execute_tools_parallel(rag_tasks)

    # ======================
    # 4. 根据执行结果记录记忆
    # ======================
    success_list = []
    fail_list = []

    for idx, file_info in enumerate(saved_files):
        res = execute_results[idx]
        filename = file_info["filename"]
        file_path = file_info["file_path"]

        if res["status"] == "success":
            # 成功 → 语义记忆
            success_list.append({
                "filename": filename,
                "file_path": file_path
            })
            memory_tool.run({
                "action": "add",
                "user_id": namespace,
                "memory_type": "semantic",
                "content": f"{now} 用户{namespace}已上传文件：{filename}，已入库知识库",
                "file_path": file_path
            })
        else:
            # 失败 → 感知记忆
            fail_list.append({
                "filename": filename,
                "error": res["result"]
            })
            memory_tool.run({
                "action": "add",
                "user_id": namespace,
                "memory_type": "perceptual",
                "content": f"{now} 用户{namespace}文件 {filename} 处理失败：{res['result']}",
                "file_path": file_path
            })

    # ======================
    # 5. 统一返回
    # ======================
    return {
        "success": len(fail_list) == 0,
        "msg": f"成功 {len(success_list)} 个，失败 {len(fail_list)} 个",
        "data": {
            "total_files": len(files),
            "saved_success": len(saved_files),
            "vector_success": len(success_list),
            "vector_failed": len(fail_list),
            "namespace": namespace
        },
        "results": success_list,
        "errors": fail_list
    }