from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from typing import List
from datetime import datetime
import logging
import os
import asyncio
import hashlib
import json

from helloAgents.tools.async_executor import AsyncToolExecutor
from helloAgents.utils.helpers import get_file_path_from_upload_async
from helloAgents.tools.builtin.rag_tool import RAGTool
from helloAgents.tools.builtin.memory_tool import MemoryTool
from helloAgents.tools.registry import global_registry

logger = logging.getLogger(__name__)
new_updateFile_router = APIRouter()

MAX_PARALLEL_FILES = 4


# ------------------------------
# 🔥 企业级文件去重 + 版本化
# ------------------------------
def calculate_content_hash(content: bytes) -> str:
    """计算文件内容 SHA256 哈希（唯一标识）"""
    return hashlib.sha256(content).hexdigest()


async def save_file_with_version_and_deduplicate(
    upload_file: UploadFile,
    namespace: str,
    save_dir: str = "./knowledge_base"
) -> dict:
    """
    企业级文件保存：
    1. 相同内容自动去重
    2. 同名不同内容自动版本化
    3. 永不覆盖
    """
    try:
        # 读取内容
        content = await upload_file.read()
        file_hash = calculate_content_hash(content)
        original_name = upload_file.filename.strip()
        name, ext = os.path.splitext(original_name)

        # 目录
        ns_path = os.path.join(save_dir, namespace)
        os.makedirs(ns_path, exist_ok=True)

        # --------------------------
        # 1. 内容去重（相同内容直接跳过）
        # --------------------------
        for fname in os.listdir(ns_path):
            fpath = os.path.join(ns_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "rb") as f:
                    existing_hash = calculate_content_hash(f.read())
                if existing_hash == file_hash:
                    return {
                        "success": True,
                        "is_duplicate": True,
                        "filename": original_name,
                        "file_path": fpath,
                        "message": "内容已存在，自动去重"
                    }
            except Exception:
                continue

        # --------------------------
        # 2. 同名自动版本化
        # --------------------------
        target_name = original_name
        version = 1
        while os.path.exists(os.path.join(ns_path, target_name)):
            version += 1
            target_name = f"{name}_v{version}{ext}"

        final_path = os.path.join(ns_path, target_name)
        with open(final_path, "wb") as f:
            f.write(content)

        await upload_file.seek(0)  # 重置指针

        return {
            "success": True,
            "is_duplicate": False,
            "filename": original_name,
            "file_path": final_path,
            "message": f"已保存（版本v{version})" if version > 1 else "已保存"
        }

    except Exception as e:
        logger.error(f"保存失败 {upload_file.filename}: {str(e)}")
        return {
            "success": False,
            "filename": upload_file.filename,
            "error": str(e)
        }


# ------------------------------
# 主处理逻辑
# ------------------------------
async def process_uploaded_files(files: List[UploadFile], namespace: str) -> dict:
    memory_tool: MemoryTool = global_registry.get_tool("memory")

    # rag_tool: RAGTool = global_registry.get_tool("rag")
    # rag_tool._clear_knowledge_base(confirm=False, namespace=namespace) 

    # ======================
    # 基础校验
    # ======================
    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个文件")
    if not namespace.strip():
        raise HTTPException(status_code=400, detail="命名空间不能为空")

    # ======================
    # 并发保存（去重+版本）
    # ======================
    save_tasks = [
        save_file_with_version_and_deduplicate(f, namespace)
        for f in files
    ]
    save_results = await asyncio.gather(*save_tasks)

    # ======================
    # 分类
    # ======================
    saved_files = []
    save_errors = []
    duplicate_files = []

    for res in save_results:
        if not res["success"]:
            save_errors.append(res)
        elif res.get("is_duplicate"):
            duplicate_files.append(res)
        else:
            saved_files.append(res)

    # ======================
    # 记录失败记忆
    # ======================
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for err in save_errors:
        memory_tool.run({
            "action": "add",
            "user_id": namespace,
            "memory_type": "perceptual",
            "content": f"{now} 用户{namespace}上传文件 {err['filename']} 保存失败：{err['error']}",
            "file_path": ""
        })

    # ======================
    # 无文件可处理
    # ======================
    if not saved_files:
        return {
            "success": len(save_errors) == 0,
            "msg": f"去重 {len(duplicate_files)} 个 | 失败 {len(save_errors)} 个",
            "duplicate_count": len(duplicate_files),
            "results": [],
            "errors": save_errors,
            "duplicates": duplicate_files
        }

    # ======================
    # 并行入库
    # ======================
    file_paths = [f["file_path"] for f in saved_files]
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

    executor = AsyncToolExecutor(global_registry, max_workers=MAX_PARALLEL_FILES)
    execute_results = await executor.execute_tools_parallel(rag_tasks)

    # ======================
    # 结果处理
    # ======================
    success_list = []
    fail_list = []

    for idx, file_info in enumerate(saved_files):
        res = execute_results[idx]
        fn = file_info["filename"]
        fp = file_info["file_path"]

        if res["status"] == "success":
            success_list.append({"filename": fn, "file_path": fp})
            memory_tool.run({
                "action": "add",
                "user_id": namespace,
                "memory_type": "semantic",
                "content": f"{now} 用户{namespace}已上传文件：{fn}，已入库知识库",
                "file_path": fp
            })
        else:
            fail_list.append({"filename": fn, "error": res["result"]})
            memory_tool.run({
                "action": "add",
                "user_id": namespace,
                "memory_type": "perceptual",
                "content": f"{now} 用户{namespace}文件 {fn} 处理失败：{res['result']}",
                "file_path": fp
            })

    # ======================
    # 最终返回（包含去重信息）
    # ======================
    return {
        "success": len(fail_list) == 0,
        "msg": f"成功 {len(success_list)} | 失败 {len(fail_list)} | 去重 {len(duplicate_files)}",
        "data": {
            "total": len(files),
            "saved": len(saved_files),
            "success": len(success_list),
            "failed": len(fail_list),
            "duplicate": len(duplicate_files),
            "namespace": namespace
        },
        "results": success_list,
        "errors": fail_list,
        "duplicates": duplicate_files
    }


# ------------------------------
# 路由
# ------------------------------
@new_updateFile_router.post("/new_update_file")
async def new_update_file(
    request: Request,
    files: List[UploadFile] = File(..., description="支持 txt, md, pdf, docx, doc, json"),
    namespace: str = Form(..., description="命名空间")
) -> dict:
    return await process_uploaded_files(files, namespace)