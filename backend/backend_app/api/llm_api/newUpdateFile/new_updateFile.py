from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from typing import List, Dict
from datetime import datetime
import logging
import os
import asyncio
import hashlib
import json
import tempfile
import shutil

from helloAgents.tools.async_executor import AsyncToolExecutor
from helloAgents.utils.helpers import get_file_path_from_upload_async
from helloAgents.tools.builtin.rag_tool import RAGTool
from helloAgents.tools.builtin.memory_tool import MemoryTool
from helloAgents.tools.registry import global_registry

logger = logging.getLogger(__name__)
new_updateFile_router = APIRouter()

# ------------------------------
# ⚙️ 企业级配置
# ------------------------------
MANIFEST_FILE = ".file_manifest.json"
CHUNK_SIZE = 8192
MAX_PARALLEL_FILES = 4

# ------------------------------
# 🛠️ 核心工具函数
# ------------------------------

def load_manifest(ns_path: str) -> Dict[str, str]:
    manifest_path = os.path.join(ns_path, MANIFEST_FILE)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning(f"读取 manifest 失败: {manifest_path}")
            return {}
    return {}

def save_manifest(ns_path: str, manifest: Dict[str, str]):
    manifest_path = os.path.join(ns_path, MANIFEST_FILE)
    temp_path = manifest_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, manifest_path)

def calculate_file_hash_stream(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()

def calculate_upload_hash_stream(file_obj) -> str:
    sha256 = hashlib.sha256()
    file_obj.file.seek(0)
    while chunk := file_obj.file.read(CHUNK_SIZE):
        sha256.update(chunk)
    file_obj.file.seek(0)
    return sha256.hexdigest()

async def save_file_with_version_and_deduplicate(
    upload_file: UploadFile,
    user_id: str,
    save_dir: str = "./knowledge_base"
) -> dict:
    original_name = upload_file.filename.strip()
    name, ext = os.path.splitext(original_name)
    ns_path = os.path.join(save_dir, user_id)

    os.makedirs(ns_path, exist_ok=True)
    manifest = load_manifest(ns_path)

    try:
        new_file_hash = calculate_upload_hash_stream(upload_file)

        for existing_name, existing_hash in manifest.items():
            if existing_hash == new_file_hash:
                return {
                    "success": True,
                    "is_duplicate": True,
                    "filename": original_name,
                    "file_path": os.path.join(ns_path, existing_name),
                    "message": f"内容已存在（同名：{existing_name}），自动去重"
                }

        target_name = original_name
        version = 1
        while target_name in manifest:
            version += 1
            target_name = f"{name}_v{version}{ext}"
        
        final_path = os.path.join(ns_path, target_name)

        with tempfile.NamedTemporaryFile(dir=ns_path, delete=False) as tmp_file:
            shutil.copyfileobj(upload_file.file, tmp_file)
            temp_file_path = tmp_file.name
        
        os.replace(temp_file_path, final_path)

        manifest[target_name] = new_file_hash
        save_manifest(ns_path, manifest)

        return {
            "success": True,
            "is_duplicate": False,
            "filename": original_name,
            "file_path": final_path,
            "hash": new_file_hash,
            "message": f"已保存（版本 v{version})" if version > 1 else "已保存"
        }

    except Exception as e:
        logger.error(f"保存失败 {upload_file.filename}: {str(e)}")
        return {
            "success": False,
            "filename": upload_file.filename,
            "error": str(e)
        }

# ------------------------------
# 主处理逻辑（已补全记忆）
# ------------------------------
async def process_uploaded_files(files: List[UploadFile], user_id: str) -> dict:
    memory_tool: MemoryTool = global_registry.get_tool("memory")
    rag_tool: RAGTool = global_registry.get_tool("rag")

    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个文件")
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="命名空间不能为空")

    semaphore = asyncio.Semaphore(MAX_PARALLEL_FILES)
    
    async def save_with_limit(f):
        async with semaphore:
            return await save_file_with_version_and_deduplicate(f, user_id)

    save_tasks = [save_with_limit(f) for f in files]
    save_results = await asyncio.gather(*save_tasks)

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

    # ============================
    # ✅ 企业级精简版：只存 1 条总结记忆（包含所有文件路径+状态）
    # ============================
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(files)
    success_cnt = len(saved_files)
    dup_cnt = len(duplicate_files)
    fail_cnt = len(save_errors)

    # 构建清晰的文件清单
    success_files = "\n".join([f"- {f['filename']} | 路径：{f['file_path']}" for f in saved_files])
    dup_files_list = "\n".join([f"- {d['filename']}" for d in duplicate_files])
    err_files_list = "\n".join([f"- {e['filename']} | 原因：{e['error']}" for e in save_errors])


    # 全部记忆 → 丢进 AsyncToolExecutor 并行后台执行
    # ============================
    async def save_all_memories_in_background():
        try:
            tasks = []

            # --------------------------
            # 1)  episodic 上传总结记忆
            # --------------------------
            summary_content = f"""【文件上传总结】{now} | 用户：{user_id}
    总文件数：{total}
    ✅ 上传成功：{success_cnt} 个
    {success_files if success_files else '无'}

    ⚠️ 重复文件：{dup_cnt} 个
    {dup_files_list if dup_files_list else '无'}

    ❌ 上传失败：{fail_cnt} 个
    {err_files_list if err_files_list else '无'}
    """
            tasks.append({
                "tool_name": "memory",
                "input_data": {
                    "action": "add",
                    "user_id": user_id,
                    "memory_type": "episodic",
                    "content": summary_content.strip(),
                    "importance": 0.8,
                    "session_id": None
                }
            })

            # --------------------------
            # 2)  perceptual 多模态文件记忆
            # --------------------------
            for file in saved_files:
                tasks.append({
                    "tool_name": "memory",
                    "input_data": {
                        "action": "add",
                        "user_id": user_id,
                        "memory_type": "perceptual",
                        "content": f"【多模态文件】{file['filename']} | 路径：{file['file_path']}",
                        "file_path": file["file_path"],
                        "importance": 0.7,
                        "session_id": None
                    }
                })

            # --------------------------
            # 🔥 并行执行所有记忆保存（完全异步）
            # --------------------------
            executor = AsyncToolExecutor(global_registry, max_workers=5)
            await executor.execute_tools_parallel(tasks)

        except Exception as e:
            logger.warning(f"后台记忆保存任务失败: {e}")

    # --------------------------
    # 提交后台 → 主流程完全不等待
    # --------------------------
    asyncio.create_task(save_all_memories_in_background())

    # ============================
    # ✅ 记忆存储结束
    # ============================

    if not saved_files:
        return {
            "success": len(save_errors) == 0,
            "msg": f"去重 {len(duplicate_files)} 个 | 失败 {len(save_errors)} 个",
            "duplicate_count": len(duplicate_files),
            "results": [],
            "errors": save_errors,
            "duplicates": duplicate_files
        }

    # RAG 入库（后台执行，不暴露）
    file_paths = [f["file_path"] for f in saved_files]
    result = rag_tool.run({
        "action": "add_document",
        "file_path": file_paths,
        "user_id": user_id
    })

    return {
        "success": True,
        "msg": f"成功 {len(file_paths)} | 去重 {len(duplicate_files)}",
        "data": {
            "total": len(files),
            "saved": len(saved_files),
            "success": file_paths,
            "duplicate_files": duplicate_files,
            "namespace": user_id,
            "result": result
        }
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