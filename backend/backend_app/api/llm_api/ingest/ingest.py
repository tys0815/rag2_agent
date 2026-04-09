import json
from typing import Dict
from fastapi import APIRouter, Form, HTTPException

from datetime import datetime
import logging
import os
from pydantic import BaseModel
from helloAgents.tools.registry import global_registry
from helloAgents.tools.builtin.rag_tool import RAGTool
# 2. 定义请求体
class FileListRequest(BaseModel):
    user_id: str

logger = logging.getLogger(__name__)
ingest_router = APIRouter()

# ------------------------------
# ⚙️ 企业级配置（和你原有一致）
# ------------------------------
MANIFEST_FILE = ".file_manifest.json"
CHUNK_SIZE = 8192
MAX_PARALLEL_FILES = 4
SAVE_BASE_DIR = "./knowledge_base"  # 统一基座目录

# ------------------------------
# 🆕 新增：获取用户上传文档接口
# ------------------------------
@ingest_router.post("/list")
async def get_user_uploaded_files(body: FileListRequest):
    """
    获取指定命名空间下所有已上传的文件列表
    返回：所有真实文件、文件路径、哈希、去重状态、文件信息
    """
    user_id = body.user_id
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="用户ID不能为空")

    # 用户文件目录
    user_dir = os.path.join(SAVE_BASE_DIR, user_id)
    if not os.path.exists(user_dir):
        return {
            "success": True,
            "user_id": user_id,
            "total_files": 0,
            "files": [],
            "msg": "该用户暂无上传文件"
        }

    # 加载 manifest 清单（你原有去重/版本管理核心）
    manifest = load_manifest(user_dir)
    if not manifest:
        return {
            "success": True,
            "user_id": user_id,
            "total_files": 0,
            "files": [],
            "msg": "该用户暂无上传文件（清单为空）"
        }

    # 构建文件列表
    file_list = []
    for filename, file_hash in manifest.items():
        file_path = os.path.join(user_dir, filename)
        
        # 获取文件基本信息
        file_stat = os.stat(file_path)
        create_time = datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        update_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        file_size = file_stat.st_size  # 字节

        # 文件格式
        _, ext = os.path.splitext(filename)
        ext = ext.lower().lstrip(".")

        file_list.append({
            "filename": filename,
            "original_filename": filename.split("_v")[0] + "." + ext if "_v" in filename else filename,
            "file_path": file_path,
            "file_hash": file_hash,
            "file_type": ext,
            "file_size_bytes": file_size,
            "file_size": f"{round(file_size / 1024 / 1024, 2)} MB" if file_size > 1024*1024 else f"{round(file_size / 1024, 2)} KB",
            "create_time": create_time,
            "update_time": update_time,
            "is_versioned": "_v" in filename  # 是否是版本文件
        })

    # 按上传时间倒序排列（最新在前）
    file_list = sorted(file_list, key=lambda x: x["create_time"], reverse=True)

    return {
        "success": True,
        "user_id": user_id,
        "total_files": len(file_list),
        "files": file_list,
        "msg": f"成功获取 {len(file_list)} 个文件"
    }



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




class DeleteFileListRequest(BaseModel):
    user_id: str
    doc_id: str
# ------------------------------
# 🗑️ 新增：删除文档接口（通过 file_hash 删除）
# ------------------------------
@ingest_router.post("/delete")
async def delete_file_by_hash(
    body: DeleteFileListRequest
):
    # 接收前端传来的 user_id 和 doc_id
    user_id = body.user_id
    doc_id = body.doc_id    
    if not user_id.strip() or not doc_id.strip():
        raise HTTPException(status_code=400, detail="user_id 和 doc_id 不能为空")

    user_dir = os.path.join(SAVE_BASE_DIR, user_id)
    manifest = load_manifest(user_dir)

    # 1. 通过 doc_id 找到对应的文件名
    target_filename = None
    for filename, f_doc_id in manifest.items():
        if f_doc_id == doc_id:
            target_filename = filename
            break

    if not target_filename:
        raise HTTPException(status_code=404, detail="文件不存在或已删除")

    # 2. 删除真实文件
    file_path = os.path.join(user_dir, target_filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # 3. 从 manifest 中移除
    del manifest[target_filename]
    save_manifest(user_dir, manifest)

    # 删除向量库
    rag_tool: RAGTool = global_registry.get_tool("rag")
    result = rag_tool.run({
        "action": "delete_document",
        "user_id": user_id,
        "doc_id": doc_id
    })

    if result:
        return {
        "success": True,
        "msg": "删除成功",
        "filename": target_filename,
        "doc_id": doc_id
    }
    else:
        return {
            "success": False,
            "msg": "删除失败，请手动检查",
            "filename": target_filename,
            "doc_id": doc_id
        }

    


def save_manifest(ns_path: str, manifest: Dict[str, str]):
    manifest_path = os.path.join(ns_path, MANIFEST_FILE)
    temp_path = manifest_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, manifest_path)