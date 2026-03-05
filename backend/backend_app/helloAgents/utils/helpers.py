"""辅助工具函数"""

import importlib
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import UploadFile, HTTPException
import os
import uuid

def format_time(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化时间
    
    Args:
        timestamp: 时间戳，默认为当前时间
        format_str: 格式字符串
        
    Returns:
        格式化后的时间字符串
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(format_str)

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    验证配置是否包含必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        是否验证通过
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"配置缺少必需的键: {missing_keys}")
    return True

def safe_import(module_name: str, class_name: Optional[str] = None) -> Any:
    """
    安全导入模块或类
    
    Args:
        module_name: 模块名
        class_name: 类名（可选）
        
    Returns:
        导入的模块或类
    """
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        raise ImportError(f"无法导入 {module_name}.{class_name or ''}: {e}")

def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """深度合并两个字典"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


async def get_file_path_from_upload_async(
    file: UploadFile, 
    namespace: str,  # 新增的 namespace 参数
    save_dir: str = "./knowledge_base"
) -> str:
    """
    修复版：异步接收文件，同步保存（解决异步上下文管理器错误）
    新增 namespace 参数，先创建用户目录，再保存文件到该目录
    """
    # 基础校验
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="文件对象或文件名不能为空")
    
    if not namespace:  # 新增 namespace 校验
        raise HTTPException(status_code=400, detail="命名空间(namespace)不能为空")
    
    # 构建完整的保存路径：基础目录 / namespace 目录
    save_dir_path = Path(save_dir) / namespace
    # 确保目录存在（parents=True 会创建多级目录）
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 生成唯一文件名（避免同名文件覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{timestamp}_{unique_id}_{Path(file.filename).name}"
    save_path = save_dir_path / safe_filename
    absolute_file_path = str(save_path.absolute())
    
    try:
        # 异步读取文件内容（UploadFile 的 read 是异步的，保留）
        file_content = await file.read()
        
        # 同步写入文件到 namespace 专属目录
        with open(save_path, "wb") as f:
            f.write(file_content)
        
        # 验证文件是否保存成功
        if not os.path.exists(absolute_file_path):
            raise HTTPException(status_code=500, detail="文件保存后验证失败：路径不存在")
        
        return absolute_file_path
    
    except Exception as e:
        # 清理临时文件（如果保存失败）
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"文件保存失败：{str(e)}")
    