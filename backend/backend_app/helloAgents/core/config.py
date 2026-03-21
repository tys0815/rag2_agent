"""企业级配置管理
基于pydantic-settings的配置管理系统，支持环境变量和配置文件
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ==================== 企业级配置类 ====================

class RAGSettings(BaseSettings):
    """RAG相关配置"""
    qdrant_urls: List[str] = ["http://localhost:6333"]
    qdrant_api_keys: Optional[List[str]] = None
    chunk_size: int = 800
    chunk_overlap: int = 100
    cache_ttl: int = 300
    cache_size: int = 1000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        case_sensitive=False
    )


class LLMSettings(BaseSettings):
    """LLM相关配置"""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_",
        case_sensitive=False
    )


# 全局应用配置
class AppSettings(BaseSettings):
    rag: RAGSettings = RAGSettings()
    llm: LLMSettings = LLMSettings()
    debug: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# ==================== 向后兼容的配置类 ====================

class Config(BaseModel):
    """HelloAgents配置类（向后兼容）"""

    # LLM配置
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 其他配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()


# ==================== 全局配置实例 ====================

# 尝试导入pydantic_settings，如果失败则回退到基本配置
try:
    from pydantic_settings import BaseSettings
    # 创建全局配置实例
    settings = AppSettings()

    # 更新Config类以支持从新配置系统获取值
    def _get_compatible_config() -> Config:
        """获取向后兼容的配置"""
        return Config(
            default_model=settings.llm.model_name,
            default_provider="openai",  # 保持默认
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            debug=settings.debug,
            log_level=settings.log_level,
            max_history_length=100
        )

    # 覆盖原有的from_env方法
    Config.from_env = classmethod(lambda cls: _get_compatible_config())

except ImportError:
    # 如果pydantic_settings不可用，使用基本配置
    print("⚠️  pydantic_settings未安装，使用基本配置管理")

    # 创建简单的settings对象
    class SimpleRAGSettings:
        qdrant_urls = ["http://localhost:6333"]
        qdrant_api_keys = None
        chunk_size = 800
        chunk_overlap = 100
        cache_ttl = 300
        cache_size = 1000

    class SimpleLLMSettings:
        model_name = "gpt-3.5-turbo"
        temperature = 0.7
        max_tokens = 2000
        timeout = 30

    class SimpleAppSettings:
        rag = SimpleRAGSettings()
        llm = SimpleLLMSettings()
        debug = False
        log_level = "INFO"

    settings = SimpleAppSettings()

    def _get_compatible_config() -> Config:
        """获取向后兼容的配置"""
        return Config(
            default_model=settings.llm.model_name,
            default_provider="openai",
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            debug=settings.debug,
            log_level=settings.log_level,
            max_history_length=100
        )


# ==================== 辅助函数 ====================

def get_config() -> Config:
    """获取配置实例（向后兼容）"""
    return _get_compatible_config()


def get_rag_config() -> RAGSettings:
    """获取RAG配置"""
    return settings.rag


def get_llm_config() -> LLMSettings:
    """获取LLM配置"""
    return settings.llm


def is_debug_mode() -> bool:
    """检查是否为调试模式"""
    return settings.debug


def get_log_level() -> str:
    """获取日志级别"""
    return settings.log_level
