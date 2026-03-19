"""媒体生成配置文件
语音和视频生成服务的配置参数
"""

import os
from typing import Dict, Any

# ==================== 基础配置 ====================

class MediaConfig:
    """媒体生成配置类"""

    # 输出目录配置
    OUTPUT_BASE_DIR = "./media_outputs"
    VOICE_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "voices")
    VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "videos")
    THUMBNAIL_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "thumbnails")
    TEMP_DIR = os.path.join(OUTPUT_BASE_DIR, "temp")

    # 语音生成配置
    VOICE_CONFIG = {
        # 默认参数
        "default_voice_type": "female",
        "default_speed": "normal",
        "default_language": "zh-CN",
        "default_format": "mp3",

        # 支持的语言
        "supported_languages": ["zh-CN", "en-US", "ja-JP", "ko-KR", "fr-FR"],

        # 支持的格式
        "supported_formats": ["mp3", "wav", "ogg", "flac"],

        # 文件大小限制（字节）
        "max_text_length": 5000,
        "max_file_size": 10 * 1024 * 1024,  # 10MB

        # 语音引擎配置
        "engine": "gtts",  # gtts, pyttsx3, edge-tts, azure, aliyun

        # 缓存配置
        "cache_enabled": True,
        "cache_ttl": 3600,  # 缓存时间（秒）
    }

    # 视频生成配置
    VIDEO_CONFIG = {
        # 默认参数
        "default_style": "animation",
        "default_duration": 5,  # 秒
        "default_resolution": "720p",
        "default_aspect_ratio": "16:9",
        "default_frame_rate": 30,

        # 支持的分辨率
        "supported_resolutions": {
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "2k": (2560, 1440),
            "4k": (3840, 2160)
        },

        # 支持的风格
        "supported_styles": [
            "animation", "realistic", "3d", "cartoon",
            "cinematic", "watercolor", "sketch", "pixel"
        ],

        # 文件大小限制
        "max_duration": 60,  # 最大60秒
        "max_file_size": 100 * 1024 * 1024,  # 100MB

        # 视频引擎配置
        "engine": "simulated",  # simulated, moviepy, svd, runwayml

        # 异步处理配置
        "async_processing": True,
        "max_workers": 4,
        "queue_timeout": 300,  # 队列超时时间（秒）
    }

    # 存储配置
    STORAGE_CONFIG = {
        "type": "local",  # local, s3, azure_blob, gcs
        "local": {
            "base_path": OUTPUT_BASE_DIR,
            "cleanup_days": 7,  # 自动清理天数
        },
        "cloud": {
            "bucket_name": "",
            "region": "",
            "access_key": "",
            "secret_key": ""
        }
    }

    # 服务质量配置
    QUALITY_CONFIG = {
        # 语音质量
        "voice_quality": {
            "low": {"bitrate": "32k", "sample_rate": 16000},
            "medium": {"bitrate": "64k", "sample_rate": 22050},
            "high": {"bitrate": "128k", "sample_rate": 44100}
        },

        # 视频质量
        "video_quality": {
            "low": {"bitrate": "500k", "crf": 28},
            "medium": {"bitrate": "1500k", "crf": 23},
            "high": {"bitrate": "5000k", "crf": 18}
        }
    }

    # API配置
    API_CONFIG = {
        "rate_limit_per_minute": 60,
        "max_concurrent_requests": 10,
        "request_timeout": 300,  # 请求超时时间（秒）

        # 安全配置
        "auth_required": False,
        "api_key_header": "X-API-Key",

        # CORS配置
        "cors_origins": ["*"],
        "cors_methods": ["*"],
        "cors_headers": ["*"],
    }

    # 监控和日志配置
    MONITORING_CONFIG = {
        "enable_metrics": True,
        "enable_logging": True,
        "log_level": "INFO",

        # 性能监控
        "monitor_latency": True,
        "monitor_errors": True,
        "monitor_usage": True,

        # 告警配置
        "alert_thresholds": {
            "error_rate": 0.05,  # 5%
            "latency_p95": 5000,  # 5秒
            "queue_size": 100
        }
    }

    @classmethod
    def get_voice_config(cls) -> Dict[str, Any]:
        """获取语音生成配置"""
        return cls.VOICE_CONFIG

    @classmethod
    def get_video_config(cls) -> Dict[str, Any]:
        """获取视频生成配置"""
        return cls.VIDEO_CONFIG

    @classmethod
    def get_storage_config(cls) -> Dict[str, Any]:
        """获取存储配置"""
        return cls.STORAGE_CONFIG

    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """获取API配置"""
        return cls.API_CONFIG

    @classmethod
    def setup_directories(cls):
        """创建必要的目录结构"""
        directories = [
            cls.OUTPUT_BASE_DIR,
            cls.VOICE_OUTPUT_DIR,
            cls.VIDEO_OUTPUT_DIR,
            cls.THUMBNAIL_OUTPUT_DIR,
            cls.TEMP_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ 目录已创建/确认: {directory}")


# ==================== 环境变量配置 ====================

def load_config_from_env() -> Dict[str, Any]:
    """从环境变量加载配置"""
    config = {
        "voice": {
            "engine": os.getenv("VOICE_ENGINE", "gtts"),
            "api_key": os.getenv("VOICE_API_KEY", ""),
            "region": os.getenv("VOICE_REGION", ""),
            "cache_enabled": os.getenv("VOICE_CACHE_ENABLED", "true").lower() == "true"
        },
        "video": {
            "engine": os.getenv("VIDEO_ENGINE", "simulated"),
            "api_key": os.getenv("VIDEO_API_KEY", ""),
            "model": os.getenv("VIDEO_MODEL", "svd"),
            "async_processing": os.getenv("VIDEO_ASYNC_PROCESSING", "true").lower() == "true"
        },
        "storage": {
            "type": os.getenv("STORAGE_TYPE", "local"),
            "bucket": os.getenv("STORAGE_BUCKET", ""),
            "region": os.getenv("STORAGE_REGION", ""),
            "access_key": os.getenv("STORAGE_ACCESS_KEY", ""),
            "secret_key": os.getenv("STORAGE_SECRET_KEY", "")
        },
        "api": {
            "rate_limit": int(os.getenv("API_RATE_LIMIT", "60")),
            "auth_required": os.getenv("API_AUTH_REQUIRED", "false").lower() == "true",
            "api_key": os.getenv("API_KEY", "")
        }
    }
    return config


# ==================== 配置验证 ====================

def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置是否有效"""
    try:
        # 验证语音配置
        voice_config = config.get("voice", {})
        if voice_config.get("engine") not in ["gtts", "pyttsx3", "edge-tts", "simulated"]:
            print(f"⚠️ 警告: 不支持的语音引擎: {voice_config.get('engine')}")

        # 验证视频配置
        video_config = config.get("video", {})
        if video_config.get("engine") not in ["simulated", "moviepy", "svd", "runwayml"]:
            print(f"⚠️ 警告: 不支持的视频引擎: {video_config.get('engine')}")

        # 验证存储配置
        storage_config = config.get("storage", {})
        storage_type = storage_config.get("type", "local")
        if storage_type not in ["local", "s3", "azure_blob", "gcs"]:
            print(f"⚠️ 警告: 不支持的存储类型: {storage_type}")

        return True

    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False


# ==================== 配置实例 ====================

# 创建默认配置实例
default_config = MediaConfig()

# 从环境变量加载配置
env_config = load_config_from_env()

# 验证配置
if validate_config(env_config):
    print("✅ 配置验证通过")
else:
    print("⚠️ 配置验证失败，使用默认配置")