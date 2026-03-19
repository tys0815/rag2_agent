# 语音生成API文档

## 概述
语音生成API提供企业级文本到语音转换服务，支持多种语音类型、语速、音调、音量等参数控制，可生成MP3、WAV、OGG等多种格式的音频文件。

## API端点

### 1. 生成语音
**POST** `/api/v1/voice/generate`

**请求体：**
```json
{
  "text": "欢迎使用企业级语音生成服务",
  "voice_type": "female",
  "speed": "normal",
  "pitch": 1.0,
  "volume": 1.0,
  "output_format": "mp3",
  "language": "zh-CN",
  "user_id": "user_12345"
}
```

**参数说明：**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | string | 是 | - | 要转换的文本内容，最长5000字符 |
| voice_type | string | 否 | female | 语音类型：female(甜美女生)、male(阳光男生)、child(卡通童声)、professional(专业播音) |
| speed | string | 否 | normal | 语速：slow(慢速)、normal(正常)、fast(快速) |
| pitch | float | 否 | 1.0 | 音调，范围0.5-2.0 |
| volume | float | 否 | 1.0 | 音量，范围0.0-2.0 |
| output_format | string | 否 | mp3 | 输出格式：mp3、wav、ogg |
| language | string | 否 | zh-CN | 语言代码：zh-CN、en-US、ja-JP |
| user_id | string | 否 | null | 用户ID，用于统计和配额管理 |

**响应：**
```json
{
  "success": true,
  "voice_url": "/api/voice/files/voice_abc123.mp3",
  "voice_id": "voice_abc123",
  "duration": 12.5,
  "file_size": 256000,
  "format": "mp3",
  "text_length": 45,
  "timestamp": "2024-01-01T12:00:00"
}
```

### 2. 获取语音文件
**GET** `/api/v1/voice/files/{voice_id}.{format}`

**参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| voice_id | string | 是 | 语音文件ID |
| format | string | 是 | 文件格式：mp3、wav、ogg |

**响应：**
返回音频文件流，Content-Type为对应的音频类型。

### 3. 获取支持的语音类型
**GET** `/api/v1/voice/supported-voices`

**响应：**
```json
{
  "success": true,
  "voices": {
    "female": {"name": "甜美女生", "description": "清晰甜美的女声"},
    "male": {"name": "阳光男生", "description": "阳光活力的男声"},
    "child": {"name": "卡通童声", "description": "可爱活泼的童声"},
    "professional": {"name": "专业播音", "description": "专业播音员声音"}
  },
  "formats": ["mp3", "wav", "ogg"],
  "languages": ["zh-CN", "en-US", "ja-JP"],
  "timestamp": "2024-01-01T12:00:00"
}
```

### 4. 列出语音文件
**GET** `/api/v1/voice/list?page=1&page_size=20`

**查询参数：**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| page | int | 否 | 1 | 页码 |
| page_size | int | 否 | 20 | 每页数量 |

**响应：**
```json
{
  "success": true,
  "voices": [
    {
      "voice_id": "voice_abc123",
      "filename": "voice_abc123.mp3",
      "size": 256000,
      "created_at": "2024-01-01T12:00:00",
      "format": "mp3"
    }
  ],
  "total": 5,
  "page": 1,
  "page_size": 20,
  "timestamp": "2024-01-01T12:00:00"
}
```

### 5. 删除语音文件
**POST** `/api/v1/voice/delete`

**请求体：**
```json
{
  "voice_id": "voice_abc123",
  "confirm": true
}
```

**响应：**
```json
{
  "success": true,
  "message": "语音文件已成功删除",
  "timestamp": "2024-01-01T12:00:00"
}
```

### 6. 健康检查
**GET** `/api/v1/voice/health`

**响应：**
```json
{
  "status": "healthy",
  "service": "voice_generation",
  "output_dir": "./voice_outputs",
  "files_count": 10,
  "timestamp": "2024-01-01T12:00:00"
}
```

## 错误处理

所有API端点都遵循统一的错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```

**HTTP状态码：**
- 200: 请求成功
- 400: 参数错误
- 404: 资源不存在
- 500: 服务器内部错误

## 企业级功能

### 1. 多语言支持
支持中文、英文、日文等多种语言，可根据需要扩展。

### 2. 参数验证
对所有输入参数进行严格验证，确保数据安全性和服务稳定性。

### 3. 错误处理与降级
完善的错误处理机制，在主要TTS服务不可用时支持降级到备用服务。

### 4. 性能监控
内置性能监控和日志记录，支持实时监控服务状态。

### 5. 缓存机制
支持语音文件缓存，减少重复生成，提高响应速度。

## 部署配置

### 环境变量
```
# 语音引擎配置
VOICE_ENGINE=gtts  # gtts, pyttsx3, edge-tts, simulated
VOICE_API_KEY=your_api_key
VOICE_REGION=us-west-1

# 存储配置
STORAGE_TYPE=local  # local, s3, azure_blob, gcs
STORAGE_BUCKET=voice-bucket
STORAGE_REGION=us-west-1

# API配置
API_RATE_LIMIT=60
API_AUTH_REQUIRED=false
```

### 依赖安装
```bash
# 基本依赖
pip install fastapi uvicorn python-multipart

# 语音生成依赖（根据选择的引擎）
pip install gTTS        # Google Text-to-Speech
# 或
pip install pyttsx3     # 离线TTS
# 或
pip install edge-tts    # Microsoft Edge TTS
```

## 使用示例

### Python客户端
```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

def generate_voice(text, voice_type="female"):
    response = requests.post(f"{BASE_URL}/voice/generate", json={
        "text": text,
        "voice_type": voice_type,
        "output_format": "mp3",
        "language": "zh-CN"
    })

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            print(f"语音生成成功: {data['voice_url']}")
            return data
    else:
        print(f"语音生成失败: {response.text}")

    return None
```

### JavaScript/TypeScript客户端
```typescript
const API_BASE_URL = 'http://localhost:8000/api/v1';

async function generateVoice(text: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/voice/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      voice_type: 'female',
      output_format: 'mp3'
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}
```

## 性能优化建议

1. **启用缓存**：对于重复的文本内容，启用缓存可大幅提升性能
2. **批量处理**：支持批量文本转换，减少API调用次数
3. **异步处理**：对于长文本，支持异步生成和状态查询
4. **CDN加速**：生成的语音文件可通过CDN分发，提高访问速度
5. **负载均衡**：在高并发场景下，建议部署多个实例并使用负载均衡

## 安全注意事项

1. **输入验证**：所有用户输入都应进行严格的验证和清理
2. **配额限制**：根据用户等级实施API调用配额限制
3. **访问控制**：敏感接口应实施身份验证和授权
4. **日志审计**：记录所有API调用，便于安全审计
5. **数据加密**：敏感数据在传输和存储时应加密