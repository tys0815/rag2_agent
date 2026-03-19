# 视频生成API文档

## 概述
视频生成API提供企业级文本到视频生成服务，支持多种视频风格、分辨率、时长等参数控制，可生成MP4、AVI、MOV等多种格式的视频文件，支持背景音乐和语音解说。

## API端点

### 1. 生成视频
**POST** `/api/v1/video/generate`

**请求体：**
```json
{
  "prompt": "一只小鸟从天空飞过，落在树枝上，背景是森林",
  "style": "animation",
  "duration": 5,
  "resolution": "720p",
  "aspect_ratio": "16:9",
  "frame_rate": 30,
  "background_music": false,
  "voice_over": true,
  "voice_over_text": "这是一只美丽的小鸟，它在森林中自由飞翔",
  "user_id": "user_12345"
}
```

**参数说明：**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| prompt | string | 是 | - | 视频描述文本，最长2000字符 |
| style | string | 否 | animation | 视频风格：animation(动画)、realistic(真人实拍)、3d(3D建模)、cartoon(卡通)、cinematic(电影感) |
| duration | int | 否 | 5 | 视频时长（秒），范围1-60 |
| resolution | string | 否 | 720p | 分辨率：480p、720p、1080p、4k |
| aspect_ratio | string | 否 | 16:9 | 宽高比：16:9、4:3、1:1、9:16 |
| frame_rate | int | 否 | 30 | 帧率，范围1-60 |
| background_music | bool | 否 | false | 是否添加背景音乐 |
| voice_over | bool | 否 | false | 是否添加语音解说 |
| voice_over_text | string | 否 | null | 语音解说文本 |
| user_id | string | 否 | null | 用户ID，用于统计和配额管理 |

**响应：**
```json
{
  "success": true,
  "video_url": "/api/video/files/video_abc123.mp4",
  "video_id": "video_abc123",
  "duration": 5,
  "file_size": 1024000,
  "resolution": "720p",
  "format": "mp4",
  "thumbnail_url": "/api/video/files/thumbnails/thumb_abc123.jpg",
  "timestamp": "2024-01-01T12:00:00",
  "estimated_generation_time": 30
}
```

### 2. 获取视频状态
**POST** `/api/v1/video/status`

**请求体：**
```json
{
  "video_id": "video_abc123"
}
```

**响应：**
```json
{
  "success": true,
  "video_id": "video_abc123",
  "status": "processing",
  "progress": 65.5,
  "estimated_completion_time": "2024-01-01T12:01:30",
  "message": "正在生成视频帧...",
  "timestamp": "2024-01-01T12:00:30"
}
```

**状态说明：**
- `pending`: 等待处理
- `processing`: 处理中
- `completed`: 已完成
- `failed`: 失败

### 3. 获取视频文件
**GET** `/api/v1/video/files/{video_id}.{format}`

**参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| video_id | string | 是 | 视频文件ID |
| format | string | 是 | 文件格式：mp4、avi、mov、webm |

**响应：**
返回视频文件流，Content-Type为对应的视频类型。

### 4. 获取视频缩略图
**GET** `/api/v1/video/files/thumbnails/{thumbnail_filename}`

**参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| thumbnail_filename | string | 是 | 缩略图文件名，格式：thumb_{video_id}.jpg |

**响应：**
返回缩略图文件流，Content-Type为image/jpeg。

### 5. 获取支持的视频风格
**GET** `/api/v1/video/supported-styles`

**响应：**
```json
{
  "success": true,
  "styles": {
    "animation": {"name": "动画", "description": "2D/3D动画风格"},
    "realistic": {"name": "真人实拍", "description": "实拍视频效果"},
    "3d": {"name": "3D建模", "description": "三维建模渲染"},
    "cartoon": {"name": "卡通", "description": "卡通漫画风格"},
    "cinematic": {"name": "电影感", "description": "电影级视觉效果"}
  },
  "resolutions": ["480p", "720p", "1080p", "4k"],
  "formats": ["mp4", "avi", "mov", "webm"],
  "timestamp": "2024-01-01T12:00:00"
}
```

### 6. 列出视频文件
**GET** `/api/v1/video/list?page=1&page_size=20`

**查询参数：**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| page | int | 否 | 1 | 页码 |
| page_size | int | 否 | 20 | 每页数量 |

**响应：**
```json
{
  "success": true,
  "videos": [
    {
      "video_id": "video_abc123",
      "filename": "video_abc123.mp4",
      "size": 1024000,
      "created_at": "2024-01-01T12:00:00",
      "format": "mp4",
      "has_thumbnail": true
    }
  ],
  "total": 8,
  "page": 1,
  "page_size": 20,
  "timestamp": "2024-01-01T12:00:00"
}
```

### 7. 删除视频文件
**POST** `/api/v1/video/delete`

**请求体：**
```json
{
  "video_id": "video_abc123",
  "confirm": true
}
```

**响应：**
```json
{
  "success": true,
  "message": "视频文件已成功删除",
  "timestamp": "2024-01-01T12:00:00"
}
```

### 8. 健康检查
**GET** `/api/v1/video/health`

**响应：**
```json
{
  "status": "healthy",
  "service": "video_generation",
  "output_dir": "./video_outputs",
  "video_files_count": 15,
  "thumbnail_files_count": 15,
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

### 1. 异步处理
支持长视频生成的异步处理，可通过状态查询接口实时获取生成进度。

### 2. 多格式支持
支持多种视频格式和分辨率，满足不同场景需求。

### 3. 缩略图生成
自动为每个视频生成缩略图，便于预览和展示。

### 4. 语音解说集成
支持为视频添加语音解说，实现文本到语音再到视频的完整流程。

### 5. 进度跟踪
实时进度跟踪和状态更新，提供良好的用户体验。

## 部署配置

### 环境变量
```
# 视频引擎配置
VIDEO_ENGINE=simulated  # simulated, moviepy, svd, runwayml
VIDEO_API_KEY=your_api_key
VIDEO_MODEL=svd

# 存储配置
STORAGE_TYPE=local  # local, s3, azure_blob, gcs
STORAGE_BUCKET=video-bucket
STORAGE_REGION=us-west-1

# 异步处理配置
VIDEO_ASYNC_PROCESSING=true
MAX_WORKERS=4
QUEUE_TIMEOUT=300

# API配置
API_RATE_LIMIT=30  # 视频生成较耗资源，限制更低
API_AUTH_REQUIRED=false
```

### 依赖安装
```bash
# 基本依赖
pip install fastapi uvicorn python-multipart

# 视频处理依赖（根据选择的引擎）
pip install moviepy          # 视频编辑和处理
pip install opencv-python    # 计算机视觉库
pip install pillow           # 图像处理

# AI模型相关（用于高级视频生成）
pip install diffusers        # 扩散模型（用于视频生成）
pip install transformers     # 已包含
pip install torch            # 已包含
```

## 使用示例

### Python客户端
```python
import requests
import time

BASE_URL = "http://localhost:8000/api/v1"

def generate_video(prompt, style="animation"):
    # 提交生成请求
    response = requests.post(f"{BASE_URL}/video/generate", json={
        "prompt": prompt,
        "style": style,
        "duration": 5,
        "resolution": "720p"
    })

    if response.status_code != 200:
        print(f"提交失败: {response.text}")
        return None

    data = response.json()
    if not data["success"]:
        print(f"生成失败: {data.get('message')}")
        return None

    video_id = data["video_id"]
    print(f"视频生成已提交，ID: {video_id}")

    # 轮询状态
    while True:
        status_response = requests.post(f"{BASE_URL}/video/status", json={
            "video_id": video_id
        })

        if status_response.status_code == 200:
            status_data = status_response.json()
            if status_data["success"]:
                status = status_data["status"]
                progress = status_data["progress"]

                print(f"进度: {progress}% - 状态: {status}")

                if status == "completed":
                    print(f"视频生成完成: {data['video_url']}")
                    return data
                elif status == "failed":
                    print("视频生成失败")
                    return None

        time.sleep(2)
```

### JavaScript/TypeScript客户端
```typescript
const API_BASE_URL = 'http://localhost:8000/api/v1';

class VideoGenerator {
  async generateVideo(prompt: string): Promise<any> {
    // 提交生成请求
    const response = await fetch(`${API_BASE_URL}/video/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt,
        style: 'animation',
        duration: 5,
        resolution: '720p'
      })
    });

    if (!response.ok) {
      throw new Error(`提交失败: ${response.status}`);
    }

    const data = await response.json();
    if (!data.success) {
      throw new Error(`生成失败: ${data.message}`);
    }

    return data;
  }

  async checkStatus(videoId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video/status`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ video_id: videoId })
    });

    if (!response.ok) {
      throw new Error(`状态查询失败: ${response.status}`);
    }

    return await response.json();
  }

  async downloadVideo(videoId: string, format: string = 'mp4'): Promise<void> {
    const url = `${API_BASE_URL}/video/files/${videoId}.${format}`;
    window.open(url, '_blank');
  }
}
```

## 性能优化建议

1. **异步队列**：使用消息队列处理视频生成任务，避免阻塞主线程
2. **GPU加速**：视频生成任务尽量使用GPU加速
3. **分级存储**：热数据使用SSD，冷数据迁移到低成本存储
4. **CDN分发**：生成的视频文件通过CDN分发，提高访问速度
5. **预览优化**：先生成低分辨率预览，再生成全分辨率版本

## 安全注意事项

1. **内容审核**：对用户输入的描述文本进行内容安全审核
2. **资源限制**：实施严格的资源使用限制，防止滥用
3. **访问控制**：敏感操作需要身份验证和授权
4. **数据隔离**：不同用户的数据进行逻辑或物理隔离
5. **监控告警**：实时监控系统资源使用情况，设置告警阈值

## 扩展功能

### 1. 批量生成
支持批量视频生成，提高处理效率。

### 2. 自定义模板
支持用户上传自定义视频模板。

### 3. 高级编辑
支持对生成的视频进行进一步编辑（裁剪、滤镜、字幕等）。

### 4. 多语言支持
支持更多语言的语音解说。

### 5. 实时预览
支持生成过程中的实时预览功能。

## 故障排除

### 常见问题
1. **生成速度慢**：检查GPU资源是否充足，考虑升级硬件或优化算法
2. **内存不足**：调整视频分辨率或时长，减少内存占用
3. **文件太大**：调整视频编码参数，优化文件大小
4. **质量不佳**：优化描述文本，调整生成参数
5. **服务不可用**：检查依赖服务状态，查看错误日志

### 日志查看
```bash
# 查看应用日志
tail -f logs/application.log

# 查看错误日志
tail -f logs/error.log

# 查看访问日志
tail -f logs/access.log
```