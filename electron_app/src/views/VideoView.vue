<template>
  <div class="p-6 h-full flex flex-col">
    <div class="max-w-2xl mx-auto w-full">
      <h2 class="text-xl font-bold mb-6 flex items-center gap-2">
        <i class="fa fa-film text-blue-500"></i>
        <span>视频生成</span>
      </h2>

      <!-- 统计信息卡片 -->
      <div class="grid grid-cols-4 gap-4 mb-6">
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-blue-600">{{ generatedCount }}</div>
          <div class="text-sm text-blue-800 mt-1">生成次数</div>
        </div>
        <div class="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-green-600">{{ estimatedTime }}</div>
          <div class="text-sm text-green-800 mt-1">预估时间(秒)</div>
        </div>
        <div class="bg-purple-50 border border-purple-200 rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-purple-600">{{ activeJobs }}</div>
          <div class="text-sm text-purple-800 mt-1">进行中任务</div>
        </div>
        <div class="bg-amber-50 border border-amber-200 rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-amber-600">{{ videoHistory.length }}</div>
          <div class="text-sm text-amber-800 mt-1">历史记录</div>
        </div>
      </div>

      <div class="border border-gray-200 rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center mb-2">
          <label class="block text-sm font-medium">视频描述</label>
          <span class="text-xs text-gray-500">{{ promptLength }}/2000 字符</span>
        </div>
        <textarea
          class="w-full border border-gray-200 rounded-lg p-3 min-h-[120px] focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 resize-none text-sm"
          v-model="prompt"
          placeholder="请输入视频的描述内容（如：一只小鸟从天空飞过，落在树枝上，背景是森林，阳光透过树叶洒下光斑）..."
          maxlength="2000"
          @input="updatePromptStats"
        ></textarea>
        <div class="text-xs text-gray-500 mt-2">
          提示：描述越详细，生成效果越好。可以包含场景、主体、动作、风格、氛围等要素。
        </div>
      </div>

      <div class="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label class="block text-sm font-medium mb-2">视频风格</label>
          <select
            class="w-full border border-gray-200 rounded-lg p-3 focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 text-sm"
            v-model="style"
          >
            <option value="animation">动画风格</option>
            <option value="realistic">真人实拍</option>
            <option value="3d">3D建模</option>
            <option value="cartoon">卡通漫画</option>
            <option value="cinematic">电影感</option>
            <option value="watercolor">水彩画风</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-2">视频时长</label>
          <select
            class="w-full border border-gray-200 rounded-lg p-3 focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 text-sm"
            v-model="duration"
          >
            <option value="3">3秒</option>
            <option value="5">5秒</option>
            <option value="10">10秒</option>
            <option value="15">15秒</option>
            <option value="30">30秒</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-2">分辨率</label>
          <select
            class="w-full border border-gray-200 rounded-lg p-3 focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 text-sm"
            v-model="resolution"
          >
            <option value="480p">480P (SD)</option>
            <option value="720p">720P (HD)</option>
            <option value="1080p">1080P (Full HD)</option>
            <option value="4k">4K (Ultra HD)</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-2">宽高比</label>
          <select
            class="w-full border border-gray-200 rounded-lg p-3 focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 text-sm"
            v-model="aspectRatio"
          >
            <option value="16:9">16:9 (宽屏)</option>
            <option value="4:3">4:3 (标准)</option>
            <option value="1:1">1:1 (方形)</option>
            <option value="9:16">9:16 (竖屏)</option>
          </select>
        </div>
      </div>

      <!-- 高级选项 -->
      <div class="border border-gray-200 rounded-lg p-4 mb-6">
        <div class="flex items-center mb-3">
          <i class="fa fa-sliders text-gray-500 mr-2"></i>
          <h3 class="text-sm font-medium">高级选项</h3>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium mb-2">帧率: {{ frameRate }} FPS</label>
            <input
              type="range"
              min="1"
              max="60"
              step="1"
              v-model="frameRate"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div class="flex justify-between text-xs text-gray-500 mt-1">
              <span>1 FPS</span>
              <span>30 FPS</span>
              <span>60 FPS</span>
            </div>
          </div>
          <div>
            <div class="flex items-center space-x-4">
              <label class="flex items-center">
                <input type="checkbox" v-model="backgroundMusic" class="mr-2">
                <span class="text-sm">背景音乐</span>
              </label>
              <label class="flex items-center">
                <input type="checkbox" v-model="voiceOver" class="mr-2">
                <span class="text-sm">语音解说</span>
              </label>
            </div>
            <div v-if="voiceOver" class="mt-3">
              <label class="block text-sm font-medium mb-2">解说文本</label>
              <textarea
                v-model="voiceOverText"
                class="w-full border border-gray-200 rounded-lg p-2 text-sm resize-none"
                rows="2"
                placeholder="请输入语音解说内容..."
              ></textarea>
            </div>
          </div>
        </div>
      </div>

      <!-- 生成按钮和状态 -->
      <div class="flex flex-col items-center mb-8">
        <button
          class="bg-blue-500 text-white px-8 py-3 rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors flex items-center gap-2 disabled:bg-gray-400 disabled:cursor-not-allowed"
          @click="generateVideo"
          :disabled="!prompt.trim() || isGenerating"
        >
          <i :class="isGenerating ? 'fa fa-spinner fa-spin' : 'fa fa-play-circle'"></i>
          <span>{{ isGenerating ? '生成中...' : '生成视频' }}</span>
        </button>

        <!-- 进度条 -->
        <div v-if="isGenerating" class="w-full max-w-md mt-4">
          <div class="flex justify-between text-xs text-gray-600 mb-1">
            <span>生成进度</span>
            <span>{{ progress }}%</span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div
              class="bg-blue-500 h-2 rounded-full transition-all duration-300"
              :style="{ width: progress + '%' }"
            ></div>
          </div>
          <div class="text-xs text-gray-500 mt-2 text-center">
            {{ statusMessage }}
            <span v-if="estimatedCompletionTime">预计完成: {{ formatTime(estimatedCompletionTime) }}</span>
          </div>
        </div>

        <!-- 错误提示 -->
        <div v-if="errorMessage" class="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg max-w-md w-full">
          <div class="flex items-center text-red-800">
            <i class="fa fa-exclamation-circle mr-2"></i>
            <span class="text-sm">{{ errorMessage }}</span>
          </div>
        </div>
      </div>

      <!-- 生成结果展示 -->
      <div v-if="videoUrl && !isGenerating" class="mt-8 border-t pt-6">
        <div class="flex justify-between items-center mb-4">
          <h3 class="text-lg font-medium flex items-center gap-2">
            <i class="fa fa-video text-green-500"></i>
            <span>生成结果</span>
          </h3>
          <div class="flex gap-2">
            <button
              @click="downloadVideo"
              class="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              <i class="fa fa-download"></i>
              <span>下载</span>
            </button>
            <button
              @click="copyVideoUrl"
              class="text-sm text-gray-600 hover:text-gray-800 flex items-center gap-1"
            >
              <i class="fa fa-copy"></i>
              <span>复制链接</span>
            </button>
            <button
              @click="refreshStatus"
              class="text-sm text-amber-600 hover:text-amber-800 flex items-center gap-1"
            >
              <i class="fa fa-sync-alt"></i>
              <span>刷新状态</span>
            </button>
          </div>
        </div>

        <!-- 视频信息 -->
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
          <div class="grid grid-cols-4 gap-4 mb-3">
            <div>
              <div class="text-xs text-gray-500">视频ID</div>
              <div class="text-sm font-medium">{{ videoInfo.videoId }}</div>
            </div>
            <div>
              <div class="text-xs text-gray-500">分辨率</div>
              <div class="text-sm font-medium">{{ videoInfo.resolution }}</div>
            </div>
            <div>
              <div class="text-xs text-gray-500">文件大小</div>
              <div class="text-sm font-medium">{{ formatFileSize(videoInfo.fileSize) }}</div>
            </div>
            <div>
              <div class="text-xs text-gray-500">生成时间</div>
              <div class="text-sm font-medium">{{ formatTime(videoInfo.timestamp) }}</div>
            </div>
          </div>
          <div class="grid grid-cols-3 gap-4">
            <div>
              <div class="text-xs text-gray-500">时长</div>
              <div class="text-sm font-medium">{{ videoInfo.duration }}秒</div>
            </div>
            <div>
              <div class="text-xs text-gray-500">帧率</div>
              <div class="text-sm font-medium">{{ videoInfo.frameRate || frameRate }} FPS</div>
            </div>
            <div>
              <div class="text-xs text-gray-500">状态</div>
              <div class="text-sm font-medium" :class="statusColor">{{ videoInfo.status || 'completed' }}</div>
            </div>
          </div>
        </div>

        <!-- 视频播放器和缩略图 -->
        <div class="grid grid-cols-3 gap-4 mb-6">
          <div class="col-span-2">
            <div class="bg-black rounded-lg overflow-hidden">
              <video
                ref="videoPlayer"
                controls
                class="w-full h-full"
                @loadedmetadata="onVideoLoaded"
              >
                <source :src="videoUrl" type="video/mp4">
                您的浏览器不支持视频播放
              </video>
            </div>
          </div>
          <div class="space-y-3">
            <div class="border border-gray-200 rounded-lg p-3">
              <div class="text-xs text-gray-500 mb-1">原始描述</div>
              <div class="text-sm line-clamp-4">{{ prompt }}</div>
            </div>
            <div class="border border-gray-200 rounded-lg p-3">
              <div class="text-xs text-gray-500 mb-1">生成参数</div>
              <div class="text-xs space-y-1">
                <div>风格: {{ getStyleName(style) }}</div>
                <div>宽高比: {{ aspectRatio }}</div>
                <div>背景音乐: {{ backgroundMusic ? '是' : '否' }}</div>
                <div>语音解说: {{ voiceOver ? '是' : '否' }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 历史记录 -->
        <div v-if="videoHistory.length > 0" class="mt-6">
          <div class="flex justify-between items-center mb-3">
            <h4 class="text-sm font-medium text-gray-700">最近生成</h4>
            <button
              @click="clearHistory"
              class="text-xs text-gray-500 hover:text-gray-700"
            >
              清空历史
            </button>
          </div>
          <div class="grid grid-cols-3 gap-3">
            <div
              v-for="(item, index) in videoHistory.slice(0, 3)"
              :key="index"
              class="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 cursor-pointer"
              @click="playHistoryItem(item)"
            >
              <div class="aspect-video bg-gray-100 rounded mb-2 flex items-center justify-center">
                <i class="fa fa-play text-gray-400"></i>
              </div>
              <div class="text-xs truncate mb-1">{{ item.prompt.substring(0, 30) }}{{ item.prompt.length > 30 ? '...' : '' }}</div>
              <div class="flex justify-between text-xs text-gray-500">
                <span>{{ formatTime(item.timestamp) }}</span>
                <span>{{ item.duration }}s</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import axios from 'axios';

// API基础URL
const API_BASE_URL = 'http://localhost:8000';

// 响应式数据
const prompt = ref('');
const style = ref('animation');
const duration = ref('5');
const resolution = ref('720p');
const aspectRatio = ref('16:9');
const frameRate = ref(30);
const backgroundMusic = ref(false);
const voiceOver = ref(false);
const voiceOverText = ref('');

// 状态管理
const videoUrl = ref('');
const isGenerating = ref(false);
const progress = ref(0);
const errorMessage = ref('');
const statusMessage = ref('');
const estimatedCompletionTime = ref('');
const generatedCount = ref(0);
const activeJobs = ref(0);
const videoHistory = ref<Array<any>>([]);

// 视频信息
const videoInfo = ref({
  videoId: '',
  format: 'mp4',
  fileSize: 0,
  duration: 0,
  resolution: '720p',
  frameRate: 30,
  status: 'completed',
  timestamp: ''
});

// 计算属性
const promptLength = computed(() => prompt.value.length);
const estimatedTime = computed(() => Math.max(5, Math.floor(promptLength.value / 10) * 2));
const statusColor = computed(() => {
  switch (videoInfo.value.status) {
    case 'pending': return 'text-yellow-600';
    case 'processing': return 'text-blue-600';
    case 'completed': return 'text-green-600';
    case 'failed': return 'text-red-600';
    default: return 'text-gray-600';
  }
});

// 视频播放器引用
const videoPlayer = ref<HTMLVideoElement | null>(null);

// 初始化
onMounted(() => {
  // 从本地存储加载历史记录
  const savedHistory = localStorage.getItem('videoHistory');
  if (savedHistory) {
    try {
      videoHistory.value = JSON.parse(savedHistory);
    } catch (e) {
      console.error('加载视频历史记录失败:', e);
    }
  }

  const savedCount = localStorage.getItem('generatedVideoCount');
  if (savedCount) {
    generatedCount.value = parseInt(savedCount);
  }

  // 获取支持的视频风格
  fetchSupportedStyles();
});

// 更新提示统计
const updatePromptStats = () => {
  // 可以添加文本分析逻辑
};

// 格式化文件大小
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// 格式化时间
const formatTime = (timestamp: string): string => {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (e) {
    return timestamp;
  }
};

// 获取风格名称
const getStyleName = (styleValue: string): string => {
  const styleMap: Record<string, string> = {
    'animation': '动画风格',
    'realistic': '真人实拍',
    '3d': '3D建模',
    'cartoon': '卡通漫画',
    'cinematic': '电影感',
    'watercolor': '水彩画风'
  };
  return styleMap[styleValue] || styleValue;
};

// 生成视频
const generateVideo = async () => {
  if (!prompt.value.trim()) {
    errorMessage.value = '请输入视频描述';
    return;
  }

  // 重置状态
  isGenerating.value = true;
  progress.value = 0;
  errorMessage.value = '';
  statusMessage.value = '正在提交生成请求...';

  try {
    // 模拟进度条
    const progressInterval = setInterval(() => {
      if (progress.value < 90) {
        progress.value += 5;
        statusMessage.value = `正在生成视频... ${progress.value}%`;
      }
    }, 500);

    // 调用后端API
    const response = await axios.post(`${API_BASE_URL}/api/v1/video/generate`, {
      prompt: prompt.value,
      style: style.value,
      duration: parseInt(duration.value),
      resolution: resolution.value,
      aspect_ratio: aspectRatio.value,
      frame_rate: frameRate.value,
      background_music: backgroundMusic.value,
      voice_over: voiceOver.value,
      voice_over_text: voiceOverText.value
    });

    clearInterval(progressInterval);
    progress.value = 100;
    statusMessage.value = '视频生成完成！';

    if (response.data.success) {
      const data = response.data;

      // 更新视频信息
      videoInfo.value = {
        videoId: data.video_id,
        format: data.format,
        fileSize: data.file_size,
        duration: data.duration,
        resolution: data.resolution,
        frameRate: frameRate.value,
        status: 'completed',
        timestamp: data.timestamp
      };

      // 构建完整URL
      videoUrl.value = `${API_BASE_URL}${data.video_url}`;

      // 更新生成次数
      generatedCount.value++;
      localStorage.setItem('generatedVideoCount', generatedCount.value.toString());

      // 保存到历史记录
      const historyItem = {
        prompt: prompt.value,
        videoUrl: videoUrl.value,
        timestamp: data.timestamp,
        videoId: data.video_id,
        duration: data.duration,
        style: style.value
      };

      videoHistory.value.unshift(historyItem);
      if (videoHistory.value.length > 10) {
        videoHistory.value = videoHistory.value.slice(0, 10);
      }
      localStorage.setItem('videoHistory', JSON.stringify(videoHistory.value));

      // 延迟重置进度条
      setTimeout(() => {
        isGenerating.value = false;
        progress.value = 0;
        statusMessage.value = '';
      }, 1000);

    } else {
      throw new Error(response.data.message || '生成失败');
    }

  } catch (error: any) {
    console.error('生成视频失败:', error);
    errorMessage.value = error.response?.data?.detail || error.message || '生成视频失败，请重试！';
    isGenerating.value = false;
    progress.value = 0;
    statusMessage.value = '';
  }
};

// 检查视频状态
const checkVideoStatus = async (videoId: string) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/v1/video/status`, {
      video_id: videoId
    });

    if (response.data.success) {
      videoInfo.value.status = response.data.status;
      videoInfo.value.timestamp = response.data.timestamp;

      if (response.data.status === 'processing') {
        progress.value = response.data.progress;
        statusMessage.value = response.data.message || '正在处理中...';
        estimatedCompletionTime.value = response.data.estimated_completion_time || '';
      }
    }
  } catch (error) {
    console.error('检查视频状态失败:', error);
  }
};

// 下载视频
const downloadVideo = () => {
  if (!videoUrl.value) return;

  const link = document.createElement('a');
  link.href = videoUrl.value;
  link.download = `video_${videoInfo.value.videoId}.${videoInfo.value.format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// 复制视频URL
const copyVideoUrl = async () => {
  if (!videoUrl.value) return;

  try {
    await navigator.clipboard.writeText(videoUrl.value);
    alert('视频链接已复制到剪贴板');
  } catch (err) {
    console.error('复制失败:', err);
    alert('复制失败，请手动复制');
  }
};

// 刷新状态
const refreshStatus = () => {
  if (videoInfo.value.videoId) {
    checkVideoStatus(videoInfo.value.videoId);
  }
};

// 视频加载完成
const onVideoLoaded = () => {
  if (videoPlayer.value) {
    // 可以添加视频播放逻辑
  }
};

// 播放历史记录项
const playHistoryItem = (item: any) => {
  prompt.value = item.prompt;
  style.value = item.style;
  duration.value = item.duration.toString();
  videoUrl.value = item.videoUrl;
  videoInfo.value = {
    videoId: item.videoId,
    format: item.videoUrl.split('.').pop() || 'mp4',
    fileSize: 0,
    duration: item.duration,
    resolution: '720p',
    frameRate: 30,
    status: 'completed',
    timestamp: item.timestamp
  };
};

// 清空历史记录
const clearHistory = () => {
  if (confirm('确定要清空历史记录吗？')) {
    videoHistory.value = [];
    localStorage.removeItem('videoHistory');
  }
};

// 获取支持的视频风格
const fetchSupportedStyles = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/v1/video/supported-styles`);
    console.log('支持的视频风格:', response.data);
  } catch (error) {
    console.error('获取支持的视频风格失败:', error);
  }
};
</script>