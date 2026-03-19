<template>
  <div class="p-6 h-full flex flex-col">
    <div class="max-w-2xl mx-auto w-full">
      <h2 class="text-xl font-bold mb-6 flex items-center gap-2">
        <i class="fa fa-microphone text-blue-500"></i>
        <span>语音生成</span>
      </h2>

      <!-- 统计信息卡片 -->
      <div class="grid grid-cols-3 gap-4 mb-6">
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-blue-600">{{ textLength }}</div>
          <div class="text-sm text-blue-800 mt-1">文本长度</div>
        </div>
        <div class="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-green-600">{{ estimatedDuration }}</div>
          <div class="text-sm text-green-800 mt-1">预估时长(秒)</div>
        </div>
        <div class="bg-purple-50 border border-purple-200 rounded-lg p-4 text-center">
          <div class="text-2xl font-bold text-purple-600">{{ generatedCount }}</div>
          <div class="text-sm text-purple-800 mt-1">生成次数</div>
        </div>
      </div>

      <div class="border border-gray-200 rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center mb-2">
          <label class="block text-sm font-medium">输入文本</label>
          <span class="text-xs text-gray-500">{{ textLength }}/5000 字符</span>
        </div>
        <textarea
          class="w-full border border-gray-200 rounded-lg p-3 min-h-[120px] focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 resize-none text-sm"
          v-model="textContent"
          placeholder="请输入要转换为语音的文本内容..."
          maxlength="5000"
          @input="updateTextStats"
        ></textarea>
      </div>

      <div class="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label class="block text-sm font-medium mb-2">语音类型</label>
          <select
            class="w-full border border-gray-200 rounded-lg p-3 focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 text-sm"
            v-model="voiceType"
          >
            <option value="female">甜美女生</option>
            <option value="male">阳光男生</option>
            <option value="child">卡通童声</option>
            <option value="professional">专业播音</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-2">语速</label>
          <select
            class="w-full border border-gray-200 rounded-lg p-3 focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 text-sm"
            v-model="speed"
          >
            <option value="slow">慢速 (0.8x)</option>
            <option value="normal">正常 (1.0x)</option>
            <option value="fast">快速 (1.2x)</option>
          </select>
        </div>
      </div>

      <div class="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label class="block text-sm font-medium mb-2">输出格式</label>
          <select
            class="w-full border border-gray-200 rounded-lg p-3 focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 text-sm"
            v-model="outputFormat"
          >
            <option value="mp3">MP3</option>
            <option value="wav">WAV</option>
            <option value="ogg">OGG</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-2">语言</label>
          <select
            class="w-full border border-gray-200 rounded-lg p-3 focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 text-sm"
            v-model="language"
          >
            <option value="zh-CN">中文</option>
            <option value="en-US">英文</option>
            <option value="ja-JP">日文</option>
          </select>
        </div>
      </div>

      <!-- 音调和音量滑块 -->
      <div class="grid grid-cols-2 gap-4 mb-8">
        <div>
          <label class="block text-sm font-medium mb-2">
            音调: {{ pitch.toFixed(1) }}
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            v-model="pitch"
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div class="flex justify-between text-xs text-gray-500 mt-1">
            <span>低沉</span>
            <span>正常</span>
            <span>尖锐</span>
          </div>
        </div>
        <div>
          <label class="block text-sm font-medium mb-2">
            音量: {{ volume.toFixed(1) }}
          </label>
          <input
            type="range"
            min="0.0"
            max="2.0"
            step="0.1"
            v-model="volume"
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div class="flex justify-between text-xs text-gray-500 mt-1">
            <span>静音</span>
            <span>正常</span>
            <span>最大</span>
          </div>
        </div>
      </div>

      <!-- 生成按钮和状态 -->
      <div class="flex flex-col items-center">
        <button
          class="bg-blue-500 text-white px-8 py-3 rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors flex items-center gap-2 disabled:bg-gray-400 disabled:cursor-not-allowed"
          @click="generateVoice"
          :disabled="!textContent.trim() || isGenerating"
        >
          <i :class="isGenerating ? 'fa fa-spinner fa-spin' : 'fa fa-play'"></i>
          <span>{{ isGenerating ? '生成中...' : '生成语音' }}</span>
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
      <div v-if="voiceUrl && !isGenerating" class="mt-8 border-t pt-6">
        <div class="flex justify-between items-center mb-4">
          <h3 class="text-lg font-medium flex items-center gap-2">
            <i class="fa fa-music text-green-500"></i>
            <span>生成结果</span>
          </h3>
          <div class="flex gap-2">
            <button
              @click="downloadVoice"
              class="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              <i class="fa fa-download"></i>
              <span>下载</span>
            </button>
            <button
              @click="copyVoiceUrl"
              class="text-sm text-gray-600 hover:text-gray-800 flex items-center gap-1"
            >
              <i class="fa fa-copy"></i>
              <span>复制链接</span>
            </button>
          </div>
        </div>

        <!-- 语音信息 -->
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
          <div class="grid grid-cols-3 gap-4 mb-3">
            <div>
              <div class="text-xs text-gray-500">文件格式</div>
              <div class="text-sm font-medium">{{ voiceInfo.format }}</div>
            </div>
            <div>
              <div class="text-xs text-gray-500">文件大小</div>
              <div class="text-sm font-medium">{{ formatFileSize(voiceInfo.fileSize) }}</div>
            </div>
            <div>
              <div class="text-xs text-gray-500">生成时间</div>
              <div class="text-sm font-medium">{{ formatTime(voiceInfo.timestamp) }}</div>
            </div>
          </div>
          <div class="text-xs text-gray-500">语音ID: {{ voiceInfo.voiceId }}</div>
        </div>

        <!-- 音频播放器 -->
        <div class="bg-white border border-gray-200 rounded-lg p-4">
          <audio
            ref="audioPlayer"
            controls
            class="w-full"
            @loadedmetadata="onAudioLoaded"
          >
            <source :src="voiceUrl" :type="'audio/' + voiceInfo.format">
            您的浏览器不支持音频播放
          </audio>
          <div class="text-xs text-gray-500 mt-2 text-center">
            时长: {{ voiceInfo.duration }}秒 | 文本长度: {{ voiceInfo.textLength }}字符
          </div>
        </div>

        <!-- 历史记录 -->
        <div v-if="voiceHistory.length > 0" class="mt-6">
          <h4 class="text-sm font-medium mb-2 text-gray-700">最近生成</h4>
          <div class="space-y-2">
            <div
              v-for="(item, index) in voiceHistory.slice(0, 3)"
              :key="index"
              class="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 cursor-pointer"
              @click="playHistoryItem(item)"
            >
              <div class="flex justify-between items-center">
                <div class="text-sm truncate">{{ item.text.substring(0, 50) }}{{ item.text.length > 50 ? '...' : '' }}</div>
                <div class="text-xs text-gray-500">{{ formatTime(item.timestamp) }}</div>
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
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// 响应式数据
const textContent = ref('');
const voiceType = ref('female');
const speed = ref('normal');
const outputFormat = ref('mp3');
const language = ref('zh-CN');
const pitch = ref(1.0);
const volume = ref(1.0);

// 状态管理
const voiceUrl = ref('');
const isGenerating = ref(false);
const progress = ref(0);
const errorMessage = ref('');
const generatedCount = ref(0);
const voiceHistory = ref<Array<any>>([]);

// 语音信息
const voiceInfo = ref({
  voiceId: '',
  format: 'mp3',
  fileSize: 0,
  duration: 0,
  textLength: 0,
  timestamp: ''
});

// 计算属性
const textLength = computed(() => textContent.value.length);
const estimatedDuration = computed(() => Math.max(1, Math.floor(textLength.value / 15)));

// 音频播放器引用
const audioPlayer = ref<HTMLAudioElement | null>(null);

// 初始化
onMounted(() => {
  // 从本地存储加载历史记录
  const savedHistory = localStorage.getItem('voiceHistory');
  if (savedHistory) {
    try {
      voiceHistory.value = JSON.parse(savedHistory);
    } catch (e) {
      console.error('加载语音历史记录失败:', e);
    }
  }

  const savedCount = localStorage.getItem('generatedVoiceCount');
  if (savedCount) {
    generatedCount.value = parseInt(savedCount);
  }
});

// 更新文本统计
const updateTextStats = () => {
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

// 生成语音
const generateVoice = async () => {
  if (!textContent.value.trim()) {
    errorMessage.value = '请输入文本内容';
    return;
  }

  // 重置状态
  isGenerating.value = true;
  progress.value = 0;
  errorMessage.value = '';
  voiceUrl.value = '';

  try {
    // 模拟进度条
    const progressInterval = setInterval(() => {
      if (progress.value < 90) {
        progress.value += 10;
      }
    }, 300);

    // 调用后端API
    const response = await axios.post(`${API_BASE_URL}/api/v1/voice/generate`, {
      text: textContent.value,
      voice_type: voiceType.value,
      speed: speed.value,
      pitch: pitch.value,
      volume: volume.value,
      output_format: outputFormat.value,
      language: language.value
    });

    clearInterval(progressInterval);
    progress.value = 100;

    if (response.data.success) {
      const data = response.data;

      // 更新语音信息
      voiceInfo.value = {
        voiceId: data.voice_id,
        format: data.format,
        fileSize: data.file_size,
        duration: data.duration,
        textLength: data.text_length,
        timestamp: data.timestamp
      };

      // 构建完整URL
      voiceUrl.value = `${API_BASE_URL}${data.voice_url}`;

      // 更新生成次数
      generatedCount.value++;
      localStorage.setItem('generatedVoiceCount', generatedCount.value.toString());

      // 保存到历史记录
      const historyItem = {
        text: textContent.value,
        voiceUrl: voiceUrl.value,
        timestamp: data.timestamp,
        voiceId: data.voice_id
      };

      voiceHistory.value.unshift(historyItem);
      if (voiceHistory.value.length > 10) {
        voiceHistory.value = voiceHistory.value.slice(0, 10);
      }
      localStorage.setItem('voiceHistory', JSON.stringify(voiceHistory.value));

      // 延迟重置进度条
      setTimeout(() => {
        isGenerating.value = false;
        progress.value = 0;
      }, 500);

    } else {
      throw new Error(response.data.message || '生成失败');
    }

  } catch (error: any) {
    console.error('生成语音失败:', error);
    errorMessage.value = error.response?.data?.detail || error.message || '生成语音失败，请重试！';
    isGenerating.value = false;
    progress.value = 0;
  }
};

// 下载语音
const downloadVoice = () => {
  if (!voiceUrl.value) return;

  const link = document.createElement('a');
  link.href = voiceUrl.value;
  link.download = `voice_${voiceInfo.value.voiceId}.${voiceInfo.value.format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// 复制语音URL
const copyVoiceUrl = async () => {
  if (!voiceUrl.value) return;

  try {
    await navigator.clipboard.writeText(voiceUrl.value);
    // 可以添加成功提示
    alert('语音链接已复制到剪贴板');
  } catch (err) {
    console.error('复制失败:', err);
    alert('复制失败，请手动复制');
  }
};

// 音频加载完成
const onAudioLoaded = () => {
  if (audioPlayer.value) {
    // 可以添加音频播放逻辑
  }
};

// 播放历史记录项
const playHistoryItem = (item: any) => {
  voiceUrl.value = item.voiceUrl;
  voiceInfo.value = {
    voiceId: item.voiceId,
    format: item.voiceUrl.split('.').pop() || 'mp3',
    fileSize: 0,
    duration: 0,
    textLength: item.text.length,
    timestamp: item.timestamp
  };
};

// 获取支持的语音类型
const fetchSupportedVoices = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/v1/voice/supported-voices`);
    console.log('支持的语音类型:', response.data);
  } catch (error) {
    console.error('获取支持的语音类型失败:', error);
  }
};

// 组件加载时获取支持的语音类型
onMounted(() => {
  fetchSupportedVoices();
});
</script>