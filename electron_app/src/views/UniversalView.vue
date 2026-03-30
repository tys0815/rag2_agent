<template>
  <div class="flex flex-col h-full overflow-hidden">
    <!-- 聊天内容区 -->
    <div class="flex-1 p-4 md:p-6 overflow-y-auto" ref="chatContainer">
      <div class="max-w-3xl mx-auto space-y-6">
        <!-- 空状态 -->
        <div v-if="!currentHistory" class="flex flex-col items-center justify-center h-full py-10 text-center">
          <i class="fa fa-comments text-5xl text-gray-300 mb-4"></i>
          <h3 class="text-lg font-medium mb-2">开始新的对话</h3>
          <p class="text-gray-500 text-sm">输入消息开始你的全能企业级助手对话吧</p>
        </div>

        <!-- 聊天记录：渲染当前历史记录的 list 数据 -->
        <div v-else>
          <div v-for="(message, index) in currentHistory.list" :key="index" class="flex">
            <!-- 用户消息 -->
            <div v-if="message.role === 'user'" class="ml-auto max-w-[75%]">
              <div class="flex items-start gap-3">
                <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                  <i class="fa fa-user text-blue-500 text-sm"></i>
                </div>
                <div class="bg-blue-50 p-3 rounded-lg rounded-tr-none">
                  <p class="text-sm whitespace-pre-wrap" :class="{ 'text-red-500 font-bold': message.content.startsWith('❌') }">
                    {{ message.content }}
                  </p>
                </div>
              </div>
            </div>

            <!-- 助手/系统消息 -->
            <div v-else class="mr-auto max-w-[75%]">
              <div class="flex items-start gap-3">
                <div class="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center flex-shrink-0">
                  <i class="fa fa-robot text-gray-500 text-sm"></i>
                </div>
                <div class="bg-gray-50 p-3 rounded-lg rounded-tl-none">
                  <p class="text-sm whitespace-pre-wrap">{{ message.content }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 输入区 -->
    <div class="border-t border-gray-200 p-4">
      <div class="max-w-3xl mx-auto">
        <form @submit.prevent="sendMessage" class="flex flex-col gap-3">
          <textarea
            v-model="messageInput"
            class="w-full border border-gray-200 rounded-lg p-3 min-h-[80px] focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 resize-none text-sm"
            placeholder="全能助手-输入消息..."
            :disabled="!currentHistory || isLoading"
          ></textarea>
          <div class="flex justify-end">
            <button
              type="submit"
              class="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors flex items-center gap-2"
              :disabled="!messageInput.trim() || !currentHistory || isLoading"
            >
              <i class="fa fa-paper-plane"></i>
              <span>发送</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick } from 'vue';
import { useAppStore, Message } from '../stores/appStore';
import { storeToRefs } from 'pinia';

// 初始化 Pinia store
const appStore = useAppStore();
const { apiUrl, currentView } = storeToRefs(appStore);

// 计算属性：监听当前选中的历史记录（响应式更新）
const currentHistory = computed(() => appStore.getCurrentHistory);

// 输入框内容
const messageInput = ref('');

// 加载状态（用于禁用按钮和输入框）
const isLoading = ref(false);

// 聊天容器引用（用于滚动到底部）
const chatContainer = ref<HTMLDivElement | null>(null);

// 转义HTML特殊字符（防止XSS，已启用）
const escapeHtml = (str: string) => {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
};

// 发送消息
const sendMessage = async () => {
  const history = currentHistory.value;
    console.log('全能助手', currentView);
    if (!messageInput.value.trim() || !currentHistory.value || isLoading.value) return;

    // 1. 构建用户消息
    const userMessage: Message = {
      role: 'user',
      content: escapeHtml(messageInput.value.trim())
    };
    const newMessage: string = messageInput.value.trim();
    // 2. 清空输入框
    messageInput.value = '';

    // 3. 追加用户消息到当前历史记录
    appStore.appendMessageToHistory(history.id, userMessage);

    // 4. 初始化助手消息（用于流式更新）
    const assistantMessage: Message = {
      role: 'system',
      content: ''
    };
    isLoading.value = true;
    let messageId = '';
    // 追加空的助手消息到历史记录
    messageId = appStore.appendMessageToHistory(currentHistory.value.id, assistantMessage);

    // 全能企业级助手API
    const apiEndpoint = 'universal/chat';
    const requestBody = {
      text: newMessage,
      user_id: 'uid_12345',
      agent_id: 'universal_assistant',
      session_id: `universal_session_20260331`,
      enable_memory: true,
      enable_rag: true,
      max_context_length: 2000,
      tool_choice: 'auto',
      max_tool_iterations: 5
    };

    const response = await fetch(`${apiUrl.value}${apiEndpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });
    console.log('!!!!!!!', response);
    if (!response.status || response.status !== 200) {
      throw new Error(`API 响应错误: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();

    // 更新store中的消息内容
    const responseText = data.data || data.response || data;
    appStore.updateMessageContent(currentHistory.value.id, messageId, responseText);
    // 滚动到底部
    await nextTick();
    chatContainer.value?.scrollTo({
      top: chatContainer.value.scrollHeight,
      behavior: 'smooth'
    });
};
</script>