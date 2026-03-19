<template>
  <div class="flex flex-col h-full overflow-hidden">
    <!-- 聊天内容区 -->
    <div class="flex-1 p-4 md:p-6 overflow-y-auto" ref="chatContainer">
      <div class="max-w-3xl mx-auto space-y-6">
        <!-- 空状态 -->
        <div v-if="!currentHistory" class="flex flex-col items-center justify-center h-full py-10 text-center">
          <i class="fa fa-comments text-5xl text-gray-300 mb-4"></i>
          <h3 class="text-lg font-medium mb-2">开始新的对话</h3>
          <p class="text-gray-500 text-sm">输入消息开始你的对话吧</p>
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
            placeholder="输入消息..."
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
const { system_prompt, llmModel, apiUrl, KnowledgeBaseItem,currentView } = storeToRefs(appStore);

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
   let newChat = true;
  //  appStore.resetCache();
  if (newChat) {
    const history = currentHistory.value;
    console.log('!!!!!!!', currentView.value);
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

    // 默认聊天API (llmModel)
    const apiEndpoint = 'new_chat/completions';
    const requestBody = {
      text: newMessage,
      namespace: 'uid_12345',
      agent_id: 'agent_001',
      session_id: `session_${Date.now()}`,
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
   
  }else{
    const history = currentHistory.value;
  console.log('!!!!!!!', history);
  if (!messageInput.value.trim() || !currentHistory.value || isLoading.value) return;

  // 1. 构建用户消息
  const userMessage: Message = {
    role: 'user',
    content: escapeHtml(messageInput.value.trim())
  };

  // 2. 清空输入框
  messageInput.value = '';

  // 3. 追加用户消息到当前历史记录
  appStore.appendMessageToHistory(history.id, userMessage);

  // 4. 初始化助手消息（用于流式更新）
  const assistantMessage: Message = {
    role: 'system',
    content: ''
  };

  // 5. 设置加载状态
  isLoading.value = true;
  let messageId = '';
  // 追加空的助手消息到历史记录
   messageId = appStore.appendMessageToHistory(currentHistory.value.id, assistantMessage);
  try {
    // 构建请求参数
    const messages: Message[] = [];
    if (system_prompt.value) {
      messages.push({ role: 'system', content: system_prompt.value });
    }
    // 合并历史消息（可根据实际需求调整，比如只传最近的消息）
    currentHistory.value.list.forEach((msg) => {
      messages.push({ role: msg.role, content: msg.content });
    });

    const requestBody = {
      model: llmModel.value,
      messages: messages,
      stream: true,
      use_context: KnowledgeBaseItem.value.length > 0,
      include_sources: true,
      ...(KnowledgeBaseItem.value.length > 0 && {
        context_filter: { docs_ids: KnowledgeBaseItem.value.map(item => item.doc_id) }
      })
    };

    console.log('请求参数:', JSON.stringify(requestBody));

    // 发送POST请求（使用fetch而非axios，因为axios处理流式响应较复杂）
    const response = await fetch(`${apiUrl.value}chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`API 响应错误: ${response.status} ${response.statusText}`);
    }

    // 获取流式响应的读取器
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('无法获取响应流读取器');
    }
    const decoder = new TextDecoder('utf-8');
    let aiContent = '';

    // 循环读取流式数据
    try {
      let incompleteJson = ''; // 全局缓存：仅定义一次
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // 解码二进制数据（保留不完整字符）
        const chunk = decoder.decode(value, { stream: true });
        // 按行分割数据（过滤空行）
        const lines = chunk.split('\n').filter((line) => line.trim());

        for (const line of lines) {
          try {
            // 处理data: 前缀（如OpenAI的流式响应格式）
            let dataStr = line.startsWith('data: ') ? line.slice(6).trim() : line.trim();
            // 跳过结束标记（用continue，不终止循环）
            if (dataStr === '[DONE]') {
              incompleteJson = '';
              continue;
            }

            // 拼接全局缓存的不完整JSON
            dataStr = incompleteJson + dataStr;
            incompleteJson = ''; // 拼接后清空缓存

            const data = JSON.parse(dataStr);
            let newContent = '';

            // 兼容不同的响应格式
            if (data.choices && data.choices.length > 0) {
              newContent = data.choices[0].delta?.content || '';
              // 若收到结束标记，清空缓存并跳过当前行
              const finishReason = data.choices[0].finish_reason;
              if (finishReason === 'stop' || finishReason === 'completed') {
                incompleteJson = '';
                continue;
              }
            } else if (data.response) {
              newContent = data.response;
            }

            // 更新助手消息内容
            if (newContent) {
              aiContent += escapeHtml(newContent);
              // 更新store中的消息内容
              appStore.updateMessageContent(currentHistory.value.id, messageId, aiContent);
              // 滚动到底部
              await nextTick();
              chatContainer.value?.scrollTo({
                top: chatContainer.value.scrollHeight,
                behavior: 'smooth'
              });
            }
          } catch (parseError) {
            if (parseError instanceof SyntaxError) {
              incompleteJson = line.startsWith('data: ') ? line.slice(6).trim() : line.trim();
              console.warn('缓存不完整JSON片段，待后续拼接:', incompleteJson);
            } else {
              console.warn('解析流式数据错误:', parseError, '行内容:', line);
            }
          }
        }
      }

      // 兜底解析：最后剩余的不完整JSON片段
      if (incompleteJson?.trim()) {
        try {
          const data = JSON.parse(incompleteJson);
          let newContent = '';
          if (data.choices && data.choices.length > 0) {
            newContent = data.choices[0].delta?.content || data.choices[0].message?.content || '';
          } else if (data.response) {
            newContent = data.response;
          }
          if (newContent) {
            aiContent += escapeHtml(newContent);
            appStore.updateMessageContent(currentHistory.value.id, messageId, aiContent);
            await nextTick();
            chatContainer.value?.scrollTo({
              top: chatContainer.value.scrollHeight,
              behavior: 'smooth'
            });
          }
        } catch (e) {
          console.warn('最后剩余JSON片段解析失败（可忽略）:', e);
        }
      }

      // 最终解码：获取最后剩余的字符
      const finalChunk = decoder.decode();
      if (finalChunk.trim()) {
        const lines = finalChunk.split('\n').filter((line) => line.trim());
        for (const line of lines) {
          try {
            let dataStr = line.startsWith('data: ') ? line.slice(6).trim() : line.trim();
            if (dataStr === '[DONE]') continue;
            const data = JSON.parse(dataStr);
            let newContent = data.choices?.[0]?.delta?.content || '';
            if (newContent) {
              aiContent += escapeHtml(newContent);
              appStore.updateMessageContent(currentHistory.value.id, messageId, aiContent);
              await nextTick();
              chatContainer.value?.scrollTo({
                top: chatContainer.value.scrollHeight,
                behavior: 'smooth'
              });
            }
          } catch (e) {
            console.warn('最终片段解析失败:', e);
          }
        }
      }
    } catch (streamError) {
      console.error('流式读取失败:', streamError);
      aiContent = `❌ 流式响应读取失败：${escapeHtml(streamError.message)}`;
      appStore.updateMessageContent(currentHistory.value.id, messageId, aiContent);
    }

    // 处理空响应
    if (!aiContent.trim()) {
      appStore.updateMessageContent(
        currentHistory.value.id,
        messageId,
        '哎呀，没获取到有效回复呢 😥，请再问一次吧～'
      );
    }
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : '未知错误';
    console.error('Ollama API 调用错误：', error);
    // 更新错误消息（添加非空判断）
    if (messageId) {
      appStore.updateMessageContent(
        currentHistory.value.id,
        messageId,
        `❌ 调用失败：${escapeHtml(errorMsg)}`
      );
    }
  } finally {
    // 重置加载状态
    isLoading.value = false;
  }
  }
 
};
</script>