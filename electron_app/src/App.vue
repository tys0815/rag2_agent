<template>
  <div class="flex flex-col h-screen bg-white text-black overflow-hidden">
    <!-- 头部组件 -->
    <Header />

    <!-- 主内容区 -->
    <div class="flex flex-1 overflow-hidden">
      <!-- 左侧列表区 -->
      <aside class="w-64 border-r border-gray-200 bg-white flex-shrink-0 overflow-y-auto">
        <!-- 我的聊天列表 -->
        <ChatList />

        <!-- 历史记录（原我的知识库） -->
        <HistoryList />
      </aside>

      <!-- 右侧功能区（根据currentView渲染对应组件） -->
      <main class="flex-1 overflow-hidden">
        <!-- 聊天视图 -->
        <ChatView v-if="appStore.currentView === 'llmModel' || appStore.currentView === 'travelModel'" />
        <!-- 语音生成视图 -->
        <VoiceView v-if="appStore.currentView === 'soulModel'" />
        <!-- 图像生成视图 -->
        <ImageView v-if="appStore.currentView === 'imgModel'" />
        <!-- 视频生成视图 -->
        <VideoView v-if="appStore.currentView === 'vidolModel'" />
      </main>
    </div>

    <!-- 底部组件 -->
    <Footer />
  </div>
</template>

<script setup lang="ts">
import { useAppStore } from './stores/appStore';
import Header from './components/Header.vue';
import Footer from './components/Footer.vue';
import ChatView from './views/ChatView.vue';
import VoiceView from './views/VoiceView.vue';
import ImageView from './views/ImageView.vue';
import VideoView from './views/VideoView.vue';
import HistoryList from './components/HistoryList.vue';
import ChatList from './components/ChatList.vue'; 

// 初始化Pinia store
const appStore = useAppStore();

</script>

<style scoped>
/* 自定义滚动条样式 */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: #d1d5db;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #9ca3af;
}
</style>