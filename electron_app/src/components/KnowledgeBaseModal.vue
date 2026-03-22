<template>
  <div v-if="modelValue" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50" @click="closeModal">
    <div class="bg-white rounded-lg w-full max-w-2xl p-6 shadow-lg" @click.stop>
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-gray-800">知识库管理</h3>
        <button class="text-gray-500 hover:text-gray-700" @click="closeModal">
          <i class="fa fa-times"></i>
        </button>
      </div>
      <div class="space-y-4">
        <!-- 知识库列表：动态渲染接口数据 -->
        <div class="border border-gray-200 rounded-md divide-y divide-gray-200 max-h-60 overflow-y-auto">
          <!-- 加载中状态 -->
          <div v-if="loading" class="px-4 py-8 text-center text-gray-500">
            <i class="fa fa-spinner fa-spin mr-2"></i>
            <span>加载中...</span>
          </div>
          <!-- 空状态 -->
          <div v-else-if="KnowledgeBaseItem.length === 0" class="px-4 py-8 text-center text-gray-500">
            <i class="fa fa-folder-open-o mr-2"></i>
            <span>暂无上传的文件</span>
          </div>
          <!-- 动态列表项 -->
          <div
            v-for="(item, index) in KnowledgeBaseItem"
            :key="index"
            class="px-4 py-3 flex items-center justify-between"
          >
            <div class="flex items-center gap-3">
              <i class="fa fa-file-text-o text-blue-500"></i>
              <span class="text-gray-800">{{ item.doc_metadata.file_name }}</span>
            </div>
            <div class="flex gap-2">
              <button class="text-blue-500 hover:text-blue-700 text-sm" @click="handleEdit(item)">
                编辑
              </button>
              <button class="text-red-500 hover:text-red-700 text-sm" @click="handleDelete(item.doc_id, index)">
                删除
              </button>
            </div>
          </div>
        </div>

        <!-- 上传模式选择 -->
        <div class="flex items-center gap-4 mb-4">
          <label class="text-sm font-medium text-gray-700">上传模式：</label>
          <select v-model="uploadMode" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option value="single">单文件</option>
            <option value="multiple">多文件</option>
            <option value="directory">文件夹</option>
          </select>
        </div>

        <!-- 上传文件区域：加载状态 + 上传按钮 -->
        <div class="flex justify-end pt-2">
          <button
            v-if="uploading"
            class="px-4 py-2 bg-green-500 text-white rounded-md flex items-center gap-2 opacity-75 cursor-not-allowed"
            disabled
          >
            <i class="fa fa-spinner fa-spin"></i>
            <span>上传中...</span>
          </button>
          <button
            v-else
            class="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors flex items-center gap-2"
            @click="triggerFileInput"
          >
            <i class="fa fa-plus"></i>
            <span>{{ uploadButtonText }}</span>
          </button>

          <!-- 隐藏的文件选择器，支持多文件和文件夹选择 -->
          <input
            ref="fileInputRef"
            type="file"
            class="hidden"
            accept=".txt,.md,.pdf,.docx,.doc,.json"
            :multiple="uploadMode === 'multiple' || uploadMode === 'directory'"
            :webkitdirectory="uploadMode === 'directory'"
            :directory="uploadMode === 'directory'"
            @change="handleFileSelect"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, watch, computed } from 'vue';
import { storeToRefs } from 'pinia';
import { useAppStore } from '../stores/appStore';
const appStore = useAppStore();
const { apiUrl, KnowledgeBaseItem } = storeToRefs(appStore);

// 定义类型
interface KnowledgeBaseItem {
  doc_id: string;
  doc_metadata: {
    file_name: string;
  };
}

// 定义 Props
const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false,
  },
});

// 定义 Emits
const emit = defineEmits(['update:modelValue']);

// 响应式数据
const fileInputRef = ref<HTMLInputElement | null>(null);
const loading = ref(false); // 列表加载状态
const uploading = ref(false); // 上传加载状态
const uploadMode = ref('multiple'); // 'single', 'multiple', 'directory'
let knowledgeBaseList = reactive<KnowledgeBaseItem[]>([]); // 知识库文件列表
let kg_knowledgeBaseList = reactive<KnowledgeBaseItem[]>([]);
// 上传按钮文本计算属性
const uploadButtonText = computed(() => {
  switch (uploadMode.value) {
    case 'single':
      return '上传文件';
    case 'multiple':
      return '上传多个文件';
    case 'directory':
      return '上传文件夹';
    default:
      return '上传文件/文件夹';
  }
});
// 监听弹窗显隐：打开时加载文件列表
watch(
  () => props.modelValue,
  async (isOpen) => {
    if (isOpen) {
      await fetchFileList(); // 弹窗打开时初始化列表
    }
  },
  { immediate: false }
);

// 1. 请求文件列表（原有逻辑的 Vue 响应式实现）
const fetchFileList = async () => {
  loading.value = true;
  try {
    const response = await fetch(`${apiUrl.value}ingest/list`);
    if (!response.ok) throw new Error("获取文件列表失败");
    console.info("初始化知识库数据：", response);
    const result = await response.json();

    // 清空原有数据
    knowledgeBaseList.length = 0;
    kg_knowledgeBaseList.length = 0;

    // 添加所有已上传的文件
    if (result.data && Array.isArray(result.data)) {
      knowledgeBaseList = result.data;
      kg_knowledgeBaseList = result.data_kg;
      console.log("知识库rag列表：", result.data);
      console.log("知识库kg_rag列表：", result.data_kg);
      appStore.updateKnowledgeBaseItem([...knowledgeBaseList]);
      appStore.updatekgKnowledgeBaseItem([...kg_knowledgeBaseList]);
    }

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : "初始化知识库失败";
    console.error("初始化知识库失败：", errorMsg);
    // 可选：添加用户提示
    // alert(errorMsg);
  } finally {
    loading.value = false;
  }
};

// 2. 触发文件选择器
const triggerFileInput = () => {
  if (fileInputRef.value) {
    fileInputRef.value.value = ''; // 清空原有选择
    fileInputRef.value.click();
  }
};

// 3. 处理文件选择并上传
const handleFileSelect = async (e: Event) => {
  const target = e.target as HTMLInputElement;
  const files = target.files;

  if (!files || files.length === 0) {
    return;
  }

  uploading.value = true;
  try {
    console.log(`选择 ${files.length} 个文件`);

    // 根据上传模式选择端点和构建 FormData
    const url = `${apiUrl.value}new_update_file/new_update_file`;

    const formData = new FormData();

    Array.from(files).forEach(file => {
        formData.append('files', file);
      });

    // 其他参数
    formData.append('namespace', 'uid_12345');
    formData.append('agent_id', 'uid_12345');
    formData.append('session_id', `session_${Date.now()}`);

    // 发送上传请求
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.data || '上传失败');
    }

    // 根据后端返回的成功/失败信息提示用户
    console.log("上传结果：", result);
    if (result.success) {
      if (result.data && Array.isArray(result.data)) {
        // 成功上传的文件列表
        const uploadedFiles = result.data.map((item: any) => item.file_name).join(', ');
        alert(`上传成功！文件：${uploadedFiles}`);

      }else{
        alert(result.msg);
      }
    } else {
      // 部分文件失败
      let errorMsg = result.data;
      if (result.errors && result.errors.length > 0) {
        errorMsg += '\n失败文件：' + result.errors.map((e: any) => e.filename).join(', ');
      }
      alert(errorMsg);
    }

    // 上传成功后刷新列表
    // await fetchFileList();
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : "文件上传失败";
    console.error("上传失败：", errorMsg);
    alert(errorMsg);
  } finally {
    uploading.value = false;
    // 清空文件选择器
    if (fileInputRef.value) {
      fileInputRef.value.value = '';
    }
  }
};

// 4. 处理编辑（预留：可扩展）
const handleEdit = (item: KnowledgeBaseItem) => {
  console.log('编辑文件：', item);
  // 示例：修改文件名（可对接后端编辑接口）
  const newFileName = prompt('请输入新的文件名', item.doc_metadata.file_name);
  if (newFileName && newFileName.trim() !== '') {
    // 此处可调用编辑接口，成功后更新列表
    item.doc_metadata.file_name = newFileName.trim();
  }
};

// 5. 处理删除（对接后端删除接口）
const handleDelete = async (docId: string, index: number) => {
  if (!confirm('确定要删除该文件吗？')) {
    return;
  }

  try {
    const kg_docId = kg_knowledgeBaseList[index].doc_id;
    const response = await fetch(`${apiUrl.value}ingest/${docId}/${kg_docId}`, {
      method: 'DELETE',
    });

    if (!response.ok) throw new Error("删除文件失败");

    // 删除成功后刷新列表
    await fetchFileList();
    console.log(`文件 ${docId} 删除成功`);
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : "删除文件失败";
    console.error("删除失败：", errorMsg);
    alert(errorMsg);
  }
};

// 关闭弹窗
const closeModal = () => {
  emit('update:modelValue', false);
};
</script>