import { useAppStore } from '../stores/appStore';
import { storeToRefs } from 'pinia';
import { type CodeValidateResult } from '../types/index';

interface UseCodeReturn {
    validateCode: (code?: string) => Promise<CodeValidateResult>;
}

export default function (): UseCodeReturn {
    const appStore = useAppStore();
    const { apiUrl } = storeToRefs(appStore);

    
    async function validateCode(code:string=appStore.code): Promise<CodeValidateResult> {
        try {
            const requestBody = {
                code: code
            };
            const response = await fetch(`${apiUrl.value}meta/code`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
          });
          console.error("验证激活码请求：", response.ok,"==============", response.body);
          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `API 响应错误: ${response.status} ${response.statusText}`);
          }

          const result: CodeValidateResult = await response.json();
          console.log("激活码验证成功，返回数据：", result);
          return result;
        } catch (error) {
            const errMsg = error instanceof Error ? error.message : "未知错误";
            console.error("验证激活码请求出错：", errMsg);
            // 可在这里添加用户提示（比如弹窗）
            // ElMessage.error(`激活码验证失败：${errMsg}`); // 如果用 Element Plus 等组件库
            throw error; // 抛出错误，让调用方也能捕获
        }
        
    }
    return {validateCode}
}