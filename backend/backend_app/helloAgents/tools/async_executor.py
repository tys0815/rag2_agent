"""异步工具执行器 - HelloAgents异步工具执行支持"""

import asyncio
import concurrent.futures
from typing import Dict, Any, List
from .registry import ToolRegistry

import time


class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: dict) -> str:
        """异步执行单个工具"""
        loop = asyncio.get_event_loop()

        def _execute():
            return self.registry.execute_tool(tool_name, input_data)
        
        try:
            result = await loop.run_in_executor(self.executor, _execute)
            return result
        except Exception as e:
            return f"❌ 工具 '{tool_name}' 异步执行失败: {e}"


    async def execute_tools_parallel(self, tasks: List[Dict]) -> List[Dict[str, Any]]:
        """
        真正并行执行 + 时间戳打印（判断是否并发）
        """
        print(f"\n🚀 开始并行执行 {len(tasks)} 个任务，当前时间：{time.strftime('%H:%M:%S')}")

        async_tasks = []
        task_list = []

        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            input_data = task.get("input_data", {})
            if not tool_name:
                continue

            # 包装任务，加入【开始执行】打印
            async def wrapped_task(idx, t_name, data):
                # print(f"➡️ 任务 {idx+1} 开始执行 | 工具：{t_name} | 时间：{time.strftime('%H:%M:%S')}")
                res = await self.execute_tool_async(t_name, data)
                # print(f"✅ 任务 {idx+1} 执行完成 | 工具：{t_name} | 时间：{time.strftime('%H:%M:%S')}")
                return res

            async_tasks.append(wrapped_task(i, tool_name, input_data))
            task_list.append(task)

        # ======================
        # 🔥 真正并行：全部同时运行
        # ======================
        results_raw = await asyncio.gather(*async_tasks, return_exceptions=True)

        # print(f"\n🎉 全部并行执行完成 | 结束时间：{time.strftime('%H:%M:%S')}")

        # 组装结果
        results = []
        for idx, (task, result) in enumerate(zip(task_list, results_raw)):
            if isinstance(result, Exception):
                results.append({
                    "task_id": idx,
                    "tool_name": task["tool_name"],
                    "status": "error",
                    "result": str(result)
                })
            else:
                results.append({
                    "task_id": idx,
                    "tool_name": task["tool_name"],
                    "status": "success",
                    "result": result
                })
        return results

    async def execute_tools_batch(self, tool_name: str, input_list: List[str]) -> List[Dict[str, Any]]:
        """
        批量执行同一个工具
        
        Args:
            tool_name: 工具名称
            input_list: 输入数据列表
            
        Returns:
            执行结果列表
        """
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks)

    def close(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        print("🔒 异步工具执行器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
async def run_parallel_tools(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函数：并行执行多个工具
    
    Args:
        registry: 工具注册表
        tasks: 任务列表
        max_workers: 最大工作线程数
        
    Returns:
        执行结果列表
    """
    executor = AsyncToolExecutor(registry, max_workers)
    
    # 下面完全不变
    results = await executor.execute_tools_parallel(tasks)
    return results


async def run_batch_tool(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函数：批量执行同一个工具
    
    Args:
        registry: 工具注册表
        tool_name: 工具名称
        input_list: 输入数据列表
        max_workers: 最大工作线程数
        
    Returns:
        执行结果列表
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_batch(tool_name, input_list)


# 同步包装函数（为了兼容性）
def run_parallel_tools_sync(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本的并行工具执行"""
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers))


def run_batch_tool_sync(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本的批量工具执行"""
    return asyncio.run(run_batch_tool(registry, tool_name, input_list, max_workers))


# 示例函数
async def demo_parallel_execution():
    """演示并行执行的示例"""
    from .registry import ToolRegistry
    
    # 创建注册表（这里假设已经注册了工具）
    registry = ToolRegistry()
    
    # 定义并行任务
    tasks = [
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "3 * 4"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
        {"tool_name": "my_calculator", "input_data": "10 / 2"},
    ]
    
    # 并行执行
    results = await run_parallel_tools(registry, tasks)
    
    # 显示结果
    print("\n📊 并行执行结果:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['tool_name']}({result['input_data']}) = {result['result']}")
    
    return results


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_parallel_execution())
