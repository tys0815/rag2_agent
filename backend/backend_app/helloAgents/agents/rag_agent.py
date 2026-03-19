"""企业级RAG Agent实现 - 支持配置管理、监控和高级功能"""

from typing import Optional, Iterator, Dict, Any
import time
import logging
import os


from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from fastapi import Depends
from helloAgents.tools.builtin.rag_tool import RAGTool
from helloAgents.tools.registry import global_registry, ToolRegistry

# 企业级配置
class RagAgentConfig:
    """RAG Agent企业级配置"""

    def __init__(self):
        # 从环境变量读取配置，提供默认值
        self.rag_tool_name = os.getenv("RAG_TOOL_NAME", "rag")
        self.enable_cache = os.getenv("RAG_ENABLE_CACHE", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("RAG_CACHE_TTL", "300"))
        self.max_input_length = int(os.getenv("RAG_MAX_INPUT_LENGTH", "5000"))
        self.enable_metrics = os.getenv("RAG_ENABLE_METRICS", "true").lower() == "true"
        self.timeout_seconds = int(os.getenv("RAG_TIMEOUT_SECONDS", "30"))
        self.enable_fallback = os.getenv("RAG_ENABLE_FALLBACK", "true").lower() == "true"
        self.fallback_response = os.getenv("RAG_FALLBACK_RESPONSE", "抱歉，我暂时无法处理这个问题。请稍后再试。")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rag_tool_name": self.rag_tool_name,
            "enable_cache": self.enable_cache,
            "cache_ttl": self.cache_ttl,
            "max_input_length": self.max_input_length,
            "enable_metrics": self.enable_metrics,
            "timeout_seconds": self.timeout_seconds,
            "enable_fallback": self.enable_fallback,
            "fallback_response": self.fallback_response
        }

logger = logging.getLogger(__name__)


class RagAgent(Agent):
    """企业级RAG Agent - 支持配置管理、监控、高级功能和故障恢复"""

    DEFAULT_SYSTEM_PROMPT = """你是一个基于知识库的智能助手。请根据提供的上下文信息回答问题。
如果上下文信息不足，请诚实地告知用户你不知道答案，不要编造信息。
请用中文回答，保持专业、准确、友好。"""
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = Depends(global_registry),
        rag_config: Optional[RagAgentConfig] = None
    ):
        """
        初始化企业级RAG Agent

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词（如未提供，使用默认RAG提示词）
            config: Agent基础配置对象
            tool_registry: 工具注册表（可选，如果提供则启用工具调用）
            rag_config: RAG专用配置（如未提供，使用默认配置）
        """
        # 使用默认RAG系统提示词（如果未提供）
        effective_system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        super().__init__(name, llm, effective_system_prompt, config)

        self.tool_registry = tool_registry
        self.rag_config = rag_config or RagAgentConfig()

        # 初始化性能指标
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # 初始化缓存（简单内存缓存，企业级应使用Redis等）
        self._cache = {}

        logger.info(f"RAG Agent '{name}' 初始化完成，配置: {self.rag_config.to_dict()}")
    
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return f"❌ 错误：未配置工具注册表"

        try:
            # 获取Tool对象
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"❌ 错误：未找到工具 '{tool_name}'"

            # 智能参数解析
            param_dict = self._parse_tool_parameters(tool_name, parameters)

            # 调用工具
            result = tool.run(param_dict)
            return f"🔧 工具 {tool_name} 执行结果：\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败：{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        import json
        param_dict = {}

        # 尝试解析JSON格式
        if parameters.strip().startswith('{'):
            try:
                param_dict = json.loads(parameters)
                # JSON解析成功，进行类型转换
                param_dict = self._convert_parameter_types(tool_name, param_dict)
                return param_dict
            except json.JSONDecodeError:
                # JSON解析失败，继续使用其他方式
                pass

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多个参数：action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数：key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()

            # 类型转换
            param_dict = self._convert_parameter_types(tool_name, param_dict)

            # 智能推断action（如果没有指定）
            if 'action' not in param_dict:
                param_dict = self._infer_action(tool_name, param_dict)
        else:
            # 直接传入参数，根据工具类型智能推断
            param_dict = self._infer_simple_parameters(tool_name, parameters)

        return param_dict

    def _convert_parameter_types(self, tool_name: str, param_dict: dict) -> dict:
        """
        根据工具的参数定义转换参数类型

        Args:
            tool_name: 工具名称
            param_dict: 参数字典

        Returns:
            类型转换后的参数字典
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        # 获取工具的参数定义
        try:
            tool_params = tool.get_parameters()
        except:
            return param_dict

        # 创建参数类型映射
        param_types = {}
        for param in tool_params:
            param_types[param.name] = param.type

        # 转换参数类型
        converted_dict = {}
        for key, value in param_dict.items():
            if key in param_types:
                param_type = param_types[key]
                try:
                    if param_type == 'number' or param_type == 'integer':
                        # 转换为数字
                        if isinstance(value, str):
                            converted_dict[key] = float(value) if param_type == 'number' else int(value)
                        else:
                            converted_dict[key] = value
                    elif param_type == 'boolean':
                        # 转换为布尔值
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ('true', '1', 'yes')
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = value
                except (ValueError, TypeError):
                    # 转换失败，保持原值
                    converted_dict[key] = value
            else:
                converted_dict[key] = value

        return converted_dict

    def _infer_action(self, tool_name: str, param_dict: dict) -> dict:
        """根据工具类型和参数推断action"""
        if tool_name == 'memory':
            if 'recall' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('recall')
            elif 'store' in param_dict:
                param_dict['action'] = 'add'
                param_dict['content'] = param_dict.pop('store')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'content' in param_dict:
                param_dict['action'] = 'add'
        elif tool_name == 'rag':
            if 'search' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('search')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'text' in param_dict:
                param_dict['action'] = 'add_text'

        return param_dict

    def _infer_simple_parameters(self, tool_name: str, parameters: str) -> dict:
        """为简单参数推断完整的参数字典"""
        if tool_name == 'rag':
            return {'action': 'search', 'query': parameters}
        elif tool_name == 'memory':
            return {'action': 'search', 'query': parameters}
        else:
            return {'input': parameters}

    def run(self, input_text: str, namespace: str, **kwargs) -> str:
        """
        运行企业级RAG Agent，支持监控、缓存和故障恢复

        Args:
            input_text: 用户输入
            namespace: 知识库命名空间
            **kwargs: 其他参数（如session_id, agent_id等）

        Returns:
            Agent响应
        """
        start_time = time.time()
        request_id = kwargs.get("request_id", "unknown")
        self.metrics["total_requests"] += 1

        logger.info(f"[{request_id}] RAG请求开始 | input_length={len(input_text)} | namespace={namespace}")

        try:
            # 1. 输入验证
            if not input_text or not input_text.strip():
                error_msg = "输入文本不能为空"
                logger.warning(f"[{request_id}] {error_msg}")
                self.metrics["failed_requests"] += 1
                return error_msg

            if len(input_text) > self.rag_config.max_input_length:
                error_msg = f"输入文本过长（{len(input_text)}字符），最大允许{self.rag_config.max_input_length}字符"
                logger.warning(f"[{request_id}] {error_msg}")
                self.metrics["failed_requests"] += 1
                return error_msg

            clean_input = input_text.strip()

            # 2. 检查缓存
            cache_key = None
            if self.rag_config.enable_cache:
                cache_key = self._generate_cache_key(clean_input, namespace, kwargs)
                cached_response = self._get_from_cache(cache_key)
                if cached_response is not None:
                    logger.info(f"[{request_id}] 缓存命中 | cache_key={cache_key}")
                    self.metrics["cache_hits"] += 1
                    response_time = time.time() - start_time
                    self.metrics["total_response_time"] += response_time
                    self.metrics["successful_requests"] += 1
                    return cached_response

            logger.debug(f"[{request_id}] 缓存未命中，执行RAG查询 | cache_key={cache_key}")
            self.metrics["cache_misses"] += 1

            # 3. 获取RAG工具
            if not self.tool_registry:
                error_msg = "未配置工具注册表"
                logger.error(f"[{request_id}] {error_msg}")
                self.metrics["failed_requests"] += 1
                return self._get_fallback_response("tool_registry_missing")

            tool = self.tool_registry.get_tool(self.rag_config.rag_tool_name)
            if not tool:
                error_msg = f"未找到RAG工具: {self.rag_config.rag_tool_name}"
                logger.error(f"[{request_id}] {error_msg}")
                self.metrics["failed_requests"] += 1
                return self._get_fallback_response("tool_not_found")

            # 4. 执行RAG查询（带超时控制）
            try:
                # 在实际企业环境中，这里应该使用异步超时控制
                # 此处为简化实现
                result = tool.ask(clean_input, namespace, **kwargs)
            except Exception as e:
                error_msg = f"RAG工具执行失败: {str(e)}"
                logger.error(f"[{request_id}] {error_msg}", exc_info=True)
                self.metrics["failed_requests"] += 1
                return self._get_fallback_response("tool_execution_error")

            # 5. 保存到缓存
            if self.rag_config.enable_cache and cache_key:
                self._save_to_cache(cache_key, result)

            # 6. 保存到历史记录
            self.add_message(Message(clean_input, "user"))
            self.add_message(Message(result, "assistant"))

            # 7. 更新性能指标
            response_time = time.time() - start_time
            self.metrics["total_response_time"] += response_time
            self.metrics["successful_requests"] += 1

            logger.info(f"[{request_id}] RAG请求成功 | response_time={response_time:.3f}s | response_length={len(result)}")

            # 8. 记录性能指标（企业级可推送到监控系统）
            if self.rag_config.enable_metrics:
                self._record_metrics({
                    "request_id": request_id,
                    "namespace": namespace,
                    "input_length": len(clean_input),
                    "response_length": len(result),
                    "response_time": response_time,
                    "cache_hit": cache_key is not None and cached_response is not None,
                    "success": True
                })

            return result

        except Exception as e:
            # 全局异常处理
            error_msg = f"RAG Agent内部错误: {str(e)}"
            logger.error(f"[{request_id}] {error_msg}", exc_info=True)
            self.metrics["failed_requests"] += 1
            return self._get_fallback_response("internal_error")

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """
        添加工具到Agent（便利方法）

        Args:
            tool: Tool对象
            auto_expand: 是否自动展开可展开的工具（默认True）

        如果工具是可展开的（expandable=True），会自动展开为多个独立工具
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 直接使用 ToolRegistry 的 register_tool 方法
        # ToolRegistry 会自动处理工具展开
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
        return False

    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent
        
        Args:
            input_text: 用户输入
            **kwargs: 其他参数
            
        Yields:
            Agent响应片段
        """
        # 构建消息列表
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": input_text})
        
        # 流式调用LLM
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        
        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))

    # ===== 企业级辅助方法 =====

    def _generate_cache_key(self, input_text: str, namespace: str, kwargs: dict) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{input_text}:{namespace}:{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """从缓存获取结果"""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            # 检查TTL
            if time.time() - entry["timestamp"] < self.rag_config.cache_ttl:
                return entry["value"]
            else:
                # 缓存过期，删除
                del self._cache[cache_key]
        return None

    def _save_to_cache(self, cache_key: str, value: str) -> None:
        """保存结果到缓存"""
        self._cache[cache_key] = {
            "value": value,
            "timestamp": time.time()
        }
        # 简单缓存清理（企业级应使用LRU等策略）
        if len(self._cache) > 1000:  # 最大缓存条目数
            # 删除最旧的条目
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

    def _get_fallback_response(self, error_type: str) -> str:
        """获取降级响应"""
        if self.rag_config.enable_fallback:
            logger.warning(f"使用降级响应，错误类型: {error_type}")
            return self.rag_config.fallback_response
        else:
            return f"❌ 服务暂时不可用（错误类型: {error_type}），请稍后重试"

    def _record_metrics(self, metrics_data: dict) -> None:
        """记录性能指标（企业级可推送到Prometheus、Datadog等）"""
        # 此处为简单实现，仅记录日志
        # 在实际企业环境中，应推送到监控系统
        logger.debug(f"性能指标: {metrics_data}")

    def get_metrics_summary(self) -> dict:
        """获取性能指标摘要"""
        total = self.metrics["total_requests"]
        successful = self.metrics["successful_requests"]
        failed = self.metrics["failed_requests"]
        avg_time = 0.0
        if successful > 0:
            avg_time = self.metrics["total_response_time"] / successful

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_response_time": round(avg_time, 3),
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate": self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 else 0.0
        }

    def reset_metrics(self) -> None:
        """重置性能指标"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("性能指标已重置")
