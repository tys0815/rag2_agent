"""RAG工具 - 检索增强生成

为HelloAgents框架提供简洁易用的RAG能力：
- 🔄 数据流程：用户数据 → 文档解析 → 向量化存储 → 智能检索 → LLM增强问答
- 📚 多格式支持：PDF、Word、Excel、PPT、图片、音频、网页等
- 🧠 智能问答：自动检索相关内容，注入提示词，生成准确答案
- 🏷️ 命名空间：支持多项目隔离，便于管理不同知识库

使用示例：
```python
# 1. 初始化RAG工具
rag = RAGTool()

# 2. 添加文档
rag.run({"action": "add_document", "file_path": "document.pdf"})

# 3. 智能问答
answer = rag.run({"action": "ask", "question": "什么是机器学习？"})
```
"""

from typing import Dict, Any, List, Optional
import os
import time

from ..base import Tool, ToolParameter, tool_action
from ...memory.rag.pipeline import create_rag_pipeline, rerank_with_cross_encoder
from ...memory.storage.load_balancer import LoadBalanceStrategy
from ...core.llm import HelloAgentsLLM
import numpy as np

class RAGTool(Tool):
    """RAG工具
    
    提供完整的 RAG 能力：
    - 添加多格式文档（PDF、Office、图片、音频等）
    - 智能检索与召回
    - LLM 增强问答
    - 知识库管理
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: Optional[str] = None,
        qdrant_urls: Optional[List[str]] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_api_keys: Optional[List[str]] = None,
        collection_name: str = "hello_agents_rag_vectors",
        user_id: str = "default",
        expandable: bool = False,
        load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        cache_size: int = 1000,
        cache_ttl: int = 300,
        redis_config: Optional[Dict[str, Any]] = None,
        redis_ttl: int = 3600,
        **load_balancer_kwargs
    ):
        super().__init__(
            name="rag",
            description="""RAG工具 - 企业内部知识库智能检索工具，用于查询公司制度、考勤、休假、报销、人事政策、开发文档、测试文档、内部流程等非公开信息。
用户询问公司规则、请假流程、报销材料、考勤规定、内部文档时必须使用此工具。""",
            expandable=expandable
        )

        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        self.user_id = user_id

        # 负载均衡配置
        self.qdrant_urls = qdrant_urls or []
        self.qdrant_api_keys = qdrant_api_keys or []
        self.load_balance_strategy = load_balance_strategy
        self.load_balancer_kwargs = load_balancer_kwargs

        # 向后兼容：如果提供了单个URL，转换为列表
        if qdrant_url is not None:
            self.qdrant_urls = [qdrant_url]
            if qdrant_api_key is not None:
                self.qdrant_api_keys = [qdrant_api_key]

        # 如果没有提供URL，尝试环境变量
        if not self.qdrant_urls:
            env_url = os.getenv("QDRANT_URL")
            if env_url:
                self.qdrant_urls = [env_url]
                env_key = os.getenv("QDRANT_API_KEY")
                if env_key:
                    self.qdrant_api_keys = [env_key]

        # 如果还是没有URL，使用本地模式
        if not self.qdrant_urls:
            self.qdrant_urls = [None]  # None表示本地Qdrant
            self.qdrant_api_keys = [None]

        # 确保API密钥数量匹配URL数量
        if not self.qdrant_api_keys:
            self.qdrant_api_keys = [None] * len(self.qdrant_urls)
        elif len(self.qdrant_api_keys) != len(self.qdrant_urls):
            raise ValueError("Qdrant URLs和API密钥数量不匹配")

        self._pipelines: Dict[str, Dict[str, Any]] = {}

        # 查询缓存配置
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl

        # 1. 本地LRU缓存（热点查询）
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_keys: List[str] = []  # 用于LRU顺序

        # 2. 分布式缓存（Redis）- 企业级部署必选
        self.redis_cache = None
        self.redis_ttl = redis_ttl
        if redis_config:
            try:
                import redis
                self.redis_cache = redis.Redis(**redis_config)
                # 测试连接
                self.redis_cache.ping()
                print(f"✅ Redis缓存连接成功，TTL: {redis_ttl}秒")
            except Exception as e:
                print(f"⚠️ Redis缓存连接失败: {e}，将仅使用本地缓存")
                self.redis_cache = None

        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0

        # 确保知识库目录存在
        os.makedirs(knowledge_base_path, exist_ok=True)

        # 初始化组件
        self._init_components()

    def _get_cache_key(self, method: str, **kwargs) -> str:
        """生成缓存键

        Args:
            method: 方法名 ('search' 或 'ask')
            **kwargs: 方法参数

        Returns:
            缓存键字符串
        """
        # 将参数转换为可哈希的字符串
        key_parts = [method]
        for k, v in sorted(kwargs.items()):
            if k not in ['self', 'cls']:
                key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """从缓存获取结果（分层缓存设计）

        Args:
            cache_key: 缓存键

        Returns:
            缓存结果，如果不存在或已过期则返回None
        """
        # 1. 优先查本地LRU缓存
        if cache_key in self.local_cache:
            # 检查是否过期
            if cache_key in self.cache_timestamps:
                timestamp = self.cache_timestamps[cache_key]
                if time.time() - timestamp > self.cache_ttl:
                    # 过期，清理
                    self._remove_from_local_cache(cache_key)
                    self.cache_misses += 1
                    return None

            # 更新LRU顺序
            if cache_key in self.cache_keys:
                self.cache_keys.remove(cache_key)
                self.cache_keys.append(cache_key)

            self.cache_hits += 1
            return self.local_cache[cache_key]

        # 2. 再查Redis缓存
        if self.redis_cache:
            try:
                import json
                redis_result = self.redis_cache.get(cache_key)
                if redis_result:
                    result = json.loads(redis_result)
                    # 回写本地缓存
                    self._set_to_local_cache(cache_key, result)
                    self.cache_hits += 1
                    return result
            except Exception as e:
                print(f"⚠️ Redis缓存读取失败: {e}")

        self.cache_misses += 1
        return None

    def _set_to_local_cache(self, cache_key: str, result: Any) -> None:
        """将结果存入本地LRU缓存

        Args:
            cache_key: 缓存键
            result: 要缓存的结果
        """
        # 如果缓存已满，移除最旧的条目
        if len(self.cache_keys) >= self.cache_size and self.cache_keys:
            oldest_key = self.cache_keys.pop(0)
            self._remove_from_local_cache(oldest_key)

        # 存储结果和时间戳
        self.local_cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
        self.cache_keys.append(cache_key)

    def _remove_from_local_cache(self, cache_key: str) -> None:
        """从本地缓存中移除条目

        Args:
            cache_key: 缓存键
        """
        self.local_cache.pop(cache_key, None)
        self.cache_timestamps.pop(cache_key, None)
        if cache_key in self.cache_keys:
            self.cache_keys.remove(cache_key)

    def _set_cached_result(self, cache_key: str, result: Any) -> None:
        """将结果存入缓存（分层缓存设计）

        Args:
            cache_key: 缓存键
            result: 要缓存的结果
        """
        # 1. 写入本地LRU缓存
        self._set_to_local_cache(cache_key, result)

        # 2. 写入Redis缓存
        if self.redis_cache:
            try:
                import json
                self.redis_cache.setex(
                    cache_key,
                    self.redis_ttl,
                    json.dumps(result, ensure_ascii=False)
                )
            except Exception as e:
                print(f"⚠️ Redis缓存写入失败: {e}")

    def _remove_from_cache(self, cache_key: str) -> None:
        """从缓存中移除条目（向后兼容）

        Args:
            cache_key: 缓存键
        """
        # 调用新的本地缓存移除方法
        self._remove_from_local_cache(cache_key)

        # 同时从Redis缓存中移除（如果存在）
        if self.redis_cache:
            try:
                self.redis_cache.delete(cache_key)
            except Exception as e:
                print(f"⚠️ Redis缓存删除失败: {e}")

    def _clean_expired_cache(self) -> None:
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []

        for cache_key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(cache_key)

        for cache_key in expired_keys:
            self._remove_from_cache(cache_key)

    def _clear_namespace_cache(self, namespace: str) -> None:
        """清理指定命名空间的缓存

        Args:
            namespace: 命名空间名称
        """
        keys_to_remove = []
        for cache_key in self.local_cache.keys():
            # 检查缓存键是否包含该命名空间
            if f"namespace={namespace}" in cache_key:
                keys_to_remove.append(cache_key)

        for cache_key in keys_to_remove:
            self._remove_from_local_cache(cache_key)

        if keys_to_remove:
            print(f"🧹 已清理命名空间 '{namespace}' 的 {len(keys_to_remove)} 个缓存条目")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息

        Returns:
            缓存统计字典
        """
        current_time = time.time()
        expired_count = 0
        namespace_counts = {}

        for cache_key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_count += 1

            # 按命名空间统计
            for ns in ["default", "test", "test_cache", "test_cache2"]:
                if f"namespace={ns}" in cache_key:
                    namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
                    break

        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        # Redis缓存统计
        redis_stats = {}
        if self.redis_cache:
            try:
                redis_info = self.redis_cache.info()
                redis_stats = {
                    "redis_connected": True,
                    "redis_used_memory": redis_info.get("used_memory_human", "unknown"),
                    "redis_keys": redis_info.get("db0", {}).get("keys", 0),
                    "redis_ttl": self.redis_ttl
                }
            except Exception as e:
                redis_stats = {
                    "redis_connected": False,
                    "redis_error": str(e)
                }
        else:
            redis_stats = {"redis_connected": False}

        return {
            "total_entries": len(self.local_cache),
            "cache_size": self.cache_size,
            "cache_ttl": self.cache_ttl,
            "expired_entries": expired_count,
            "namespace_entries": namespace_counts,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "redis": redis_stats,
            "cache_type": "layered_cache",
            "cache_layers": ["local_lru", "redis"] if self.redis_cache else ["local_lru"]
        }

    def reset_cache_stats(self) -> None:
        """重置缓存统计信息"""
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self, user_id: Optional[str] = None) -> None:
        """清理缓存

        Args:
            user_id: 指定用户ID，如果为None则清理所有缓存
        """
        if user_id is None:
            # 清理所有缓存
            self.query_cache.clear()
            self.cache_timestamps.clear()
            self.cache_keys.clear()
            print(f"🧹 已清理所有缓存")
        else:
            # 清理指定命名空间的缓存
            self._clear_namespace_cache(user_id)

    def _init_components(self):
        """初始化RAG组件"""
        try:

            # 单实例模式
            default_pipeline = create_rag_pipeline(
                qdrant_url=self.qdrant_urls[0],
                qdrant_api_key=self.qdrant_api_keys[0],
                collection_name=self.collection_name,
                user_id=self.user_id
            )
            print(f"✅ RAG工具初始化成功（单实例模式）: user_id={self.user_id}, collection={self.collection_name}")

            self._pipelines[self.user_id] = default_pipeline

            # 初始化 LLM 用于回答生成
            self.llm = HelloAgentsLLM()

            self.initialized = True

        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"❌ RAG工具初始化失败: {e}")

    def _get_pipeline(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取指定用户的 RAG 管道，若不存在则自动创建"""
        target_user = user_id or self.user_id
        if target_user in self._pipelines:
            return self._pipelines[target_user]

        
        # 单实例模式
        pipeline = create_rag_pipeline(
            qdrant_url=self.qdrant_urls[0],
            qdrant_api_key=self.qdrant_api_keys[0],
            collection_name=self.collection_name,
            user_id=target_user
        )
       

        self._pipelines[target_user] = pipeline
        return pipeline

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具（非展开模式）

        Args:
            parameters: 工具参数字典，必须包含action参数

        Returns:
            执行结果字符串
        """
        if not self.validate_parameters(parameters):
            return "❌ 参数验证失败：缺少必需的参数"

        if not self.initialized:
            return f"❌ RAG工具未正确初始化，请检查配置: {getattr(self, 'init_error', '未知错误')}"

        action = parameters.get("action")

        # 根据action调用对应的方法，传入提取的参数
        try:
            if action == "add_document":
                return self._add_document(
                    file_path=parameters.get("file_path"),
                    document_id=parameters.get("document_id"),
                    user_id=parameters.get("user_id", "default"),
                    chunk_size=parameters.get("chunk_size", 800),
                    chunk_overlap=parameters.get("chunk_overlap", 100)
                )
            elif action == "add_text":
                return self._add_text(
                    text=parameters.get("text"),
                    document_id=parameters.get("document_id"),
                    namespace=parameters.get("user_id", "default"),
                    chunk_size=parameters.get("chunk_size", 800),
                    chunk_overlap=parameters.get("chunk_overlap", 100)
                )
            elif action == "ask":
                return self._ask(
                    question=parameters.get("question"),
                    limit=parameters.get("limit", 5),
                    enable_advanced_search=parameters.get("enable_advanced_search", True),
                    include_citations=parameters.get("include_citations", True),
                    max_chars=parameters.get("max_chars", 12000),
                    namespace=parameters.get("user_id", "default")
                )
            elif action == "search":
                return self._search(
                    query=parameters.get("query"),
                    limit=parameters.get("limit", 5),
                    min_score=parameters.get("min_score", 0.1),
                    enable_advanced_search=parameters.get("enable_advanced_search", True),
                    max_chars=parameters.get("max_chars", 1200),
                    include_citations=parameters.get("include_citations", True),
                    user_id=parameters.get("user_id", "default")
                )
            elif action == "stats":
                return self._get_stats(namespace=parameters.get("namespace", "default"))
            elif action == "clear":
                return self._clear_knowledge_base(
                    confirm=parameters.get("confirm", False),
                    namespace=parameters.get("namespace", "default")
                )
            else:
                return f"❌ 不支持的操作: {action}"
        except Exception as e:
            return f"❌ 执行操作 '{action}' 时发生错误: {str(e)}"

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义 - Tool基类要求的接口"""
        return [
            # 核心操作参数
            ToolParameter(
                name="action",
                type="string",
                description="操作类型： search(搜索)",
                required=True
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询词（用于基础搜索）",
                required=False
            ),
        ]

    @tool_action("rag_add_document", "添加文档到知识库（支持PDF、Word、Excel、PPT、图片、音频等多种格式）")
    def _add_document(
        self,
        file_path: list[str],
        document_id: str = None,
        user_id: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> str:
        """添加文档到知识库

        Args:
            file_path: 文档文件路径
            document_id: 文档ID（可选）
            namespace: 知识库命名空间（用于隔离不同项目）
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            执行结果
        """
        try:
            
            pipeline = self._get_pipeline(user_id)
            t0 = time.time()

            chunks_added = pipeline["add_documents"](
                file_paths=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                user_id=user_id
            )
            
            t1 = time.time()
            process_ms = int((t1 - t0) * 1000)
            
            if chunks_added == 0:
                return f"⚠️ 未能从文件解析内容: {os.path.basename(file_path)}"
            
            # 添加文档后清理该命名空间的缓存
            self._clear_namespace_cache(user_id)

            return (
                f"✅ 文档已添加到知识库: {', '.join(os.path.basename(p) for p in file_path)}\n"
                f"📊 分块数量: {chunks_added}\n"
                f"⏱️ 处理时间: {process_ms}ms\n"
                f"📝 命名空间: {user_id}"
                f"\n🧹 已清理相关缓存"
            )
            
        except Exception as e:
            return f"❌ 添加文档失败: {str(e)}"
    
    @tool_action("rag_add_text", "添加文本到知识库")
    def _add_text(
        self,
        text: str,
        document_id: str = None,
        namespace: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> str:
        """添加文本到知识库

        Args:
            text: 要添加的文本内容
            document_id: 文档ID（可选）
            namespace: 知识库命名空间
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            执行结果
        """
        metadata = None
        try:
            if not text or not text.strip():
                return "❌ 文本内容不能为空"
            
            # 创建临时文件
            document_id = document_id or f"text_{abs(hash(text)) % 100000}"
            tmp_path = os.path.join(self.knowledge_base_path, f"{document_id}.md")
            
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                pipeline = self._get_pipeline(namespace)
                t0 = time.time()

                chunks_added = pipeline["add_documents"](
                    file_paths=[tmp_path],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                t1 = time.time()
                process_ms = int((t1 - t0) * 1000)
                
                if chunks_added == 0:
                    return f"⚠️ 未能从文本生成有效分块"
                
                # 添加文本后清理该命名空间的缓存
                self._clear_namespace_cache(pipeline.get('namespace', self.rag_namespace))

                return (
                    f"✅ 文本已添加到知识库: {document_id}\n"
                    f"📊 分块数量: {chunks_added}\n"
                    f"⏱️ 处理时间: {process_ms}ms\n"
                    f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}"
                    f"\n🧹 已清理相关缓存"
                )
                
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            
        except Exception as e:
            return f"❌ 添加文本失败: {str(e)}"
    
    @tool_action("rag_search", "搜索知识库中的相关内容")
    def _search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
        enable_advanced_search: bool = True,
        max_chars: int = 1200,
        include_citations: bool = True,
        user_id: str = "default",
        metadata_filters: Optional[List[Dict[str, Any]]] = None,
        enable_rerank: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        enable_hybrid_search: bool = False,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> str:
        """搜索知识库

        Args:
            query: 搜索查询词
            limit: 返回结果数量
            min_score: 最低相关度分数
            enable_advanced_search: 是否启用高级搜索（MQE、HyDE）
            max_chars: 每个结果最大字符数
            include_citations: 是否包含引用来源
            user_id: 知识库命名空间
            metadata_filters: 元数据过滤条件列表，支持复杂查询
                格式示例：
                [
                    {"field": "source_path", "operator": "eq", "value": "document.pdf"},
                    {"field": "lang", "operator": "eq", "value": "zh"},
                    {"field": "start", "operator": "gte", "value": 100},
                    {"field": "end", "operator": "lte", "value": 500},
                    {"field": "file_ext", "operator": "in", "value": [".pdf", ".docx"]}
                ]
                支持的运算符：eq, ne, gt, gte, lt, lte, in, contains
            enable_rerank: 是否启用结果重排序
            reranker_model: 重排序模型名称
            enable_hybrid_search: 是否启用混合搜索（向量+关键词）
            vector_weight: 向量搜索权重（0.0-1.0）
            keyword_weight: 关键词搜索权重（0.0-1.0）

        Returns:
            搜索结果
        """
        try:
            # if not query or not query.strip():
            #     return "❌ 搜索查询不能为空"

            # 生成缓存键
            # cache_key = self._get_cache_key(
            #     "search",
            #     query=query,
            #     limit=limit,
            #     min_score=min_score,
            #     enable_advanced_search=enable_advanced_search,
            #     max_chars=max_chars,
            #     include_citations=include_citations,
            #     user_id=user_id,
            #     metadata_filters=metadata_filters,
            #     enable_rerank=enable_rerank,
            #     reranker_model=reranker_model,
            #     enable_hybrid_search=enable_hybrid_search,
            #     vector_weight=vector_weight,
            #     keyword_weight=keyword_weight
            # )

            # # 检查缓存
            # cached_result = self._get_cached_result(cache_key)
            # if cached_result is not None:
            #     print(f"🔍 缓存命中: {query}")
            #     return cached_result

            # print(f"🔍 缓存未命中，执行搜索: {query}")

            # 使用统一 RAG 管道搜索
            pipeline = self._get_pipeline(user_id)

            if enable_hybrid_search:
                # 混合搜索
                results = pipeline["search_hybrid"](
                    query=query,
                    top_k=limit,
                    score_threshold=min_score if min_score > 0 else None,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    enable_advanced_search=enable_advanced_search
                )
            elif enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=query,
                    top_k=limit,
                    enable_mqe=True,
                    enable_hyde=True,
                    score_threshold=min_score if min_score > 0 else None,
                    user_id=user_id
                )
            else:
                results = pipeline["search"](
                    query=query,
                    top_k=limit,
                    score_threshold=min_score if min_score > 0 else None
                )

            if not results:
                # 缓存空结果
                result = f"🔍 未找到与 '{query}' 相关的内容"
                # self._set_cached_result(cache_key, result)
                return result

            # 应用结果重排序
            if enable_rerank and results:
                # 准备重排序数据
                rerank_items = []
                for res in results:
                    item = res.copy()
                    # 确保有content字段
                    if "content" not in item:
                        item["content"] = item.get("metadata", {}).get("content", "")
                    rerank_items.append(item)
                # 应用重排序
                results = rerank_with_cross_encoder(
                    query=query,
                    items=rerank_items,
                    model_name=reranker_model,
                    top_k=limit
                )

            # 格式化搜索结果
            search_result = ["搜索结果："]
            for i, result in enumerate(results, 1):
                meta = result.get("metadata", {})
                # 获取分数：优先使用hybrid_score（混合搜索），然后是rerank_score（重排序），最后是原始score
                score = result.get("hybrid_score", result.get("rerank_score", result.get("score", 0.0)))
                content = meta.get("content", "")[:200] + "..."
                source = meta.get("source_path", "unknown")

                # 安全处理Unicode
                def clean_text(text):
                    try:
                        return str(text).encode('utf-8', errors='ignore').decode('utf-8')
                    except Exception:
                        return str(text)

                clean_content = clean_text(content)
                clean_source = clean_text(source)

                # 构建分数显示字符串
                score_display = f"相关度: {score:.3f}"
                if enable_hybrid_search:
                    vector_score = result.get("vector_score", 0.0)
                    keyword_score = result.get("keyword_score", 0.0)
                    score_display = f"综合: {score:.3f} (向量: {vector_score:.3f}, 关键词: {keyword_score:.3f})"
                elif enable_rerank:
                    original_score = result.get("score", 0.0)
                    rerank_score = result.get("rerank_score", 0.0)
                    score_display = f"重排序: {rerank_score:.3f} (原始: {original_score:.3f})"

                search_result.append(f"\n{i}. 文档: **{clean_source}** ({score_display})")
                search_result.append(f"   {clean_content}")

                if include_citations and meta.get("heading_path"):
                    clean_heading = clean_text(str(meta['heading_path']))
                    search_result.append(f"   章节: {clean_heading}")
            
            result = "\n".join(search_result)
            # 缓存搜索结果
            # self._set_cached_result(cache_key, result)
            return result
            
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"


    # ===================== 【优化后】核心 _ask 方法（完整流程） =====================
    @tool_action("rag_ask", "基于知识库进行智能问答")
    def _ask(
        self,
        question: str,
        limit: int = 5,
        enable_advanced_search: bool = True,
        include_citations: bool = True,
        max_chars: int = 3000,
        namespace: str = "default",
        enable_rerank: bool = True,
        reranker_model: str = "BAAI/bge-small-reranker-v2.0",
        enable_hybrid_search: bool = False,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        min_relevance_score: float = 0.1,
        llm_timeout: int = 30,
    ) -> str:
        """智能问答完整流程：校验 → 缓存 → 检索 → 重排 → 上下文 → LLM → 格式化 → 缓存写入"""
        try:
            # ============== 1. 前置校验 ==============
            if not question or not question.strip():
                return "❌ 请提供要询问的问题"
            user_question = question.strip()

            # ============== 2. 生成缓存 key ==============
            cache_key = self._get_cache_key(
                "ask",
                question=user_question,
                limit=limit,
                adv_search=enable_advanced_search,
                rerank=enable_rerank,
                hybrid=enable_hybrid_search,
                namespace=namespace,
            )

            # ============== 3. 读取缓存（快速返回）==============
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                print(f"✅ 问答缓存命中: {user_question[:30]}...")
                return cached_result

            # ============== 4. 执行检索（原始结果）==============
            raw_results = self._get_raw_search_results(
                query=user_question,
                namespace=namespace,
                limit=limit,
                min_score=min_relevance_score,
                enable_advanced_search=enable_advanced_search,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                enable_hybrid_search=enable_hybrid_search,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )

            # ============== 5. 无结果处理 ==============
            if not raw_results:
                empty_answer = self._build_no_result_answer(user_question)
                self._set_cached_result(cache_key, empty_answer)
                return empty_answer

            # ============== 6. 构建高质量上下文 ==============
            context, citations = self._build_qualified_context(
                raw_results=raw_results,
                max_chars=max_chars,
                include_citations=include_citations
            )

            # ============== 7. 构建提示词 ==============
            prompt = self._build_rag_prompt(user_question, context)

            # ============== 8. LLM 生成（带超时）==============
            llm_start = time.time()
            try:
                answer = self.llm.invoke(prompt, timeout=llm_timeout)
            except Exception as e:
                return f"❌ LLM 生成失败（超时/异常）: {str(e)}"
            llm_time = int((time.time() - llm_start) * 1000)

            if not answer or not answer.strip():
                return "❌ AI 未能生成有效答案，请更换问题或检查知识库"

            # ============== 9. 格式化最终答案 ==============
            avg_score = np.mean([r.get("score", 0) for r in raw_results])
            final_answer = self._format_answer(
                answer=answer.strip(),
                citations=citations,
                llm_cost=llm_time,
                avg_score=avg_score
            )

            # ============== 10. 写入缓存 ==============
            self._set_cached_result(cache_key, final_answer)
            return final_answer

        except Exception as e:
            return f"❌ 智能问答异常: {str(e)}\n请检查知识库或联系管理员"

    # ===================== 【必须配套】内部工具方法（完整流程依赖） =====================
    def _get_raw_search_results(
        self,
        query: str,
        namespace: str,
        limit: int = 5,
        min_score: float = 0.1,
        enable_advanced_search: bool = True,
        enable_rerank: bool = False,
        reranker_model: str = "",
        enable_hybrid_search: bool = False,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[Dict]:
        """内部方法：获取原始检索结果（不格式化，专供 ask 使用）"""
        try:
            pipeline = self._get_pipeline(namespace)

            # 执行搜索
            if enable_hybrid_search:
                res = pipeline["search_hybrid"](
                    query=query,
                    top_k=limit,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight
                )
            elif enable_advanced_search:
                res = pipeline["search_advanced"](query=query, top_k=limit)
            else:
                res = pipeline["search"](query=query, top_k=limit)

            # 低相关性过滤
            res = [r for r in res if r.get("score", 0) >= min_score]

            # 重排序
            if enable_rerank and res:
                from ...memory.rag.pipeline import rerank_with_cross_encoder
                items = []
                for r in res:
                    content = r.get("metadata", {}).get("content", "")
                    items.append({"content": content})
                res = rerank_with_cross_encoder(query, items, model_name=reranker_model, top_k=limit)
            return res
        except Exception:
            return []

    def _build_qualified_context(self, raw_results: List[Dict], max_chars: int = 3000, include_citations: bool = True):
        """构建：去重 + 清洗 + 长度控制 + 引用列表"""
        context_parts = []
        citations = []
        seen_content = set()

        for idx, res in enumerate(raw_results):
            meta = res.get("metadata", {})
            content = meta.get("content", "").strip()
            if not content or content in seen_content:
                continue
            seen_content.add(content)

            # 文本清洗
            content = " ".join(content.split())
            context_parts.append(f"【参考片段{idx+1}】{content}")

            # 引用来源
            if include_citations:
                citations.append({
                    "idx": idx + 1,
                    "source": os.path.basename(meta.get("source_path", "未知文档")),
                    "score": round(res.get("score", 0), 3)
                })

        # 智能截断
        context = "\n".join(context_parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "...\n⚠️ 上下文已自动裁剪"
        return context, citations

    def _build_rag_prompt(self, question: str, context: str) -> List[Dict[str, str]]:
        """标准 RAG 提示词（严格依据上下文、不编造）"""
        return [
            {
                "role": "system",
                "content": "你是专业知识助手，严格依据提供的参考材料回答，不编造、不扩展、不联想。语言简洁、准确、结构化。无法回答时请明确说明。"
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n参考材料：\n{context}\n\n请直接给出答案。"
            }
        ]

    def _build_no_result_answer(self, question: str) -> str:
        """无结果友好答案"""
        return (
            f"🤖 **智能问答结果**\n\n"
            f"❌ 未在知识库中找到与「{question}」相关的内容。\n\n"
            f"💡 建议：\n"
            f"• 更换关键词重试\n"
            f"• 确认已上传对应文档\n"
            f"• 检查命名空间是否正确"
        )

    def _format_answer(self, answer: str, citations: List[Dict], llm_cost: int, avg_score: float) -> str:
        """最终答案格式化（带引用、耗时、评分）"""
        lines = [f"🤖 **智能问答结果**\n", answer]

        # 参考来源
        if citations:
            lines.append("\n📚 **参考来源**")
            for cit in citations:
                if cit["score"] >= 0.7:
                    emoji = "🟢"
                elif cit["score"] >= 0.5:
                    emoji = "🟡"
                else:
                    emoji = "🔵"
                lines.append(f"{emoji} [{cit['idx']}] {cit['source']} (相似度: {cit['score']:.3f})")

        # 性能信息
        lines.append(f"\n⚡ 生成: {llm_cost}ms | 平均相似度: {avg_score:.3f}")
        return "\n".join(lines)
    
    def _clean_content_for_context(self, content: str) -> str:
        """清理内容用于上下文"""
        # 移除过多的换行和空格
        content = " ".join(content.split())
        # 截断过长内容
        # if len(content) > 300:
        #     content = content[:300] + "..."
        return content
    
    def _smart_truncate_context(self, context: str, max_chars: int) -> str:
        """智能截断上下文，保持段落完整性"""
        if len(context) <= max_chars:
            return context
        
        # 寻找最近的段落分隔符
        truncated = context[:max_chars]
        last_break = truncated.rfind("\n\n")
        
        if last_break > max_chars * 0.7:  # 如果断点位置合理
            return truncated[:last_break] + "\n\n[...更多内容被截断]"
        else:
            return truncated[:max_chars-20] + "...[内容被截断]"
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return (
            "你是一个专业的知识助手，具备以下能力：\n"
            "1. 📖 精准理解：仔细理解用户问题的核心意图\n"
            "2. 🎯 可信回答：严格基于提供的上下文信息回答，不编造内容\n"
            "3. 🔍 信息整合：从多个片段中提取关键信息，形成完整答案\n"
            "4. 💡 清晰表达：用简洁明了的语言回答，适当使用结构化格式\n"
            "5. 🚫 诚实表达：如果上下文不足以回答问题，请坦诚说明\n\n"
            "回答格式要求：\n"
            "• 直接回答核心问题\n"
            "• 必要时使用要点或步骤\n"
            "• 引用关键原文时使用引号\n"
            "• 避免重复和冗余"
        )
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        """构建用户提示词"""
        return (
            f"请基于以下上下文信息回答问题：\n\n"
            f"【问题】{question}\n\n"
            f"【相关上下文】\n{context}\n\n"
            f"【要求】请提供准确、有帮助的回答。如果上下文信息不足，请说明需要什么额外信息。"
        )
    
    def _format_final_answer(self, question: str, answer: str, citations: Optional[List[Dict]] = None, search_time: int = 0, llm_time: int = 0, avg_score: float = 0) -> str:
        """格式化最终答案"""
        result = [f"🤖 **智能问答结果**\n"]
        result.append(answer)
        
        if citations:
            result.append("\n\n📚 **参考来源**")
            for citation in citations:
                score_emoji = "🟢" if citation["score"] > 0.8 else "🟡" if citation["score"] > 0.6 else "🔵"
                result.append(f"{score_emoji} [{citation['index']}] {citation['source']} (相似度: {citation['score']:.3f})")
        
        # 添加性能信息（调试模式）
        result.append(f"\n⚡ 检索: {search_time}ms | 生成: {llm_time}ms | 平均相似度: {avg_score:.3f}")
        
        return "\n".join(result)

    @tool_action("rag_clear", "清空知识库（危险操作，请谨慎使用）")
    def _clear_knowledge_base(self, confirm: bool = False, namespace: str = "default") -> str:
        """清空知识库

        Args:
            confirm: 确认执行（必须设置为True）
            namespace: 知识库命名空间

        Returns:
            执行结果
        """
        try:
            if not confirm:
                return (
                    "⚠️ 危险操作：清空知识库将删除所有数据！\n"
                    "请使用 confirm=true 参数确认执行。"
                )
            
            pipeline = self._get_pipeline(namespace)
            store = pipeline.get("store")
            namespace_id = pipeline.get("namespace", self.rag_namespace)
            success = store.clear_collection() if store else False
            
            if success:
                # 重新初始化该命名空间
                self._pipelines[namespace_id] = create_rag_pipeline(
                    qdrant_url=self.qdrant_url,
                    qdrant_api_key=self.qdrant_api_key,
                    collection_name=self.collection_name,
                    rag_namespace=namespace_id
                )
                # 清空知识库后清理缓存
                self._clear_namespace_cache(namespace_id)
                return f"✅ 知识库已成功清空（命名空间：{namespace_id}）\n🧹 已清理相关缓存"
            else:
                return "❌ 清空知识库失败"
            
        except Exception as e:
            return f"❌ 清空知识库失败: {str(e)}"

    @tool_action("rag_stats", "获取知识库统计信息")
    def _get_stats(self, namespace: str = "default") -> str:
        """获取知识库统计

        Args:
            namespace: 知识库命名空间

        Returns:
            统计信息
        """
        try:
            pipeline = self._get_pipeline(namespace)
            stats = pipeline["get_stats"]()
            
            stats_info = [
                "📊 **RAG 知识库统计**",
                f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}",
                f"📋 集合名称: {self.collection_name}",
                f"📂 存储根路径: {self.knowledge_base_path}"
            ]
            
            # 添加存储统计
            if stats:
                store_type = stats.get("store_type", "unknown")
                total_vectors = (
                    stats.get("points_count") or 
                    stats.get("vectors_count") or 
                    stats.get("count") or 0
                )
                
                stats_info.extend([
                    f"📦 存储类型: {store_type}",
                    f"📊 文档分块数: {int(total_vectors)}",
                ])
                
                if "config" in stats:
                    config = stats["config"]
                    if isinstance(config, dict):
                        vector_size = config.get("vector_size", "unknown")
                        distance = config.get("distance", "unknown")
                        stats_info.extend([
                            f"🔢 向量维度: {vector_size}",
                            f"📎 距离度量: {distance}"
                        ])
            
            # 添加系统状态
            stats_info.extend([
                "",
                "🟢 **系统状态**",
                f"✅ RAG 管道: {'正常' if self.initialized else '异常'}",
                f"✅ LLM 连接: {'正常' if hasattr(self, 'llm') else '异常'}"
            ])
            
            return "\n".join(stats_info)
            
        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"

    def get_relevant_context(self, query: str, limit: int = 3, max_chars: int = 1200, namespace: Optional[str] = None) -> str:
        """为查询获取相关上下文
        
        这个方法可以被Agent调用来获取相关的知识库上下文
        """
        try:
            if not query:
                return ""
            
            # 使用统一 RAG 管道搜索
            pipeline = self._get_pipeline(namespace)
            results = pipeline["search"](
                query=query,
                top_k=limit
            )
            
            if not results:
                return ""
            
            # 合并上下文
            context_parts = []
            for result in results:
                content = result.get("metadata", {}).get("content", "")
                if content:
                    context_parts.append(content)
            
            merged_context = "\n\n".join(context_parts)
            
            # 限制长度
            if len(merged_context) > max_chars:
                merged_context = merged_context[:max_chars] + "..."
            
            return merged_context
            
        except Exception as e:
            return f"获取上下文失败: {str(e)}"
    
    def batch_add_texts(self, texts: List[str], document_ids: Optional[List[str]] = None, chunk_size: int = 800, chunk_overlap: int = 100, namespace: Optional[str] = None) -> str:
        """批量添加文本"""
        try:
            if not texts:
                return "❌ 文本列表不能为空"
            
            if document_ids and len(document_ids) != len(texts):
                return "❌ 文本数量和文档ID数量不匹配"
            
            pipeline = self._get_pipeline(namespace)
            t0 = time.time()
            
            total_chunks = 0
            successful_files = []
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                    
                doc_id = document_ids[i] if document_ids else f"batch_text_{i}"
                tmp_path = os.path.join(self.knowledge_base_path, f"{doc_id}.md")
                
                try:
                    with open(tmp_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    chunks_added = pipeline["add_documents"](
                        file_paths=[tmp_path],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    total_chunks += chunks_added
                    successful_files.append(doc_id)
                    
                finally:
                    # 清理临时文件
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
            
            t1 = time.time()
            process_ms = int((t1 - t0) * 1000)
            
            return (
                f"✅ 批量添加完成\n"
                f"📊 成功文件: {len(successful_files)}/{len(texts)}\n"
                f"📊 总分块数: {total_chunks}\n"
                f"⏱️ 处理时间: {process_ms}ms"
            )
            
        except Exception as e:
            return f"❌ 批量添加失败: {str(e)}"
    
    def clear_all_namespaces(self) -> str:
        """清空当前工具管理的所有命名空间数据"""
        try:
            for ns, pipeline in self._pipelines.items():
                store = pipeline.get("store")
                if store:
                    store.clear_collection()
            self._pipelines.clear()
            # 重新初始化默认命名空间
            self._init_components()
            return "✅ 所有命名空间数据已清空并重新初始化"
        except Exception as e:
            return f"❌ 清空所有命名空间失败: {str(e)}"
    
    # ========================================
    # 便捷接口方法（简化用户调用）
    # ========================================
    
    def add_document(self, file_path: str, namespace: str = "default") -> str:
        """便捷方法：添加单个文档"""
        return self.run({
            "action": "add_document",
            "file_path": file_path,
            "namespace": namespace
        })
    
    def add_text(self, text: str, namespace: str = "default", document_id: str = None) -> str:
        """便捷方法：添加文本内容"""
        return self.run({
            "action": "add_text",
            "text": text,
            "namespace": namespace,
            "document_id": document_id
        })
    
    def ask(self, question: str, namespace: str = "default", **kwargs) -> str:
        """便捷方法：智能问答"""
        params = {
            "action": "ask",
            "question": question,
            "namespace": namespace
        }
        params.update(kwargs)
        return self.run(params)
    
    def search(self, query: str, namespace: str = "default", **kwargs) -> str:
        """便捷方法：搜索知识库"""
        params = {
            "action": "search",
            "query": query,
            "namespace": namespace
        }
        params.update(kwargs)
        return self.run(params)
    
    def add_documents_batch(self, file_paths: List[str], namespace: str = "default") -> str:
        """批量添加多个文档"""
        if not file_paths:
            return "❌ 文件路径列表不能为空"
        
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        start_time = time.time()
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"📄 处理文档 {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            try:
                result = self.add_document(file_path, namespace)
                if "✅" in result:
                    successful += 1
                    # 提取分块数量
                    if "分块数量:" in result:
                        chunks = int(result.split("分块数量: ")[1].split("\n")[0])
                        total_chunks += chunks
                else:
                    failed += 1
                    results.append(f"❌ {os.path.basename(file_path)}: 处理失败")
            except Exception as e:
                failed += 1
                results.append(f"❌ {os.path.basename(file_path)}: {str(e)}")
        
        process_time = int((time.time() - start_time) * 1000)
        
        summary = [
            "📊 **批量处理完成**",
            f"✅ 成功: {successful}/{len(file_paths)} 个文档",
            f"📊 总分块数: {total_chunks}",
            f"⏱️ 总耗时: {process_time}ms",
            f"📝 命名空间: {namespace}"
        ]
        
        if failed > 0:
            summary.append(f"❌ 失败: {failed} 个文档")
            summary.append("\n**失败详情:**")
            summary.extend(results)
        
        return "\n".join(summary)
    
    def add_texts_batch(self, texts: List[str], namespace: str = "default", document_ids: Optional[List[str]] = None) -> str:
        """批量添加多个文本"""
        if not texts:
            return "❌ 文本列表不能为空"
        
        if document_ids and len(document_ids) != len(texts):
            return "❌ 文本数量和文档ID数量不匹配"
        
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        start_time = time.time()
        
        for i, text in enumerate(texts):
            doc_id = document_ids[i] if document_ids else f"batch_text_{i+1}"
            print(f"📝 处理文本 {i+1}/{len(texts)}: {doc_id}")
            
            try:
                result = self.add_text(text, namespace, doc_id)
                if "✅" in result:
                    successful += 1
                    # 提取分块数量
                    if "分块数量:" in result:
                        chunks = int(result.split("分块数量: ")[1].split("\n")[0])
                        total_chunks += chunks
                else:
                    failed += 1
                    results.append(f"❌ {doc_id}: 处理失败")
            except Exception as e:
                failed += 1
                results.append(f"❌ {doc_id}: {str(e)}")
        
        process_time = int((time.time() - start_time) * 1000)
        
        summary = [
            "📊 **批量文本处理完成**",
            f"✅ 成功: {successful}/{len(texts)} 个文本",
            f"📊 总分块数: {total_chunks}",
            f"⏱️ 总耗时: {process_time}ms",
            f"📝 命名空间: {namespace}"
        ]
        
        if failed > 0:
            summary.append(f"❌ 失败: {failed} 个文本")
            summary.append("\n**失败详情:**")
            summary.extend(results)

        return "\n".join(summary)

