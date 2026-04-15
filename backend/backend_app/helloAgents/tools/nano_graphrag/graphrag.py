import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from helloAgents.core.llm import HelloAgentsLLM

# ======================== 内部依赖导入 ========================
# LLM 模型相关：OpenAI / Azure / Amazon Bedrock 对话 + 向量嵌入
from ._llm import (
    llm_complete,
    openai_embedding               # OpenAI 嵌入向量

)

# 核心操作模块：分块、实体抽取、社区报告、查询
from ._op import (

    extract_entities,               # 抽取实体 + 关系
    generate_community_report,       # 生成社区报告（图谱摘要）

    local_query,                    # 本地检索（实体+图+向量）
    global_query,                   # 全局检索（全图）
    naive_query,                    # 朴素检索（纯向量）
)

# 存储模块：KV存储、向量库、图存储
from ._storage import (

    NanoVectorDBStorage,            # 轻量级向量库
    Neo4jStorage,                # 知识图谱存储
)

# 工具函数：哈希、限流、事件循环、日志、分词器
from ._utils import (
    EmbeddingFunc,                  # 向量嵌入函数类型

    limit_async_func_call,          # 异步并发限流
    convert_response_to_json,       # LLM 响应转 JSON
    always_get_an_event_loop,       # 获取/创建异步事件循环
    logger,                         # 日志工具

)

# 基础抽象类：定义存储/图谱/查询接口规范
from .base import (
    BaseGraphStorage,               # 图存储基类
    BaseKVStorage,                  # KV 存储基类
    BaseVectorStorage,              # 向量存储基类
    QueryParam,                     # 查询参数类
)

# ======================== GraphRAG 核心类 ========================
@dataclass
class GraphRAG:
    """
    【企业级 GraphRAG 核心引擎】
    功能：实现 文本分块 → 实体抽取 → 知识图谱构建 → 社区聚类 → 社区报告 → 向量+图双检索 → 问答生成
    模式：local(双检索) / global(全局图) / naive(纯向量)
    兼容：OpenAI / Azure OpenAI / Amazon Bedrock
    """

    # --------------------- 1. 基础工作目录配置 ---------------------
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )  # 自动生成带时间戳的缓存目录，存储所有数据

    # --------------------- 2. 检索模式开关 ---------------------
    enable_local: bool = True        # 启用本地模式（实体+图谱+向量 双检索，默认开启）
    enable_naive_rag: bool = False   # 启用朴素RAG（纯向量检索，默认关闭）

    # --------------------- 4. 实体抽取配置 ---------------------
    entity_extract_max_gleaning: int = 1  # 实体抽取重试次数
    entity_summary_to_max_tokens: int = 500  # 实体摘要最大长度

    # --------------------- 5. 图聚类（社区发现）配置 ---------------------
    graph_cluster_algorithm: str = "leiden"  # 聚类算法：leiden（效果最好）
    max_graph_cluster_size: int = 10        # 最大社区大小
    graph_cluster_seed: int = 0xDEADBEEF    # 随机种子（保证结果可复现）


    # --------------------- 7. 社区报告 LLM 参数 ---------------------
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )  # 强制 LLM 返回 JSON 格式，保证报告结构化

    # --------------------- 8. 向量嵌入模型配置 ---------------------
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32          # 向量批处理大小
    embedding_func_max_async: int = 16     # 向量异步最大并发
    query_better_than_threshold: float = 0.2  # 检索相似度阈值


    # 高性能模型（用于实体抽取、报告生成、最终回答）
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768    # 最大上下文
    best_model_max_async: int = 16            # 最大并发

    # --------------------- 10. 实体抽取函数 ---------------------
    entity_extraction_func: callable = extract_entities

    # --------------------- 11. 存储层配置（可替换为企业级数据库） ---------------------
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage  # KV 存储
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage    # 向量库
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = Neo4jStorage             # 图存储
    enable_llm_cache: bool = True  # 开启 LLM 响应缓存（极大减少重复调用）

    # --------------------- 12. 扩展配置 ---------------------
    always_create_working_dir: bool = True  # 自动创建工作目录
    addon_params: dict = field(default_factory=dict)  # 自定义扩展参数
    convert_response_to_json_func: callable = convert_response_to_json  # JSON 转换

    # ======================== 初始化方法：类创建后自动执行 ========================
    def __post_init__(self):
        """
        初始化核心逻辑：
        1. 打印配置日志
        2. 初始化分词器
        3. 自动切换模型厂商（OpenAI/Azure/Bedrock）
        4. 创建工作目录
        5. 初始化所有存储组件
        6. 给 LLM/向量函数添加限流
        """
        # 打印初始化参数
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        # --------------------- 自动切换为 Azure OpenAI ---------------------
        self.best_model_func : None
        self.embedding_func : None

        # 自动创建工作目录
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # ======================== 初始化所有存储实例 ========================
        # 社区报告存储（图谱摘要）
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )

        # 实体关系知识图谱存储（核心！）
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        # 给向量函数添加并发限流，防止超配额
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        # 实体向量库（用于实体检索，双检索核心）
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )

        # 文本块向量库（纯向量检索用）
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        # 给 LLM 函数添加限流 + 缓存
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func)
        )


    # ======================== 对外接口：同步插入文档 ========================
    def insert(self, inserting_chunks:List[any]):
        """同步接口：插入文本/文本列表，自动去重、分块、建图、生成报告"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(inserting_chunks))

    # ======================== 对外接口：同步查询 ========================
    def query(self, query: str, param: QueryParam = QueryParam()):
        """同步接口：提问，自动选择检索模式返回答案"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    # ======================== 异步查询核心 ========================
    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步查询：支持三种模式
        1. local：本地双检索（向量+图+社区报告，企业首选）
        2. global：全局图谱检索（宏观总结）
        3. naive：纯向量检索（传统RAG）
        """
        # 模式校验
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        
        # 路由到对应检索函数
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,  # 知识图谱
                self.entities_vdb,                 # 实体向量库
                self.community_reports,            # 社区报告
                self.text_chunks,                  # 文本分块
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,    # 纯文本向量库
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        
        await self._query_done()
        return response

    # ======================== 异步插入核心 ========================
    async def ainsert(self, inserting_chunks:List[any]):
        """
        完整入库流程（企业级全 pipeline）：
        1. 文档去重 → 2. 文本分块 → 3. 实体抽取 → 4. 构建图谱 → 5. 图聚类 → 6. 生成社区报告 → 7. 持久化
        """
        try:

            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            #  naive RAG 模式：插入向量库
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # 目前不支持增量社区更新，清空旧报告
            await self.community_reports.drop()

            # --------------------- 3. 实体抽取 + 构建知识图谱 ---------------------
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                tokenizer_wrapper=self.tokenizer_wrapper,
                global_config=asdict(self),
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            # --------------------- 4. 图聚类 + 生成社区报告 ---------------------
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, self.tokenizer_wrapper, asdict(self)
            )


        finally:
            pass
