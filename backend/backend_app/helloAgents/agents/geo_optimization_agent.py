"""GEO优化助手
根据输入的产品生成软文，并针对多平台进行GEO优化
核心原理：
1. 内容结构化：通过Schema标记、实体关系梳理，让AI模型更易解析内容。
2. 权威信号构建：在内容中融入可信来源（政策、数据、专家观点）提升采信概率。
3. 多平台适配：同时针对搜索引擎（如百度、Google）与生成式AI平台（ChatGPT、豆包、DeepSeek）进行优化。
4. 持续迭代：通过实时监测平台反馈数据，动态调整关键词与语义布局。
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry, global_registry
from ..tools.builtin.memory_tool import MemoryTool
from ..tools.builtin.protocol_tools import MCPTool


class GeoOptimizationAgent(Agent):
    """GEO优化助手

    特点：
    1. 产品信息提取：从用户输入中提取产品名称、特点、目标受众等关键信息
    2. 软文生成：基于提取的信息生成高质量软文
    3. GEO优化：应用内容结构化、权威信号构建、多平台适配等GEO优化技术
    4. 多平台适配：生成适合搜索引擎和生成式AI平台的内容版本
    5. 记忆管理：自动记录对话历史，检索相关记忆
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = None,
        default_agent_id: str = "geo_optimizer",
        mcp_server_command: Optional[List[str]] = None,
        mcp_server_args: Optional[List[str]] = None,
        mcp_env: Optional[Dict[str, str]] = None
    ):
        """
        初始化GEO优化助手

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 工具注册表（默认使用全局注册表）
            default_agent_id: 默认助手ID
            mcp_server_command: MCP服务器启动命令
            mcp_server_args: MCP服务器参数
            mcp_env: MCP服务器环境变量
        """
        super().__init__(name, llm, system_prompt, config)

        # 使用全局工具注册表或传入的注册表
        self.tool_registry = tool_registry or global_registry

        # 默认助手ID
        self.default_agent_id = default_agent_id

        # 获取记忆工具
        self.memory_tool: Optional[MemoryTool] = None
        if self.tool_registry:
            self.memory_tool = self.tool_registry.get_tool("memory")

        # 初始化MCP工具（可选，用于获取行业数据、权威来源等）
        self.mcp_tool: Optional[MCPTool] = None
        self._init_mcp_tool(mcp_server_command, mcp_server_args, mcp_env)

    def _init_mcp_tool(
        self,
        server_command: Optional[List[str]] = None,
        server_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """初始化MCP工具"""
        try:
            # 如果没有指定MCP服务器，使用内置演示服务器
            if not server_command:
                print("[MCP] 使用内置MCP演示服务器（可替换为行业数据MCP服务器）")
                self.mcp_tool = MCPTool(
                    name="geo_mcp",
                    description="GEO优化MCP工具，提供行业数据、权威来源等信息查询",
                    auto_expand=True
                )
            else:
                print(f"[MCP] 连接到外部MCP服务器: {' '.join(server_command)}")
                self.mcp_tool = MCPTool(
                    name="geo_mcp",
                    description="GEO优化MCP工具，提供行业数据、权威来源等信息查询",
                    server_command=server_command,
                    server_args=server_args,
                    env=env,
                    auto_expand=True
                )

            # 如果启用了工具注册表，注册MCP工具
            if self.tool_registry and self.mcp_tool:
                self.tool_registry.register_tool(self.mcp_tool, auto_expand=True)
                print("[OK] MCP工具已注册到工具注册表")

        except Exception as e:
            print(f"[WARN] MCP工具初始化失败: {e}")
            self.mcp_tool = None

    def run(
        self,
        input_text: str,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_memory: bool = True,
        max_context_length: int = 2000,
        **kwargs
    ) -> str:
        """
        运行GEO优化助手

        Args:
            input_text: 用户输入（产品描述或需求）
            user_id: 用户ID（必需）
            agent_id: 助手ID（默认为geo_optimizer）
            session_id: 会话ID（可选，不传则自动生成）
            enable_memory: 是否启用记忆
            max_context_length: 最大上下文长度
            **kwargs: 其他参数

        Returns:
            生成的GEO优化软文
        """
        # 参数处理
        agent_id = agent_id or self.default_agent_id
        session_id = session_id or self._generate_session_id()

        print(f"[GEO] GEO优化助手启动: user={user_id}, agent={agent_id}, session={session_id}")

        # 1. 记录用户输入到工作记忆
        if enable_memory and self.memory_tool:
            self._record_to_memory(
                content=f"用户产品需求: {input_text}",
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type="working"
            )

        # 2. 提取产品关键信息
        extracted_info = self._extract_product_info(input_text)
        print(f"[Info] 提取的产品信息: {extracted_info}")

        # 3. 检索相关记忆
        memory_context = ""
        if enable_memory and self.memory_tool:
            memory_context = self._retrieve_relevant_memories(
                query=input_text,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                limit=5
            )

        # 4. 调用MCP服务获取行业数据、权威来源等（可选）
        industry_data = self._call_mcp_services(extracted_info)
        print(f"[MCP] 行业数据: {industry_data}")

        # 5. 构建增强上下文
        enhanced_context = self._build_enhanced_context(
            user_input=input_text,
            extracted_info=extracted_info,
            memory_context=memory_context,
            industry_data=industry_data,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id
        )

        # 6. 构建消息列表
        messages = self._build_messages(
            input_text=input_text,
            enhanced_context=enhanced_context,
            max_context_length=max_context_length
        )

        # 7. 调用LLM生成GEO优化软文
        try:
            response = self.llm.invoke(messages, **kwargs)
        except Exception as e:
            response = f"❌ 生成GEO优化软文时出错: {str(e)}"

        # 8. 记录助手回答到工作记忆
        if enable_memory and self.memory_tool:
            self._record_to_memory(
                content=f"GEO优化软文: {response}",
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type="working"
            )

        # 9. 保存到Agent历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

        return response

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        import uuid
        return f"geo_ses_{uuid.uuid4().hex[:16]}"

    def _record_to_memory(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        memory_type: str = "working",
        importance: float = 0.5
    ):
        """记录内容到记忆"""
        if not self.memory_tool:
            return

        try:
            self.memory_tool.run({
                "action": "add",
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id
            })
        except Exception as e:
            print(f"[WARN] 记录到记忆失败: {e}")

    def _retrieve_relevant_memories(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        limit: int = 5
    ) -> str:
        """检索相关记忆"""
        if not self.memory_tool:
            return ""

        try:
            result = self.memory_tool.run({
                "action": "search",
                "query": query,
                "limit": limit,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id
            })

            # 解析记忆结果
            if "未找到" in result or "❌" in result:
                return ""

            return result
        except Exception as e:
            print(f"[WARN] 检索记忆失败: {e}")
            return ""

    def _extract_product_info(self, text: str) -> Dict[str, Any]:
        """使用function calling提取产品关键信息"""
        # 定义产品信息提取的function schema
        product_info_schema = {
            "type": "function",
            "function": {
                "name": "extract_product_info",
                "description": "从用户输入中提取产品相关信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "产品名称"
                        },
                        "product_category": {
                            "type": "string",
                            "description": "产品类别，如'科技产品'、'消费品'、'服务'等"
                        },
                        "key_features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "产品主要特点/卖点"
                        },
                        "target_audience": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "目标受众，如['年轻人', '企业主', '家庭用户']"
                        },
                        "price_range": {
                            "type": "string",
                            "description": "价格区间，如'高端'、'中端'、'经济型'"
                        },
                        "competitors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "主要竞争对手"
                        },
                        "unique_selling_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "独特卖点"
                        },
                        "industry_trends": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "相关行业趋势"
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "相关关键词"
                        }
                    },
                    "required": []
                }
            }
        }

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "你是一个产品信息提取助手。请从用户输入中提取产品相关信息，包括产品名称、类别、特点、目标受众等。如果某些信息未明确提及，请设为null或空数组。"
            },
            {
                "role": "user",
                "content": text
            }
        ]

        try:
            # 调用LLM的function calling功能
            client = getattr(self.llm, "_client", None)
            if client is None:
                raise RuntimeError("LLM客户端未初始化")

            response = client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                tools=[product_info_schema],
                tool_choice={"type": "function", "function": {"name": "extract_product_info"}},
                temperature=0.1  # 低温度以获得更确定的结果
            )

            # 解析响应
            choice = response.choices[0]
            message = choice.message

            if message.tool_calls and len(message.tool_calls) > 0:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "extract_product_info":
                    import json
                    arguments = json.loads(tool_call.function.arguments)

                    # 确保所有字段都有默认值
                    info = {
                        "product_name": arguments.get("product_name"),
                        "product_category": arguments.get("product_category"),
                        "key_features": arguments.get("key_features", []),
                        "target_audience": arguments.get("target_audience", []),
                        "price_range": arguments.get("price_range"),
                        "competitors": arguments.get("competitors", []),
                        "unique_selling_points": arguments.get("unique_selling_points", []),
                        "industry_trends": arguments.get("industry_trends", []),
                        "keywords": arguments.get("keywords", [])
                    }

                    # 如果没有检测到产品名称，使用占位符
                    if not info["product_name"]:
                        info["product_name"] = "未指定产品"

                    print(f"[Info] 通过function calling提取的产品信息: {info}")
                    return info

        except Exception as e:
            print(f"[WARN] 使用function calling提取产品信息失败: {e}")
            # 失败时回退到简单规则提取
            return self._fallback_extract_product_info(text)

        # 如果没有提取到信息，使用回退方法
        return self._fallback_extract_product_info(text)

    def _fallback_extract_product_info(self, text: str) -> Dict[str, Any]:
        """回退方法：使用简单规则提取产品信息"""
        info = {
            "product_name": "未指定产品",
            "product_category": None,
            "key_features": [],
            "target_audience": [],
            "price_range": None,
            "competitors": [],
            "unique_selling_points": [],
            "industry_trends": [],
            "keywords": []
        }

        # 简单关键词匹配
        import re

        # 提取产品名称（简单规则：提取引号内的内容或第一个名词短语）
        name_match = re.search(r'["「](.+?)["」]', text)
        if name_match:
            info["product_name"] = name_match.group(1)
        else:
            # 尝试提取可能的产品名称
            words = text.split()
            if len(words) > 0:
                # 简单假设第一个名词短语是产品名称
                info["product_name"] = words[0]

        # 产品类别识别
        category_keywords = {
            "科技产品": ["手机", "电脑", "软件", "应用", "APP", "智能", "数码", "电子"],
            "消费品": ["食品", "饮料", "服装", "化妆品", "日用品", "家居"],
            "服务": ["咨询", "培训", "教育", "医疗", "旅游", "金融", "保险"],
            "工业品": ["设备", "机械", "原材料", "零部件", "工具"],
            "文化产品": ["书籍", "电影", "音乐", "游戏", "艺术品"]
        }

        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    info["product_category"] = category
                    break
            if info["product_category"]:
                break

        # 提取关键词（简单提取名词）
        # 这里使用简单的分词，实际应用可以使用更复杂的方法
        nouns = re.findall(r'\b[\u4e00-\u9fa5]{2,5}\b', text)
        info["keywords"] = nouns[:10]  # 限制前10个

        return info

    def _call_mcp_services(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP服务获取行业数据、权威来源等"""
        industry_data = {
            "industry_stats": {},
            "authoritative_sources": [],
            "trend_data": [],
            "competitor_analysis": []
        }

        if not self.mcp_tool:
            print("[WARN] MCP工具未初始化，返回模拟数据")
            # 返回模拟数据用于演示
            return self._get_mock_industry_data(product_info)

        try:
            # 获取MCP工具列表
            tools_result = self.mcp_tool.run({"action": "list_tools"})
            print(f"[MCP] 可用的MCP工具: {tools_result}")

            # 根据产品信息调用相应的MCP工具
            product_category = product_info.get("product_category")
            if product_category:
                # 尝试调用行业数据查询工具
                try:
                    industry_stats = self.mcp_tool.run({
                        "action": "call_tool",
                        "tool_name": "get_industry_stats",
                        "arguments": {"category": product_category}
                    })
                    industry_data["industry_stats"] = industry_stats
                except:
                    # 如果工具不存在，使用模拟数据
                    industry_data["industry_stats"] = {"category": product_category, "growth_rate": "15%", "market_size": "千亿级"}

                # 尝试调用权威来源查询工具
                try:
                    sources_result = self.mcp_tool.run({
                        "action": "call_tool",
                        "tool_name": "search_authoritative_sources",
                        "arguments": {"topic": product_category}
                    })
                    industry_data["authoritative_sources"] = sources_result
                except:
                    industry_data["authoritative_sources"] = [
                        {"title": f"{product_category}行业白皮书", "url": "https://example.com/whitepaper"},
                        {"title": f"{product_category}政策文件", "url": "https://example.com/policy"}
                    ]

        except Exception as e:
            print(f"⚠️ 调用MCP服务失败: {e}")
            # 返回模拟数据
            industry_data = self._get_mock_industry_data(product_info)

        return industry_data

    def _get_mock_industry_data(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """获取模拟的行业数据（用于演示）"""
        product_category = product_info.get("product_category", "通用产品")
        product_name = product_info.get("product_name", "未知产品")

        return {
            "industry_stats": {
                "category": product_category,
                "market_size": "预计2025年达到5000亿元",
                "growth_rate": "年均增长率15%",
                "consumer_demand": "持续增长",
                "competition_intensity": "中等"
            },
            "authoritative_sources": [
                {"title": f"{product_category}行业发展报告2024", "source": "国家统计局", "url": "https://example.com/report1"},
                {"title": f"{product_category}技术创新白皮书", "source": "行业研究院", "url": "https://example.com/report2"},
                {"title": f"{product_category}消费趋势分析", "source": "市场研究机构", "url": "https://example.com/report3"}
            ],
            "trend_data": [
                {"trend": "智能化", "description": "产品智能化成为主要发展方向"},
                {"trend": "个性化", "description": "消费者对个性化定制需求增加"},
                {"trend": "绿色环保", "description": "环保和可持续发展成为重要考量"}
            ],
            "competitor_analysis": [
                {"name": "竞品A", "strengths": ["品牌知名度高", "渠道广泛"], "weaknesses": ["创新不足", "价格偏高"]},
                {"name": "竞品B", "strengths": ["技术领先", "用户体验好"], "weaknesses": ["市场覆盖有限", "服务网络不足"]}
            ]
        }

    def _build_enhanced_context(
        self,
        user_input: str,
        extracted_info: Dict[str, Any],
        memory_context: str,
        industry_data: Dict[str, Any],
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> str:
        """构建增强上下文"""
        context_parts = []

        # 系统信息
        context_parts.append(f"# GEO优化助手上下文")
        context_parts.append(f"- 用户: {user_id}")
        context_parts.append(f"- 助手: {agent_id}")
        context_parts.append(f"- 会话: {session_id}")
        context_parts.append(f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_parts.append("")

        # 用户问题
        context_parts.append(f"## 用户产品需求")
        context_parts.append(f"{user_input}")
        context_parts.append("")

        # 提取的产品信息
        context_parts.append(f"## 提取的关键信息")
        for key, value in extracted_info.items():
            if value:
                if isinstance(value, list):
                    context_parts.append(f"- {key}: {', '.join(str(v) for v in value)}")
                else:
                    context_parts.append(f"- {key}: {value}")
        context_parts.append("")

        # 相关记忆
        if memory_context:
            context_parts.append(f"## 相关记忆")
            context_parts.append(f"{memory_context}")
            context_parts.append("")

        # 行业数据
        context_parts.append(f"## 行业数据与权威来源")

        # 行业统计
        if industry_data.get("industry_stats"):
            stats = industry_data["industry_stats"]
            context_parts.append(f"### 行业统计")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    context_parts.append(f"- {key}: {value}")
            else:
                context_parts.append(f"{stats}")
            context_parts.append("")

        # 权威来源
        if industry_data.get("authoritative_sources"):
            context_parts.append(f"### 权威来源")
            sources = industry_data["authoritative_sources"]
            if isinstance(sources, list):
                for i, source in enumerate(sources[:3], 1):
                    if isinstance(source, dict):
                        title = source.get("title", f"来源{i}")
                        source_name = source.get("source", "未知来源")
                        url = source.get("url", "")
                        context_parts.append(f"{i}. {title} ({source_name}) {url}")
                    else:
                        context_parts.append(f"{i}. {source}")
            else:
                context_parts.append(f"{sources}")
            context_parts.append("")

        # 趋势数据
        if industry_data.get("trend_data"):
            context_parts.append(f"### 行业趋势")
            trends = industry_data["trend_data"]
            if isinstance(trends, list):
                for trend in trends[:5]:
                    if isinstance(trend, dict):
                        trend_name = trend.get("trend", "趋势")
                        description = trend.get("description", "")
                        context_parts.append(f"- {trend_name}: {description}")
                    else:
                        context_parts.append(f"- {trend}")
            else:
                context_parts.append(f"{trends}")
            context_parts.append("")

        # 竞争对手分析
        if industry_data.get("competitor_analysis"):
            context_parts.append(f"### 竞争对手分析")
            competitors = industry_data["competitor_analysis"]
            if isinstance(competitors, list):
                for i, competitor in enumerate(competitors[:3], 1):
                    if isinstance(competitor, dict):
                        name = competitor.get("name", f"竞品{i}")
                        strengths = competitor.get("strengths", [])
                        weaknesses = competitor.get("weaknesses", [])
                        context_parts.append(f"{i}. {name}")
                        if strengths:
                            context_parts.append(f"   优势: {', '.join(strengths)}")
                        if weaknesses:
                            context_parts.append(f"   劣势: {', '.join(weaknesses)}")
                    else:
                        context_parts.append(f"{i}. {competitor}")
            else:
                context_parts.append(f"{competitors}")
            context_parts.append("")

        # GEO优化要求
        context_parts.append(f"## GEO优化要求")
        context_parts.append("请基于以上信息，生成一篇高质量的GEO优化软文，要求如下：")
        context_parts.append("")
        context_parts.append("### 1. 内容结构化")
        context_parts.append("- 使用清晰的标题层级（H1, H2, H3）")
        context_parts.append("- 段落分明，逻辑清晰")
        context_parts.append("- 包含产品介绍、特点分析、使用场景、用户评价等模块")
        context_parts.append("- 使用列表、要点等格式化元素")
        context_parts.append("")
        context_parts.append("### 2. 权威信号构建")
        context_parts.append("- 引用行业数据、研究报告等权威来源")
        context_parts.append("- 融入专家观点、用户案例等可信内容")
        context_parts.append("- 使用具体的数据和事实支持观点")
        context_parts.append("")
        context_parts.append("### 3. 多平台适配")
        context_parts.append("#### 搜索引擎优化（SEO）")
        context_parts.append("- 合理布局关键词，自然融入内容")
        context_parts.append("- 优化元描述（在内容开头提供简洁的摘要）")
        context_parts.append("- 使用内部链接和外部引用")
        context_parts.append("")
        context_parts.append("#### 生成式AI平台优化（AIO）")
        context_parts.append("- 内容易于AI解析和理解")
        context_parts.append("- 实体关系清晰，语义明确")
        context_parts.append("- 提供结构化数据暗示")
        context_parts.append("")
        context_parts.append("### 4. 软文要素")
        context_parts.append("- 吸引人的标题")
        context_parts.append("- 引人入胜的开头")
        context_parts.append("- 详细的产品介绍和优势分析")
        context_parts.append("- 用户案例或应用场景")
        context_parts.append("- 行动号召（CTA）")
        context_parts.append("")
        context_parts.append("### 5. 输出格式")
        context_parts.append("- 使用Markdown格式")
        context_parts.append("- 包含标题、副标题、正文、列表等")
        context_parts.append("- 长度适中（800-1500字）")
        context_parts.append("- 语言：中文")

        return "\n".join(context_parts)

    def _build_messages(
        self,
        input_text: str,
        enhanced_context: str,
        max_context_length: int = 2000
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []

        # 系统提示词
        system_prompt = self.system_prompt or self._get_default_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 历史消息（限制长度）
        history_context = self._get_recent_history(max_context_length // 2)
        if history_context:
            messages.append({"role": "system", "content": f"## 对话历史\n{history_context}"})

        # 增强上下文
        if enhanced_context:
            # 截断上下文以避免超出token限制
            if len(enhanced_context) > max_context_length:
                enhanced_context = enhanced_context[:max_context_length] + "...[内容被截断]"
            messages.append({"role": "user", "content": enhanced_context})
        else:
            # 如果没有增强上下文，直接使用用户输入
            messages.append({"role": "user", "content": input_text})

        return messages

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个专业的GEO优化助手，专门为产品生成高质量的软文并进行多平台优化。

你具备以下核心能力：
1. 🏗️ 内容结构化：通过Schema标记、实体关系梳理，让AI模型更易解析内容
2. 🏛️ 权威信号构建：在内容中融入可信来源（政策、数据、专家观点）提升采信概率
3. 🌐 多平台适配：同时针对搜索引擎（如百度、Google）与生成式AI平台（ChatGPT、豆包、DeepSeek）进行优化
4. 🔄 持续迭代：通过实时监测平台反馈数据，动态调整关键词与语义布局

具体技能：
1. 产品分析：深入理解产品特点、目标受众和市场定位
2. 软文创作：撰写吸引人、有说服力的营销内容
3. SEO优化：合理布局关键词，优化内容结构
4. AI优化：使内容易于AI解析和推荐
5. 数据驱动：基于行业数据和用户反馈优化内容

请基于提供的上下文信息，生成专业、高质量、经过GEO优化的软文。"""

    def _get_recent_history(self, max_length: int = 1000) -> str:
        """获取最近的对话历史"""
        if not self._history:
            return ""

        history_parts = []
        current_length = 0

        # 从新到旧遍历历史
        for msg in reversed(self._history):
            content = f"{msg.role}: {msg.content}"
            if current_length + len(content) > max_length:
                break
            history_parts.append(content)
            current_length += len(content)

        # 反转回时间顺序
        history_parts.reverse()
        return "\n".join(history_parts) if history_parts else ""

    def get_memory_stats(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取记忆统计"""
        if not self.memory_tool:
            return {"error": "记忆工具未初始化"}

        try:
            result = self.memory_tool.run({
                "action": "stats",
                "user_id": user_id,
                "agent_id": agent_id or self.default_agent_id,
                "session_id": session_id
            })
            return {"success": True, "stats": result}
        except Exception as e:
            return {"error": str(e)}

    def clear_memories(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """清空记忆"""
        if not self.memory_tool:
            return False

        try:
            result = self.memory_tool.run({
                "action": "clear_all",
                "user_id": user_id,
                "agent_id": agent_id or self.default_agent_id,
                "session_id": session_id
            })
            return "已清空" in result
        except Exception as e:
            print(f"[WARN] 清空记忆失败: {e}")
            return False