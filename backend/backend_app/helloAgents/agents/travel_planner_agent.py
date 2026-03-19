"""旅游路线规划助手
集成MCP服务实现旅游路线规划功能，包括：
- 提取地点、时间、天数
- 搜索天气、酒店、景点等信息
- 生成路线方案
- 使用记忆工具记录对话内容
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


class TravelPlannerAgent(Agent):
    """旅游路线规划助手

    特点：
    1. MCP服务集成：通过MCP工具调用天气、酒店、景点等外部服务
    2. 记忆管理：自动记录对话历史，检索相关记忆
    3. 信息提取：自动提取用户输入中的地点、时间、天数等关键信息
    4. 路线生成：基于提取的信息和外部服务数据生成个性化路线方案
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = None,
        default_agent_id: str = "travel_planner",
        mcp_server_command: Optional[List[str]] = None,
        mcp_server_args: Optional[List[str]] = None,
        mcp_env: Optional[Dict[str, str]] = None
    ):
        """
        初始化旅游路线规划助手

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

        # 初始化MCP工具（旅游服务）
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
                print("[MCP] 使用内置MCP演示服务器（可替换为旅游服务MCP服务器）")
                self.mcp_tool = MCPTool(
                    name="travel_mcp",
                    description="旅游服务MCP工具，提供天气、酒店、景点等信息查询",
                    auto_expand=True
                )
            else:
                print(f"[MCP] 连接到外部MCP服务器: {' '.join(server_command)}")
                self.mcp_tool = MCPTool(
                    name="travel_mcp",
                    description="旅游服务MCP工具，提供天气、酒店、景点等信息查询",
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
        运行旅游路线规划助手

        Args:
            input_text: 用户输入
            user_id: 用户ID（必需）
            agent_id: 助手ID（默认为travel_planner）
            session_id: 会话ID（可选，不传则自动生成）
            enable_memory: 是否启用记忆
            max_context_length: 最大上下文长度
            **kwargs: 其他参数

        Returns:
            Agent响应
        """
        # 参数处理
        agent_id = agent_id or self.default_agent_id
        session_id = session_id or self._generate_session_id()

        print(f"[Travel] 旅游路线规划助手启动: user={user_id}, agent={agent_id}, session={session_id}")

        # 1. 记录用户输入到工作记忆
        if enable_memory and self.memory_tool:
            self._record_to_memory(
                content=f"用户旅游需求: {input_text}",
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type="working"
            )

        # 2. 提取旅游关键信息（地点、时间、天数等）
        extracted_info = self._extract_travel_info(input_text)
        print(f"[Info] 提取的旅游信息: {extracted_info}")

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

        # 4. 调用MCP服务获取外部数据
        mcp_data = self._call_mcp_services(extracted_info)
        print(f"[MCP] MCP服务数据: {mcp_data}")

        # 5. 构建增强上下文
        enhanced_context = self._build_enhanced_context(
            user_input=input_text,
            extracted_info=extracted_info,
            memory_context=memory_context,
            mcp_data=mcp_data,
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

        # 7. 调用LLM生成回答（旅游路线方案）
        try:
            response = self.llm.invoke(messages, **kwargs)
        except Exception as e:
            response = f"❌ 生成旅游路线方案时出错: {str(e)}"

        # 8. 记录助手回答到工作记忆
        if enable_memory and self.memory_tool:
            self._record_to_memory(
                content=f"旅游路线方案: {response}",
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
        return f"travel_ses_{uuid.uuid4().hex[:16]}"

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

    def _extract_travel_info(self, text: str) -> Dict[str, Any]:
        """使用function calling提取旅游关键信息：地点、时间、天数等"""
        # 定义旅游信息提取的function schema
        travel_info_schema = {
            "type": "function",
            "function": {
                "name": "extract_travel_info",
                "description": "从用户输入中提取旅游相关信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "locations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "旅游目的地列表，如['北京', '上海']"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "旅行开始日期，格式YYYY-MM-DD，如未指定则为null"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "旅行结束日期，格式YYYY-MM-DD，如未指定则为null"
                        },
                        "duration_days": {
                            "type": "integer",
                            "description": "旅行天数，如3天则为3"
                        },
                        "budget": {
                            "type": "string",
                            "description": "旅行预算，如'5000元'或'1万元'"
                        },
                        "interests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "旅行兴趣偏好，如['历史文化', '美食', '自然风光']"
                        },
                        "travelers": {
                            "type": "integer",
                            "description": "旅行人数"
                        },
                        "travel_type": {
                            "type": "string",
                            "description": "旅行类型，如'家庭游', '情侣游', '商务旅行', '自由行'等"
                        },
                        "accommodation_preference": {
                            "type": "string",
                            "description": "住宿偏好，如'酒店', '民宿', '经济型'等"
                        },
                        "transportation_preference": {
                            "type": "string",
                            "description": "交通偏好，如'飞机', '高铁', '自驾'等"
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
                "content": "你是一个旅游信息提取助手。请从用户输入中提取旅游相关信息，包括地点、时间、天数、预算、兴趣等。如果某些信息未明确提及，请设为null。"
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
                tools=[travel_info_schema],
                tool_choice={"type": "function", "function": {"name": "extract_travel_info"}},
                temperature=0.1  # 低温度以获得更确定的结果
            )

            # 解析响应
            choice = response.choices[0]
            message = choice.message

            if message.tool_calls and len(message.tool_calls) > 0:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "extract_travel_info":
                    import json
                    arguments = json.loads(tool_call.function.arguments)

                    # 确保所有字段都有默认值
                    info = {
                        "locations": arguments.get("locations", []),
                        "start_date": arguments.get("start_date"),
                        "end_date": arguments.get("end_date"),
                        "duration_days": arguments.get("duration_days"),
                        "budget": arguments.get("budget"),
                        "interests": arguments.get("interests", []),
                        "travelers": arguments.get("travelers"),
                        "travel_type": arguments.get("travel_type"),
                        "accommodation_preference": arguments.get("accommodation_preference"),
                        "transportation_preference": arguments.get("transportation_preference")
                    }

                    # 如果没有检测到地点，使用占位符
                    if not info["locations"]:
                        info["locations"] = ["未指定"]

                    print(f"[Info] 通过function calling提取的旅游信息: {info}")
                    return info

        except Exception as e:
            print(f"[WARN] 使用function calling提取旅游信息失败: {e}")
            # 失败时回退到简单规则提取
            return self._fallback_extract_travel_info(text)

        # 如果没有提取到信息，使用回退方法
        return self._fallback_extract_travel_info(text)

    def _fallback_extract_travel_info(self, text: str) -> Dict[str, Any]:
        """回退方法：使用简单规则提取旅游信息"""
        info = {
            "locations": [],
            "start_date": None,
            "end_date": None,
            "duration_days": None,
            "budget": None,
            "interests": [],
            "travelers": None,
            "travel_type": None,
            "accommodation_preference": None,
            "transportation_preference": None
        }

        # 简单关键词匹配
        import re

        # 提取天数
        day_match = re.search(r'(\d+)\s*天', text)
        if day_match:
            info["duration_days"] = int(day_match.group(1))

        # 提取人数
        people_match = re.search(r'(\d+)\s*人', text)
        if people_match:
            info["travelers"] = int(people_match.group(1))

        # 提取预算
        budget_match = re.search(r'预算\s*(\d+[元万]?)', text)
        if budget_match:
            info["budget"] = budget_match.group(1)

        # 简单地点识别
        locations = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京", "武汉", "重庆",
                    "苏州", "厦门", "青岛", "大连", "天津", "长沙", "郑州", "合肥", "福州", "南宁",
                    "昆明", "贵阳", "兰州", "西宁", "银川", "乌鲁木齐", "拉萨", "香港", "澳门", "台湾"]
        for loc in locations:
            if loc in text:
                info["locations"].append(loc)

        # 兴趣识别
        interests_keywords = {
            "历史文化": ["历史", "文化", "古迹", "博物馆", "遗址", "传统"],
            "自然风光": ["自然", "风景", "山水", "公园", "森林", "湖泊", "海滩"],
            "美食": ["美食", "小吃", "餐厅", "特色菜", "吃货", "美味"],
            "购物": ["购物", "商场", "逛街", "买", "购物中心", "商业街"],
            "娱乐": ["娱乐", "游乐场", "主题公园", "电影", "KTV", "酒吧"],
            "户外运动": ["户外", "运动", "登山", "徒步", "骑行", "滑雪", "潜水"],
            "亲子": ["亲子", "儿童", "孩子", "家庭", "小朋友", "宝宝"]
        }

        for interest_type, keywords in interests_keywords.items():
            for keyword in keywords:
                if keyword in text and interest_type not in info["interests"]:
                    info["interests"].append(interest_type)
                    break

        # 旅行类型识别
        if "家庭" in text or "亲子" in text or "带孩子" in text:
            info["travel_type"] = "家庭游"
        elif "情侣" in text or "夫妻" in text or "蜜月" in text:
            info["travel_type"] = "情侣游"
        elif "商务" in text or "出差" in text or "会议" in text:
            info["travel_type"] = "商务旅行"
        elif "自由行" in text or "自助游" in text:
            info["travel_type"] = "自由行"
        elif "跟团" in text or "旅行团" in text:
            info["travel_type"] = "跟团游"

        # 如果没有检测到地点，使用占位符
        if not info["locations"]:
            info["locations"] = ["未指定"]

        return info

    def _call_mcp_services(self, travel_info: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP服务获取外部数据"""
        mcp_data = {
            "weather": {},
            "hotels": [],
            "attractions": [],
            "transportation": []
        }

        if not self.mcp_tool:
            print("[WARN] MCP工具未初始化，返回模拟数据")
            # 返回模拟数据用于演示
            return self._get_mock_mcp_data(travel_info)

        try:
            # 获取MCP工具列表
            tools_result = self.mcp_tool.run({"action": "list_tools"})
            print(f"[MCP] 可用的MCP工具: {tools_result}")

            # 根据旅游信息调用相应的MCP工具
            locations = travel_info.get("locations", [])
            if locations and locations[0] != "未指定":
                location = locations[0]

                # 尝试调用天气查询工具
                try:
                    weather_result = self.mcp_tool.run({
                        "action": "call_tool",
                        "tool_name": "get_weather",
                        "arguments": {"location": location}
                    })
                    mcp_data["weather"] = weather_result
                except:
                    # 如果工具不存在，使用模拟数据
                    mcp_data["weather"] = {"location": location, "temperature": "25°C", "condition": "晴朗"}

                # 尝试调用酒店查询工具
                try:
                    hotels_result = self.mcp_tool.run({
                        "action": "call_tool",
                        "tool_name": "search_hotels",
                        "arguments": {"location": location, "check_in": "2024-01-01", "check_out": "2024-01-05"}
                    })
                    mcp_data["hotels"] = hotels_result
                except:
                    mcp_data["hotels"] = [
                        {"name": f"{location}大酒店", "price": "¥500/晚", "rating": 4.5},
                        {"name": f"{location}精品酒店", "price": "¥300/晚", "rating": 4.2}
                    ]

                # 尝试调用景点查询工具
                try:
                    attractions_result = self.mcp_tool.run({
                        "action": "call_tool",
                        "tool_name": "search_attractions",
                        "arguments": {"location": location}
                    })
                    mcp_data["attractions"] = attractions_result
                except:
                    mcp_data["attractions"] = [
                        {"name": f"{location}著名景点1", "type": "历史文化", "price": "¥100"},
                        {"name": f"{location}著名景点2", "type": "自然风光", "price": "¥80"}
                    ]

        except Exception as e:
            print(f"⚠️ 调用MCP服务失败: {e}")
            # 返回模拟数据
            mcp_data = self._get_mock_mcp_data(travel_info)

        return mcp_data

    def _get_mock_mcp_data(self, travel_info: Dict[str, Any]) -> Dict[str, Any]:
        """获取模拟的MCP数据（用于演示）"""
        locations = travel_info.get("locations", ["未知地点"])
        location = locations[0] if locations else "未知地点"

        return {
            "weather": {
                "location": location,
                "temperature": "22-28°C",
                "condition": "多云转晴",
                "humidity": "65%",
                "wind": "微风"
            },
            "hotels": [
                {"name": f"{location}国际大酒店", "price": "¥680/晚", "rating": 4.7, "address": "市中心"},
                {"name": f"{location}精品民宿", "price": "¥350/晚", "rating": 4.5, "address": "景区附近"},
                {"name": f"{location}经济型酒店", "price": "¥220/晚", "rating": 4.0, "address": "交通便利"}
            ],
            "attractions": [
                {"name": f"{location}历史文化街区", "type": "文化古迹", "price": "免费", "time_needed": "2-3小时"},
                {"name": f"{location}自然公园", "type": "自然风光", "price": "¥60", "time_needed": "3-4小时"},
                {"name": f"{location}博物馆", "type": "文化教育", "price": "¥80", "time_needed": "2小时"}
            ],
            "transportation": [
                {"type": "地铁", "description": "覆盖主要景点", "cost": "¥5-10/次"},
                {"type": "公交", "description": "线路丰富", "cost": "¥2/次"},
                {"type": "出租车", "description": "方便快捷", "cost": "起步价¥12"}
            ]
        }

    def _build_enhanced_context(
        self,
        user_input: str,
        extracted_info: Dict[str, Any],
        memory_context: str,
        mcp_data: Dict[str, Any],
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> str:
        """构建增强上下文"""
        context_parts = []

        # 系统信息
        context_parts.append(f"# 旅游路线规划助手上下文")
        context_parts.append(f"- 用户: {user_id}")
        context_parts.append(f"- 助手: {agent_id}")
        context_parts.append(f"- 会话: {session_id}")
        context_parts.append(f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_parts.append("")

        # 用户问题
        context_parts.append(f"## 用户旅游需求")
        context_parts.append(f"{user_input}")
        context_parts.append("")

        # 提取的旅游信息
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

        # MCP服务数据
        context_parts.append(f"## 旅游服务信息")

        # 天气信息
        if mcp_data.get("weather"):
            weather = mcp_data["weather"]
            context_parts.append(f"### 天气情况")
            if isinstance(weather, dict):
                for key, value in weather.items():
                    context_parts.append(f"- {key}: {value}")
            else:
                context_parts.append(f"{weather}")
            context_parts.append("")

        # 酒店信息
        if mcp_data.get("hotels"):
            context_parts.append(f"### 推荐酒店")
            hotels = mcp_data["hotels"]
            if isinstance(hotels, list):
                for i, hotel in enumerate(hotels[:3], 1):
                    if isinstance(hotel, dict):
                        name = hotel.get("name", f"酒店{i}")
                        price = hotel.get("price", "价格未知")
                        rating = hotel.get("rating", "评分未知")
                        context_parts.append(f"{i}. {name} | {price} | 评分: {rating}")
                    else:
                        context_parts.append(f"{i}. {hotel}")
            else:
                context_parts.append(f"{hotels}")
            context_parts.append("")

        # 景点信息
        if mcp_data.get("attractions"):
            context_parts.append(f"### 推荐景点")
            attractions = mcp_data["attractions"]
            if isinstance(attractions, list):
                for i, attr in enumerate(attractions[:5], 1):
                    if isinstance(attr, dict):
                        name = attr.get("name", f"景点{i}")
                        type_ = attr.get("type", "景点")
                        price = attr.get("price", "价格未知")
                        context_parts.append(f"{i}. {name} ({type_}) | {price}")
                    else:
                        context_parts.append(f"{i}. {attr}")
            else:
                context_parts.append(f"{attractions}")
            context_parts.append("")

        # 交通信息
        if mcp_data.get("transportation"):
            context_parts.append(f"### 交通方式")
            transport = mcp_data["transportation"]
            if isinstance(transport, list):
                for i, trans in enumerate(transport, 1):
                    if isinstance(trans, dict):
                        type_ = trans.get("type", f"交通方式{i}")
                        desc = trans.get("description", "")
                        cost = trans.get("cost", "")
                        context_parts.append(f"- {type_}: {desc} ({cost})")
                    else:
                        context_parts.append(f"- {trans}")
            else:
                context_parts.append(f"{transport}")
            context_parts.append("")

        # 路线规划要求
        context_parts.append(f"## 路线规划要求")
        context_parts.append("1. 基于以上信息生成个性化的旅游路线方案")
        context_parts.append("2. 路线应包括：每日行程安排、景点推荐、餐饮建议、交通方式、住宿推荐")
        context_parts.append("3. 考虑时间、预算、兴趣等约束条件")
        context_parts.append("4. 提供实用建议：最佳游览时间、注意事项、必备物品等")
        context_parts.append("5. 使用清晰的结构化格式（如表格、时间线）")
        context_parts.append("6. 保持友好、专业的语气")

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
        return """你是一个专业的旅游路线规划助手，具备以下能力：
1. 🧳 信息提取：能从用户需求中提取地点、时间、天数、预算、兴趣等关键信息
2. 🌐 数据整合：能整合天气、酒店、景点、交通等外部服务信息
3. 📅 路线规划：能制定详细、可行的每日行程安排
4. 💡 实用建议：能提供游览建议、注意事项、必备物品等实用信息
5. 💰 预算管理：能根据用户预算推荐合适的住宿、餐饮和活动
6. 🎯 个性化：能根据用户兴趣（历史文化、自然风光、美食购物等）定制路线

请基于提供的上下文信息，为用户生成专业、详细、个性化的旅游路线方案。"""

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