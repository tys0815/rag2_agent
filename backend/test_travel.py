#!/usr/bin/env python3
"""测试旅游路线规划助手导入和基本功能"""

import sys
import os
import logging

# 设置日志级别，减少输出
logging.basicConfig(level=logging.WARNING)
logging.getLogger("helloAgents").setLevel(logging.WARNING)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 添加backend_app到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend_app'))

try:
    # 测试导入
    from helloAgents.agents.travel_planner_agent import TravelPlannerAgent
    from helloAgents.core.llm import HelloAgentsLLM
    from helloAgents.tools.registry import global_registry
    from helloAgents.tools.builtin.memory_tool import MemoryTool
    # from helloAgents.tools.builtin.rag_tool import RAGTool  # TravelPlannerAgent不需要RAG工具

    print("[OK] 导入成功")

    # 测试初始化
    # 先注册工具到全局注册表
    memory_tool = MemoryTool()
    global_registry.register_tool(memory_tool)
    # rag_tool = RAGTool()  # TravelPlannerAgent不需要RAG工具
    # global_registry.register_tool(rag_tool)

    llm = HelloAgentsLLM()
    agent = TravelPlannerAgent(
        name="test_travel_agent",
        llm=llm,
        tool_registry=global_registry,
        mcp_server_command=None  # 使用内置演示服务器
    )

    print("[OK] Agent初始化成功")
    print(f"Agent名称: {agent.name}")
    print(f"默认助手ID: {agent.default_agent_id}")
    print(f"MCP工具: {'已加载' if agent.mcp_tool else '未加载'}")
    print(f"记忆工具: {'已加载' if agent.memory_tool else '未加载'}")

    # 测试信息提取方法
    print("\n[TEST] 测试旅游信息提取...")
    try:
        test_text = "我想去北京旅游3天，预算5000元，喜欢历史文化和美食"
        extracted_info = agent._extract_travel_info(test_text)
        print(f"[OK] 信息提取成功")
        print(f"提取的信息: {extracted_info}")
    except Exception as e:
        print(f"[WARN] 信息提取测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试简单运行（不实际调用LLM，但会调用MCP工具）
    print("\n[TEST] 测试运行方法...")
    try:
        response = agent.run(
            input_text="我想去上海旅游2天",
            user_id="test_user_001",
            agent_id="test_travel_agent",
            session_id="test_travel_session_001",
            enable_memory=False,
            max_context_length=1000
        )
        print(f"[OK] 运行成功，响应长度: {len(response)}")
        print(f"响应前200字符: {response[:200]}...")
    except Exception as e:
        print(f"[WARN] 运行测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试记忆统计
    print("\n[TEST] 测试记忆统计...")
    try:
        stats = agent.get_memory_stats(
            user_id="test_user_001",
            agent_id="test_travel_agent",
            session_id="test_travel_session_001"
        )
        print(f"[OK] 记忆统计成功: {stats}")
    except Exception as e:
        print(f"[WARN] 记忆统计测试失败: {e}")

    print("\n[DONE] 旅游路线规划助手测试完成")

except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)