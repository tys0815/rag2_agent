#!/usr/bin/env python3
"""测试企业级RAG助手导入和基本功能"""

import sys
import os

# 添加backend_app到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend_app'))

try:
    # 测试导入
    from helloAgents.agents.enterprise_rag_agent import EnterpriseRagAgent
    from helloAgents.core.llm import HelloAgentsLLM
    from helloAgents.tools.registry import global_registry

    print("[OK] 导入成功")

    # 测试初始化
    llm = HelloAgentsLLM()
    agent = EnterpriseRagAgent(
        name="test_enterprise_agent",
        llm=llm,
        tool_registry=global_registry
    )

    print("[OK] Agent初始化成功")
    print(f"Agent名称: {agent.name}")
    print(f"默认助手ID: {agent.default_agent_id}")
    print(f"RAG工具: {'已加载' if agent.rag_tool else '未加载'}")
    print(f"记忆工具: {'已加载' if agent.memory_tool else '未加载'}")

    # 测试简单运行（不实际调用LLM）
    print("\n[TEST] 测试运行方法...")
    try:
        # 模拟运行，不实际调用LLM
        response = agent.run(
            input_text="你好",
            user_id="test_user_001",
            agent_id="test_agent",
            session_id="test_session_001",
            enable_memory=False,
            enable_rag=False
        )
        print(f"[OK] 运行成功，响应长度: {len(response)}")
    except Exception as e:
        print(f"[WARN] 运行测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n[DONE] 企业级RAG助手测试完成")

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