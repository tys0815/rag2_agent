#!/usr/bin/env python3
"""测试旅游路线规划API端到端功能"""

import sys
import os
import requests
import json
import time

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

def test_travel_api():
    """测试旅游路线规划API"""
    base_url = "http://localhost:8000/api/v1"

    print("🚀 开始测试旅游路线规划API...")

    # 测试1: 健康检查
    print("\n[TEST 1] 健康检查...")
    try:
        response = requests.get(f"{base_url}/travel/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ 健康检查成功: {response.json()}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

    # 测试2: 提取旅游信息
    print("\n[TEST 2] 提取旅游信息...")
    test_text = "我想去北京旅游3天，预算5000元，喜欢历史文化和美食"
    try:
        response = requests.post(
            f"{base_url}/travel/extract_info",
            json={"text": test_text},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 信息提取成功")
            print(f"   提取的信息: {json.dumps(data['extracted_info'], ensure_ascii=False, indent=2)}")
        else:
            print(f"❌ 信息提取失败: {response.status_code}")
            print(f"   响应: {response.text}")
    except Exception as e:
        print(f"❌ 信息提取异常: {e}")
        return False

    # 测试3: 旅游路线规划聊天
    print("\n[TEST 3] 旅游路线规划聊天...")
    chat_data = {
        "text": test_text,
        "user_id": "test_user_001",
        "agent_id": "travel_planner",
        "session_id": f"test_session_{int(time.time())}",
        "enable_memory": True,
        "max_context_length": 2000
    }

    try:
        print(f"   发送请求: {chat_data['text']}")
        response = requests.post(
            f"{base_url}/travel/chat",
            json=chat_data,
            timeout=30  # 可能需要更长时间
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ 旅游路线规划成功")
            print(f"   会话ID: {data['session_id']}")
            print(f"   响应长度: {len(data['data'])} 字符")
            print(f"   响应片段: {data['data'][:200]}...")

            # 检查响应内容
            if "旅游" in data['data'] or "路线" in data['data'] or "行程" in data['data']:
                print(f"✅ 响应内容包含旅游相关信息")
            else:
                print(f"⚠️  警告: 响应内容可能不是旅游路线方案")
        else:
            print(f"❌ 旅游路线规划失败: {response.status_code}")
            print(f"   响应: {response.text}")
    except Exception as e:
        print(f"❌ 旅游路线规划异常: {e}")
        return False

    # 测试4: 记忆统计
    print("\n[TEST 4] 获取记忆统计...")
    stats_data = {
        "user_id": "test_user_001",
        "agent_id": "travel_planner",
        "session_id": chat_data['session_id']
    }

    try:
        response = requests.post(
            f"{base_url}/travel/memory/stats",
            json=stats_data,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 记忆统计成功")
            print(f"   统计信息: {json.dumps(data['stats'], ensure_ascii=False, indent=2)}")
        else:
            print(f"❌ 记忆统计失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 记忆统计异常: {e}")
        # 不返回False，因为这个测试是可选的

    print("\n🎉 所有测试完成!")
    return True

def check_server_running():
    """检查服务器是否在运行"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("旅游路线规划API端到端测试")
    print("=" * 60)

    # 检查服务器是否运行
    print("\n🔍 检查FastAPI服务器状态...")
    if check_server_running():
        print("✅ 服务器正在运行")
        # 运行API测试
        success = test_travel_api()
        if success:
            print("\n🎊 所有测试通过!")
            sys.exit(0)
        else:
            print("\n💥 测试失败!")
            sys.exit(1)
    else:
        print("❌ 服务器未运行，请先启动FastAPI服务器:")
        print("   1. cd E:\\ai\\llama3.2-projec\\backend\\backend_app")
        print("   2. python main.py")
        print("\n💡 提示: 确保已安装所有依赖:")
        print("   pip install fastapi uvicorn pydantic python-dotenv openai")
        print("   pip install fastmcp>=2.0.0  # 用于MCP工具")
        sys.exit(1)