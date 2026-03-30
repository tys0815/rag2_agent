"""记忆工具（并发安全·无状态·四层架构）
所有ID必须通过方法传递，不存储在实例中，多用户绝对不会乱
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

from ..base import Tool, ToolParameter, tool_action
from ...memory import MemoryManager, MemoryConfig


class MemoryTool(Tool):
    """无状态记忆工具（并发安全）
    不存储 user_id / agent_id / session_id
    所有ID由上层Agent/请求每次调用传递
    """

    def __init__(self, expandable: bool = False):
        super().__init__(
            name="memory",
            description="记忆工具 - 存储和检索对话历史、知识、经验",
            expandable=expandable
        )

        # 只初始化记忆系统，不存任何用户状态
        self.memory_config = MemoryConfig()
        self.memory_manager = MemoryManager(
            config=self.memory_config
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        # if not self.validate_parameters(parameters):
        #     return "❌ 参数验证失败：缺少必需的参数"

        action = parameters.get("action")

        user_id = parameters.get("user_id", "default_user")
        session_id = parameters.get("session_id")

        if action == "add":
            return self._add_memory(
                content=parameters.get("content", ""),
                memory_type=parameters.get("memory_type", "working"),
                importance=parameters.get("importance", 0.5),
                file_path=parameters.get("file_path"),
                modality=parameters.get("modality"),
                user_id=user_id,
                session_id=session_id,
                user_content=parameters.get("user_content", ""),
                assistant_content=parameters.get("assistant_content", "")
            )
        elif action == "search":
            return self._search_memory(
                query=parameters.get("query"),
                limit=parameters.get("limit", 5),
                memory_types=parameters.get("memory_types"),
                min_importance=parameters.get("min_importance", 0.1),
                user_id=user_id,
                session_id=session_id
            )
        elif action == "summary":
            return self._get_summary(
                limit=parameters.get("limit", 10),
                user_id=user_id,
                session_id=session_id
            )
        elif action == "stats":
            return self._get_stats(
                user_id=user_id,
                session_id=session_id
            )
        elif action == "update":
            return self._update_memory(
                memory_id=parameters.get("memory_id"),
                content=parameters.get("content"),
                importance=parameters.get("importance"),
                user_id=user_id,
                session_id=session_id
            )
        elif action == "remove":
            return self._remove_memory(
                memory_id=parameters.get("memory_id"),
                user_id=user_id,
                session_id=session_id
            )
        elif action == "forget":
            return self._forget(
                strategy=parameters.get("strategy", "importance_based"),
                threshold=parameters.get("threshold", 0.1),
                max_age_days=parameters.get("max_age_days", 30),
                user_id=user_id,
                session_id=session_id
            )
        elif action == "consolidate":
            return self._consolidate(
                from_type=parameters.get("from_type", "working"),
                to_type=parameters.get("to_type", "episodic"),
                importance_threshold=parameters.get("importance_threshold", 0.7),
                user_id=user_id,
                session_id=session_id
            )
        elif action == "clear_all":
            return self._clear_all(
                user_id=user_id,
                session_id=session_id
            )
        else:
            return f"❌ 不支持的操作: {action}"

    def get_parameters(self) -> List[ToolParameter]:
        """工具参数（四层记忆架构 · 仅查询）"""
        return []
    #     return [
    #         ToolParameter(
    #             name="action",
    #             type="string",
    #             required=True,
    #             description="【记忆操作】仅支持 search：检索用户的历史记忆、上下文、偏好、知识",
    #             enum=["search"]
    #         ),
    #         ToolParameter(
    #             name="query",
    #             type="string",
    #             required=True,
    #             description="【必填】检索关键词，用于查找相关记忆内容"
    #         ),
    #         ToolParameter(
    #             name="memory_type",
    #             type="string",
    #             required=False,
    #             default="working",
    #             description="""四层记忆类型（自动选择即可）：
    # - working：工作记忆｜当前对话上下文、短期信息，会话级、临时、自动清理
    # - episodic：情景记忆｜历史交互事件、学习经历、长期对话记录
    # - semantic：语义记忆｜抽象知识、用户偏好、规则、知识点、长期知识体系
    # - perceptual：感知记忆｜图片、音频、文件、多模态信息、上传记录""",
    #             enum=["working", "episodic", "semantic", "perceptual"]
    #         )
    #     ]

    # -------------------------------------------------------------------------
    # 核心：所有方法都必须接收 user_id, agent_id, session_id
    # -------------------------------------------------------------------------
    @tool_action("memory_add", "添加记忆")
    def _add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        file_path: str = None,
        modality: str = None,
        user_id: str = "default_user",
        session_id: str = None,
        **kwargs
    ) -> str:
        try:
            metadata = {}

            # ==============================
            # 关键：知识库文档 不绑定 session_id
            # ==============================
            is_knowledge = memory_type in ["semantic", "perceptual"]
            use_session = None if is_knowledge else session_id

            # 感知记忆文件处理
            if file_path and memory_type == "perceptual":
                metadata["modality"] = modality or self._infer_modality(file_path)
                metadata["raw_data"] = file_path

            if memory_type == "perceptual" and metadata["modality"] == "text":
                return "❌ 感知记忆要求提供非文本文件路径"

            # 会话信息
            metadata.update({
                "session_id": use_session,
                "timestamp": datetime.now().isoformat()
            })

            # 调用记忆管理器（全部传参，无内部状态）
            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                user_id=user_id,
                session_id=use_session,
                importance=importance,
                metadata=metadata,
                **kwargs
            )

            scope = "全局知识库" if is_knowledge else f"会话{use_session}"
            return f"✅ 记忆添加成功 | {scope} ID:{memory_id[:8]}"

        except Exception as e:
            return f"❌ 添加失败：{str(e)}"

    @tool_action("memory_search", "搜索相关记忆")
    def _search_memory(
        self,
        query: str,
        user_id: str,
        session_id: str = None,
        limit: int = 5,
        memory_types: List[str] = None,
        min_importance: float = 0.1
    ) -> str:
        try:
            if not query:
                return "❌ 搜索查询不能为空"

            results = self.memory_manager.retrieve_memories(
                query=query,
                limit=limit,
                user_id=user_id,
                session_id=session_id,
                memory_types=memory_types if memory_types  else None,
                min_importance=min_importance
            )

            if not results:
                return f"🔍 未找到与 '{query}' 相关的记忆"

            # 格式化结果
            # 1. 按记忆类型分组
            grouped = defaultdict(list)
            type_map = {
                "working": "工作记忆",
                "semantic": "语义记忆",
                "episodic": "情景记忆",
                "perceptual": "感知记忆"
            }

            for m in results:
                type_label = type_map.get(m.memory_type, m.memory_type)
                grouped[type_label].append(m.content.strip())

            formatted_results = []

            # 2. 按你要的格式输出
            for type_label, items in grouped.items():
                if not items:
                    continue
                formatted_results.append(f"【{type_label}】")
                for idx, content in enumerate(items, 1):
                    formatted_results.append(f"{idx}. {content}")

            # 如果没有记忆
            if not formatted_results:
                formatted_results = ["暂无相关记忆"]

            return "\n".join(formatted_results)

        except Exception as e:
            return f"❌ 搜索失败：{str(e)}"

    @tool_action("memory_summary", "获取记忆摘要")
    def _get_summary(
        self,
        limit: int = 10,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = None
    ) -> str:
        try:
            stats = self.memory_manager.get_memory_stats(
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )

            summary = [
                f"📊 记忆系统摘要",
                f"用户: {user_id} | 智能体: {agent_id}",
                f"总记忆数: {stats.get('total_memories', 0)}",
            ]

            important = self.memory_manager.retrieve_memories(
                query="", limit=limit * 2, min_importance=0.5,
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )

            if important:
                summary.append(f"\n⭐ 重要记忆（前{limit}条）：")
                for i, m in enumerate(important[:limit], 1):
                    pre = m.content[:60] + "..." if len(m.content) > 60 else m.content
                    summary.append(f"  {i}. {pre} (重要性: {m.importance:.2f})")

            return "\n".join(summary)

        except Exception as e:
            return f"❌ 获取摘要失败：{str(e)}"

    @tool_action("memory_stats", "获取记忆统计")
    def _get_stats(
        self,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = None
    ) -> str:
        try:
            stats = self.memory_manager.get_memory_stats(
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )
            return (
                f"📈 记忆统计\n"
                f"用户: {user_id}\n"
                f"总数量: {stats.get('total_memories', 0)}\n"
                f"启用类型: {', '.join(stats.get('enabled_types', []))}"
            )
        except Exception as e:
            return f"❌ 获取统计失败：{str(e)}"

    @tool_action("memory_update", "更新记忆")
    def _update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = None
    ) -> str:
        try:
            if not memory_id:
                return "❌ 请提供 memory_id"

            success = self.memory_manager.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id
            )
            return "✅ 记忆已更新" if success else "⚠️ 未找到记忆"

        except Exception as e:
            return f"❌ 更新失败：{str(e)}"

    @tool_action("memory_remove", "删除记忆")
    def _remove_memory(
        self,
        memory_id: str,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = None
    ) -> str:
        try:
            if not memory_id:
                return "❌ 请提供 memory_id"

            success = self.memory_manager.remove_memory(
                memory_id=memory_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id
            )
            return "✅ 记忆已删除" if success else "⚠️ 未找到记忆"

        except Exception as e:
            return f"❌ 删除失败：{str(e)}"

    @tool_action("memory_forget", "批量遗忘记忆")
    def _forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = None
    ) -> str:
        try:
            count = self.memory_manager.forget_memories(
                strategy=strategy,
                threshold=threshold,
                max_age_days=max_age_days,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id
            )
            return f"🧹 已遗忘 {count} 条低价值记忆"

        except Exception as e:
            return f"❌ 遗忘失败：{str(e)}"

    @tool_action("memory_consolidate", "整合为长期记忆")
    def _consolidate(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = None
    ) -> str:
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id
            )
            return f"🔄 已整合 {count} 条重要记忆（{from_type} → {to_type}）"

        except Exception as e:
            return f"❌ 整合失败：{str(e)}"

    @tool_action("memory_clear_all", "清空所有记忆")
    def _clear_all(
        self,
        user_id: str = "default_user",
        session_id: str = None
    ) -> str:
        try:
            self.memory_manager.clear_all_memories(
                user_id=user_id, session_id=session_id
            )
            return "🧹 已清空当前范围所有记忆"
        except Exception as e:
            return f"❌ 清空失败：{str(e)}"

    # -------------------------------------------------------------------------
    # 扩展方法（全部无状态）
    # -------------------------------------------------------------------------
    def auto_record_conversation(
        self,
        user_input: str,
        agent_response: str,
        user_id: str,
        agent_id: str,
        session_id: str
    ):
        """自动记录对话（必须传全部ID，并发安全）"""
        self._add_memory(
            content=f"用户：{user_input}",
            memory_type="working",
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id
        )
        self._add_memory(
            content=f"助手：{agent_response}",
            memory_type="working",
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id
        )

    def add_knowledge(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        importance: float = 0.9
    ):
        """添加知识到语义记忆"""
        return self._add_memory(
            content=content,
            memory_type="semantic",
            importance=importance,
            user_id=user_id,
            agent_id=agent_id,
            session_id=None
        )

    def get_context_for_query(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        session_id: str = None,
        limit: int = 3
    ) -> str:
        """为查询获取相关上下文"""
        try:
            results = self.memory_manager.retrieve_memories(
                query=query, limit=limit, min_importance=0.3,
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )
            if not results:
                return ""
            return "\n".join([f"- {m.content}" for m in results])
        except:
            return ""

    def _infer_modality(self, path: str) -> str:
        try:
            ext = path.split('.')[-1].lower()
            if ext in {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'}:
                return 'image'
            if ext in {'mp3', 'wav', 'flac', 'm4a', 'ogg'}:
                return 'audio'
        except:
            pass
        return 'text'