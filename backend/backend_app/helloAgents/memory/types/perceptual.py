"""感知记忆实现（长存的多模态）

按照第8章架构设计的感知记忆（长期、多模态），提供：
- 多模态数据存储（文本、图像、音频等）
- 结构化元数据 + 向量索引（SQLite + Qdrant）
- 同模态检索（跨模态在无CLIP/CLAP依赖时有限）
- 懒加载编码：文本用 sentence-transformers；图像/音频用轻量确定性哈希向量
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib
import os
import random
import logging

logger = logging.getLogger(__name__)

from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..storage import SQLiteDocumentStore, QdrantVectorStore
from ..embedding import get_text_embedder, get_dimension


class Perception:
    """感知数据实体"""

    def __init__(
        self,
        perception_id: str,
        data: Any,
        modality: str,
        encoding: Optional[List[float]] = None,
        metadata: Dict[str, Any] = None
    ):
        self.perception_id = perception_id
        self.data = data
        self.modality = modality  # text, image, audio, video, structured
        self.encoding = encoding or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.data_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """计算数据哈希"""
        if isinstance(self.data, str):
            return hashlib.md5(self.data.encode()).hexdigest()
        elif isinstance(self.data, bytes):
            return hashlib.md5(self.data).hexdigest()
        else:
            return hashlib.md5(str(self.data).encode()).hexdigest()


class PerceptualMemory(BaseMemory):
    """感知记忆实现（企业级知识库·只按用户隔离）

    特点：
    - 支持多模态数据（文本、图像、音频等）
    - 跨模态相似性搜索
    - 感知数据的语义理解
    - 支持内容生成和检索
    - 知识库：只按 user_id 隔离，不绑定 agent_id / session_id
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # 感知数据存储（内存缓存）
        self.perceptions: Dict[str, Perception] = {}
        self.perceptual_memories: List[MemoryItem] = []

        # 模态索引
        self.modality_index: Dict[str, List[str]] = {}  # modality -> perception_ids

        # 支持的模态
        self.supported_modalities = set(self.config.perceptual_memory_modalities)

        # 文档权威存储（SQLite）
        db_dir = getattr(self.config, 'storage_path', "./memory_data")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)

        # 嵌入维度（与统一文本嵌入保持一致）
        self.text_embedder = get_text_embedder()
        self.vector_dim = get_dimension(getattr(self.text_embedder, 'dimension', 384))

        # 可选加载：图像CLIP与音频CLAP（缺依赖则优雅降级为哈希编码）
        self._clip_model = None
        self._clip_processor = None
        self._clap_model = None
        self._clap_processor = None
        self._image_dim = None
        self._audio_dim = None
        try:
            from transformers import CLIPModel, CLIPProcessor
            clip_name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
            self._clip_model = CLIPModel.from_pretrained(clip_name)
            self._clip_processor = CLIPProcessor.from_pretrained(clip_name)
            self._image_dim = self._clip_model.config.projection_dim if hasattr(self._clip_model.config,
                                                                                'projection_dim') else 512
        except Exception:
            self._clip_model = None
            self._clip_processor = None
            self._image_dim = self.vector_dim
        try:
            from transformers import ClapProcessor, ClapModel
            clap_name = os.getenv("CLAP_MODEL", "laion/clap-htsat-unfused")
            self._clap_model = ClapModel.from_pretrained(clap_name)
            self._clap_processor = ClapProcessor.from_pretrained(clap_name)
            self._audio_dim = getattr(self._clap_model.config, 'projection_dim', None) or 512
        except Exception:
            self._clap_model = None
            self._clap_processor = None
            self._audio_dim = self.vector_dim

        # 向量存储（Qdrant）— 按模态拆分集合，避免维度冲突
        from ..storage.qdrant_store import QdrantConnectionManager
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        base_collection = os.getenv("QDRANT_COLLECTION", "hello_agents_vectors")
        distance = os.getenv("QDRANT_DISTANCE", "cosine")

        self.vector_stores: Dict[str, QdrantVectorStore] = {}
        self.vector_stores["text"] = QdrantConnectionManager.get_instance(
            url=qdrant_url, api_key=qdrant_api_key,
            collection_name=f"{base_collection}_perceptual_text",
            vector_size=self.vector_dim, distance=distance
        )
        self.vector_stores["image"] = QdrantConnectionManager.get_instance(
            url=qdrant_url, api_key=qdrant_api_key,
            collection_name=f"{base_collection}_perceptual_image",
            vector_size=int(self._image_dim or self.vector_dim), distance=distance
        )
        self.vector_stores["audio"] = QdrantConnectionManager.get_instance(
            url=qdrant_url, api_key=qdrant_api_key,
            collection_name=f"{base_collection}_perceptual_audio",
            vector_size=int(self._audio_dim or self.vector_dim), distance=distance
        )

        self.encoders = self._init_encoders()
        logger.info("✅ 感知记忆初始化完成（多模态+用户隔离知识库）")

    # =========================================================================
    # 🔹 核心：添加感知记忆（只存 user_id，不存 agent_id / session_id）
    # =========================================================================
    def add(self, memory_item: MemoryItem, agent_id: str = None, session_id: str = None) -> str:
        """添加感知记忆（知识库：只归属用户）"""
        modality = memory_item.metadata.get("modality", "text")
        raw_data = memory_item.metadata.get("raw_data", memory_item.content)
        if modality not in self.supported_modalities:
            raise ValueError(f"不支持的模态类型: {modality}")

        # 编码
        perception = self._encode_perception(raw_data, modality, memory_item.id)

        # 内存缓存
        self.perceptions[perception.perception_id] = perception
        self.modality_index.setdefault(modality, []).append(perception.perception_id)

        # ======================== 关键修改 ========================
        # 只保留必要元数据，不存 agent_id / session_id
        # ===========================================================
        memory_item.metadata["perception_id"] = perception.perception_id
        memory_item.metadata["modality"] = modality
        self.perceptual_memories.append(memory_item)

        # 1) SQLite 权威入库（只存 user_id）
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="perceptual",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "perception_id": perception.perception_id,
                "modality": modality,
                "context": memory_item.metadata.get("context", {}),
                "tags": memory_item.metadata.get("tags", []),
            }
        )

        # 2) Qdrant 向量入库（只存 user_id）
        try:
            vector = perception.encoding
            store = self._get_vector_store_for_modality(modality)
            store.add_vectors(
                vectors=[vector],
                metadata=[{
                    "memory_id": memory_item.id,
                    "user_id": memory_item.user_id,
                    "memory_type": "perceptual",
                    "modality": modality,
                    "importance": memory_item.importance,
                    "content": memory_item.content,
                }],
                ids=[memory_item.id]
            )
        except Exception:
            pass

        return memory_item.id

    def retrieve(self, query: str, limit: int = 5,
                user_id=None, agent_id=None, session_id=None,
                target_modality=None, **kwargs) -> List[MemoryItem]:
        """检索感知记忆（只按 user_id 过滤，忽略 agent_id / session_id）"""
        query_modality = kwargs.get("query_modality", target_modality or "text")

        try:
            qvec = self._encode_data(query, query_modality)
            where = {"memory_type": "perceptual"}
            if user_id:
                where["user_id"] = user_id

            if target_modality:
                where["modality"] = target_modality

            store = self._get_vector_store_for_modality(target_modality or query_modality)
            hits = store.search_similar(
                query_vector=qvec,
                limit=max(limit * 5, 20),
                where=where
            )
        except Exception:
            hits = []

        # 融合排序
        now_ts = int(datetime.now().timestamp())
        results: List[Tuple[float, MemoryItem]] = []
        seen = set()

        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = meta.get("memory_id")
            if not mem_id or mem_id in seen:
                continue
            doc = self.doc_store.get_memory(mem_id)
            if not doc:
                continue

            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(doc["timestamp"])) / 86400.0)
            recency_score = 1.0 / (1.0 + age_days)
            imp = float(doc.get("importance", 0.5))

            base_relevance = vec_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (imp * 0.4)
            combined = base_relevance * importance_weight

            item = MemoryItem(
                id=doc["memory_id"],
                content=doc["content"],
                memory_type=doc["memory_type"],
                user_id=doc["user_id"],
                timestamp=datetime.fromtimestamp(doc["timestamp"]),
                importance=imp,
                metadata={
                    **doc.get("properties", {}),
                    "relevance_score": combined,
                    "vector_score": vec_score,
                    "recency_score": recency_score
                }
            )
            results.append((combined, item))
            seen.add(mem_id)

        # 兜底匹配
        if not results:
            for m in self.perceptual_memories:
                if user_id and m.user_id != user_id:
                    continue
                if target_modality and m.metadata.get("modality") != target_modality:
                    continue
                if query.lower() in (m.content or "").lower():
                    age = (now_ts - int(m.timestamp.timestamp())) / 86400.0
                    recency_score = 1.0 / (1.0 + max(0.0, age))
                    base = 0.5 * 0.8 + recency_score * 0.2
                    weight = 0.8 + m.importance * 0.4
                    combined = base * weight
                    results.append((combined, m))

        results.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in results[:limit]]

    # =========================================================================
    # 🔹 所有操作：只校验 user_id，忽略 agent_id / session_id
    # =========================================================================
    def update(self, memory_id: str, content=None, importance=None, metadata=None,
               agent_id=None, session_id=None) -> bool:
        mem = next((m for m in self.perceptual_memories if m.id == memory_id), None)
        if not mem:
            return False

        updated = False
        modality = mem.metadata.get("modality", "text")
        if content is not None:
            mem.content = content
            updated = True
        if importance is not None:
            mem.importance = importance
            updated = True
        if metadata is not None:
            mem.metadata.update(metadata)
            updated = True

        self.doc_store.update_memory(memory_id, content, importance, metadata)

        if content is not None or (metadata and "raw_data" in metadata):
            try:
                raw = metadata.get("raw_data", content) if metadata else content
                perc = self._encode_perception(raw or "", modality, memory_id)
                store = self._get_vector_store_for_modality(modality)
                store.add_vectors(
                    vectors=[perc.encoding],
                    metadata=[{
                        "memory_id": memory_id,
                        "user_id": mem.user_id,
                        "modality": modality,
                        "importance": mem.importance,
                        "content": mem.content
                    }],
                    ids=[memory_id]
                )
            except Exception:
                pass
        return updated

    def remove(self, memory_id: str, agent_id=None, session_id=None) -> bool:
        mem = next((m for m in self.perceptual_memories if m.id == memory_id), None)
        if not mem:
            return False

        try:
            self.perceptual_memories.remove(mem)
            perc_id = mem.metadata.get("perception_id")
            if perc_id in self.perceptions:
                modality = self.perceptions[perc_id].modality
                del self.perceptions[perc_id]
                if modality in self.modality_index and perc_id in self.modality_index[modality]:
                    self.modality_index[modality].remove(perc_id)
                    if not self.modality_index[modality]:
                        del self.modality_index[modality]

            self.doc_store.delete_memory(memory_id)
            for s in self.vector_stores.values():
                s.delete_memories([memory_id])
            return True
        except Exception:
            return False

    def has_memory(self, memory_id: str, agent_id=None, session_id=None) -> bool:
        mem = next((m for m in self.perceptual_memories if m.id == memory_id), None)
        return mem is not None

    def forget(self, strategy="importance_based", threshold=0.1, max_age_days=30,
               user_id=None, agent_id=None, session_id=None) -> int:
        now = datetime.now()
        to_del = []
        for m in self.perceptual_memories:
            if user_id and m.user_id != user_id:
                continue

            forget = False
            if strategy == "importance_based" and m.importance < threshold:
                forget = True
            if strategy == "time_based":
                if m.timestamp < now - timedelta(days=max_age_days):
                    forget = True
            if forget:
                to_del.append(m.id)

        cnt = 0
        for mid in list(set(to_del)):
            if self.remove(mid):
                cnt += 1
        return cnt

    def clear(self, user_id=None, agent_id=None, session_id=None):
        to_del = []
        for m in self.perceptual_memories:
            if user_id and m.user_id != user_id:
                continue
            to_del.append(m.id)
        for mid in to_del:
            self.remove(mid)

    def get_all(self, user_id=None, agent_id=None, session_id=None) -> List[MemoryItem]:
        res = []
        for m in self.perceptual_memories:
            if user_id and m.user_id != user_id:
                continue
            res.append(m)
        return res

    def get_stats(self, user_id=None, agent_id=None, session_id=None) -> Dict[str, Any]:
        filtered = self.get_all(user_id)
        modality_counts = {k: len(v) for k, v in self.modality_index.items()}
        return {
            "count": len(filtered),
            "perceptions_count": len(self.perceptions),
            "modality_counts": modality_counts,
            "supported_modalities": list(self.supported_modalities),
            "avg_importance": sum(m.importance for m in filtered) / len(filtered) if filtered else 0.0,
            "memory_type": "perceptual"
        }

    # =========================================================================
    # 原有逻辑保持不变
    # =========================================================================
    def cross_modal_search(self, query: Any, query_modality: str, target_modality=None, limit=5):
        return self.retrieve(
            query=str(query), limit=limit,
            query_modality=query_modality, target_modality=target_modality
        )

    def get_by_modality(self, modality: str, limit=10, user_id=None, agent_id=None, session_id=None):
        if modality not in self.modality_index:
            return []
        res = []
        for m in self.perceptual_memories:
            if m.metadata.get("modality") != modality:
                continue
            if user_id and m.user_id != user_id:
                continue
            res.append(m)
            if len(res) >= limit:
                break
        return res

    def generate_content(self, prompt: str, target_modality: str):
        if target_modality not in self.supported_modalities:
            return None
        rel = self.retrieve(prompt, limit=3)
        if not rel:
            return None
        if target_modality == "text":
            return "生成文本：\n" + "\n".join(m.content for m in rel)
        return f"生成{target_modality}内容"

    def _init_encoders(self):
        return {
            "text": self._text_encoder,
            "image": self._image_encoder,
            "audio": self._audio_encoder,
            "default": self._default_encoder
        }

    def _encode_perception(self, data, modality, memory_id):
        enc = self._encode_data(data, modality)
        return Perception(f"perception_{memory_id}", data, modality, enc)

    def _encode_data(self, data, modality):
        dim = self._get_dim_for_modality(modality)
        vec = self.encoders.get(modality, self._default_encoder)(data)
        vec = list(vec) if isinstance(vec, (list, tuple)) else [float(x) for x in vec]
        if len(vec) < dim:
            vec += [0.0] * (dim - len(vec))
        return vec[:dim]

    def _text_encoder(self, text):
        emb = self.text_embedder.encode(text or "")
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def _image_encoder_hash(self, img):
        try:
            b = img if isinstance(img, bytes) else (open(img, 'rb').read() if isinstance(img, str) and os.path.exists(img) else str(img).encode())
            h = hashlib.sha256(b).hexdigest()
            return self._hash_to_vector(h, self._get_dim_for_modality("image"))
        except Exception:
            return self._hash_to_vector(str(img), self._get_dim_for_modality("image"))

    def _image_encoder(self, data):
        if not self._clip_model:
            return self._image_encoder_hash(data)
        try:
            from PIL import Image
            im = Image.open(data).convert('RGB') if isinstance(data, str) else Image.open(data).convert('RGB')
            inputs = self._clip_processor(images=im, return_tensors="pt")
            with self._no_grad():
                f = self._clip_model.get_image_features(**inputs)
            return f[0].detach().cpu().numpy().tolist()
        except Exception:
            return self._image_encoder_hash(data)

    def _audio_encoder_hash(self, audio):
        try:
            b = audio if isinstance(audio, bytes) else (open(audio, 'rb').read() if isinstance(audio, str) and os.path.exists(audio) else str(audio).encode())
            h = hashlib.sha256(b).hexdigest()
            return self._hash_to_vector(h, self._get_dim_for_modality("audio"))
        except Exception:
            return self._hash_to_vector(str(audio), self._get_dim_for_modality("audio"))

    def _audio_encoder(self, data):
        if not self._clap_model:
            return self._audio_encoder_hash(data)
        try:
            import librosa
            w, _ = librosa.load(data, sr=48000, mono=True)
            inputs = self._clap_processor(audios=w, sampling_rate=48000, return_tensors="pt")
            with self._no_grad():
                f = self._clap_model.get_audio_features(**inputs)
            return f[0].detach().cpu().numpy().tolist()
        except Exception:
            return self._audio_encoder_hash(data)

    def _default_encoder(self, data):
        try:
            return self._text_encoder(str(data))
        except Exception:
            return self._hash_to_vector(str(data), self.vector_dim)

    def _hash_to_vector(self, s, dim):
        rng = random.Random(int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2 ** 32)
        return [rng.random() for _ in range(dim)]

    class _no_grad:
        def __enter__(self):
            try:
                import torch
                self.prev = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
            except:
                self.prev = None
            return self

        def __exit__(self, *args):
            try:
                import torch
                torch.set_grad_enabled(self.prev)
            except:
                pass

    def _get_vector_store_for_modality(self, m):
        return self.vector_stores.get((m or "text").lower(), self.vector_stores["text"])

    def _get_dim_for_modality(self, m):
        m = (m or "text").lower()
        if m == "image": return int(self._image_dim or self.vector_dim)
        if m == "audio": return int(self._audio_dim or self.vector_dim)
        return int(self.vector_dim)