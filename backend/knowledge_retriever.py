"""外部知识库RAG检索模块

保持与现有记忆链路解耦:
- Memory retrieval 继续由 HelloAgents MemoryManager 负责
- Knowledge retrieval 只读取 knowledge_base 文档并写入独立 Qdrant collection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import re

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from config import settings


@dataclass
class KnowledgeChunk:
    """单条知识块"""

    point_id: str
    content: str
    source: str
    title: str
    scope: str
    tags: List[str]
    chunk_index: int
    score: float = 0.0

    def to_payload(self) -> Dict:
        """转换为Qdrant payload"""
        return {
            "content": self.content,
            "source": self.source,
            "title": self.title,
            "scope": self.scope,
            "tags": self.tags,
            "chunk_index": self.chunk_index,
        }

    def to_dict(self) -> Dict:
        """转换为可序列化结构"""
        return {
            "id": self.point_id,
            "content": self.content,
            "source": self.source,
            "title": self.title,
            "scope": self.scope,
            "tags": self.tags,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
        }


class KnowledgeRetriever:
    """独立的外部知识检索器"""

    SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

    def __init__(self):
        self.enabled = settings.KNOWLEDGE_ENABLED
        self.base_dir = Path(settings.KNOWLEDGE_BASE_DIR)
        self.collection_name = settings.KNOWLEDGE_COLLECTION
        self.top_k = settings.KNOWLEDGE_TOP_K
        self.chunk_size = settings.KNOWLEDGE_CHUNK_SIZE
        self.chunk_overlap = settings.KNOWLEDGE_CHUNK_OVERLAP

        self._client: Optional[QdrantClient] = None
        self._embedder: Optional[SentenceTransformer] = None

        if not self.enabled:
            return

        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_TIMEOUT,
            check_compatibility=False,
            trust_env=False,
        )

    def available(self) -> bool:
        """知识检索器是否可用"""
        return self.enabled and self._client is not None

    def _get_embedder(self) -> SentenceTransformer:
        """延迟加载embedding模型"""
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.EMBED_MODEL_NAME)
        return self._embedder

    def _ensure_collection(self):
        """确保知识库集合存在"""
        if not self._client:
            raise RuntimeError("Qdrant client 未初始化")

        if self._client.collection_exists(self.collection_name):
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=settings.QDRANT_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    def _iter_documents(self) -> List[Path]:
        """遍历知识文档"""
        if not self.base_dir.exists():
            return []

        docs = []
        for path in self.base_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_SUFFIXES:
                docs.append(path)
        return sorted(docs)

    def ingest(self) -> Dict[str, int]:
        """读取知识库目录并写入Qdrant"""
        if not self.available():
            raise RuntimeError("知识检索器不可用")

        self._ensure_collection()

        chunks: List[KnowledgeChunk] = []
        for path in self._iter_documents():
            chunks.extend(self._load_document_chunks(path))

        if not chunks:
            return {"documents": 0, "chunks": 0}

        vectors = self._get_embedder().encode(
            [chunk.content for chunk in chunks],
            normalize_embeddings=True,
        )

        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                models.PointStruct(
                    id=chunk.point_id,
                    vector=vector.tolist(),
                    payload=chunk.to_payload(),
                )
            )

        self._client.upsert(collection_name=self.collection_name, points=points)
        return {"documents": len(self._iter_documents()), "chunks": len(chunks)}

    def search(self, query: str, limit: Optional[int] = None, scope: str = "global") -> List[KnowledgeChunk]:
        """检索外部知识块"""
        if not self.available() or not query.strip():
            return []

        try:
            if not self._client.collection_exists(self.collection_name):
                return []

            vector = self._get_embedder().encode(query, normalize_embeddings=True).tolist()
            search_limit = limit or self.top_k
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="scope",
                        match=models.MatchValue(value=scope),
                    )
                ]
            )

            response = self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=search_limit,
                with_payload=True,
                query_filter=search_filter,
            )
            hits = response.points

            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    KnowledgeChunk(
                        point_id=str(hit.id),
                        content=payload.get("content", ""),
                        source=payload.get("source", ""),
                        title=payload.get("title", "未知文档"),
                        scope=payload.get("scope", "global"),
                        tags=payload.get("tags", []),
                        chunk_index=payload.get("chunk_index", 0),
                        score=float(hit.score or 0.0),
                    )
                )
            return results
        except Exception as e:
            print(f"⚠️  外部知识检索失败，已自动降级: {e}")
            return []

    def build_prompt_context(
        self,
        query: str,
        chunks: List[KnowledgeChunk],
        max_chars_per_chunk: int = 220,
        total_budget: int = 360,
    ) -> str:
        """构建注入到prompt的知识上下文

        只注入命中片段，而不是整段正文，避免 external knowledge 挤爆 prompt。
        """
        if not chunks:
            return ""

        lines = ["【外部知识】"]
        used_chars = 0
        for chunk in chunks:
            snippet = self._extract_hit_snippet(
                query=query,
                content=chunk.content,
                max_chars=max_chars_per_chunk,
            )
            if not snippet:
                continue
            remaining = total_budget - used_chars
            if remaining <= 0:
                break
            if len(snippet) > remaining:
                snippet = snippet[:remaining].rstrip()
            lines.append(
                f"[{chunk.title} | score={chunk.score:.3f}] {snippet}"
            )
            used_chars += len(snippet)
        return "\n".join(lines)

    def _extract_hit_snippet(self, query: str, content: str, max_chars: int = 220) -> str:
        """根据查询词提取命中片段"""
        text = re.sub(r"\s+", " ", content).strip()
        if len(text) <= max_chars:
            return text

        keywords = self._extract_query_keywords(query)
        lowered = text.lower()

        best_index = -1
        best_keyword = ""
        for keyword in keywords:
            index = lowered.find(keyword.lower())
            if index != -1 and len(keyword) > len(best_keyword):
                best_index = index
                best_keyword = keyword

        if best_index == -1:
            return text[:max_chars].rstrip()

        half_window = max_chars // 2
        start = max(best_index - half_window, 0)
        end = min(start + max_chars, len(text))
        if end - start < max_chars:
            start = max(end - max_chars, 0)

        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    def _extract_query_keywords(self, query: str) -> List[str]:
        """提取查询中的关键短语"""
        tokens = [
            token.strip()
            for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", query)
            if token.strip()
        ]
        # 优先保留较长关键词，提升中文片段命中质量
        return sorted(set(tokens), key=len, reverse=True)

    def _load_document_chunks(self, path: Path) -> List[KnowledgeChunk]:
        """读取并切块单个文档"""
        raw = path.read_text(encoding="utf-8")
        metadata, body = self._parse_frontmatter(raw)
        title = metadata.get("title") or self._extract_title(path, body)
        scope = metadata.get("scope", "global")
        tags = self._normalize_tags(metadata.get("tags", []))
        source = str(path.relative_to(self.base_dir))

        paragraphs = self._split_paragraphs(body)
        chunk_texts = self._chunk_paragraphs(paragraphs)

        chunks = []
        for index, content in enumerate(chunk_texts):
            point_id = self._make_point_id(source, index, content)
            chunks.append(
                KnowledgeChunk(
                    point_id=point_id,
                    content=content,
                    source=source,
                    title=title,
                    scope=scope,
                    tags=tags,
                    chunk_index=index,
                )
            )
        return chunks

    def _parse_frontmatter(self, raw: str) -> tuple[Dict, str]:
        """解析简化frontmatter"""
        if not raw.startswith("---\n"):
            return {}, raw

        end_index = raw.find("\n---\n", 4)
        if end_index == -1:
            return {}, raw

        frontmatter_text = raw[4:end_index]
        body = raw[end_index + 5 :]
        metadata: Dict[str, object] = {}
        for line in frontmatter_text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "tags":
                metadata[key] = [item.strip() for item in value.split(",") if item.strip()]
            else:
                metadata[key] = value
        return metadata, body

    def _extract_title(self, path: Path, body: str) -> str:
        """提取文档标题"""
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return path.stem

    def _split_paragraphs(self, body: str) -> List[str]:
        """按段落切分正文"""
        cleaned = body.replace("\r\n", "\n").strip()
        if not cleaned:
            return []
        return [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]

    def _chunk_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """按段落聚合为知识块"""
        if not paragraphs:
            return []

        chunks: List[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._slice_long_text(paragraph))
                continue

            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current.strip())
                current = paragraph

        if current:
            chunks.append(current.strip())
        return chunks

    def _slice_long_text(self, text: str) -> List[str]:
        """处理超长段落"""
        if len(text) <= self.chunk_size:
            return [text]

        stride = max(self.chunk_size - self.chunk_overlap, 1)
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            start += stride
        return [chunk for chunk in chunks if chunk]

    def _normalize_tags(self, tags: object) -> List[str]:
        """统一tags格式"""
        if isinstance(tags, list):
            return [str(tag).strip() for tag in tags if str(tag).strip()]
        if isinstance(tags, str):
            return [tag.strip() for tag in tags.split(",") if tag.strip()]
        return []

    def _make_point_id(self, source: str, chunk_index: int, content: str) -> str:
        """构造稳定point id，避免重复ingest时产生大量重复数据"""
        raw = f"{source}:{chunk_index}:{content}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
