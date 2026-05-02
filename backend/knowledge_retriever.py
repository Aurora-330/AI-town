"""外部知识库RAG检索模块

保持与现有记忆链路解耦:
- Memory retrieval 继续由 HelloAgents MemoryManager 负责
- Knowledge retrieval 只读取 knowledge_base 文档并写入独立 Qdrant collection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    raw_score: float = 0.0
    rerank_score: float = 0.0
    debug_signals: Dict[str, object] = field(default_factory=dict)

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
            "raw_score": round(self.raw_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "debug_signals": self.debug_signals,
        }


class KnowledgeRetriever:
    """独立的外部知识检索器"""

    SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
    KNOWN_NPCS = ("风泠", "郁米", "顾辰")
    CANDIDATE_MULTIPLIER = 4
    MIN_RERANK_SCORE = 0.22
    CROSS_NPC_FILTER_THRESHOLD = 0.28

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

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        scope: str = "global",
        npc_name: str = "",
        allow_cross_npc: bool = False,
    ) -> List[KnowledgeChunk]:
        """兼容旧接口，只返回结果列表。"""
        results, _ = self.search_with_debug(
            query=query,
            limit=limit,
            scope=scope,
            npc_name=npc_name,
            allow_cross_npc=allow_cross_npc,
        )
        return results

    def search_with_debug(
        self,
        query: str,
        limit: Optional[int] = None,
        scope: str = "global",
        npc_name: str = "",
        allow_cross_npc: bool = False,
    ) -> Tuple[List[KnowledgeChunk], Dict]:
        """检索外部知识块"""
        if not self.available() or not query.strip():
            return [], {
                "scope": scope,
                "limit": limit or self.top_k,
                "candidate_count": 0,
                "selected_count": 0,
                "selected_or_filtered_reason": "retriever_unavailable_or_empty_query",
                "candidates": [],
            }

        try:
            if not self._client.collection_exists(self.collection_name):
                return [], {
                    "scope": scope,
                    "limit": limit or self.top_k,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "selected_or_filtered_reason": "collection_missing",
                    "candidates": [],
                }

            vector = self._get_embedder().encode(query, normalize_embeddings=True).tolist()
            result_limit = limit or self.top_k
            search_limit = max(result_limit * self.CANDIDATE_MULTIPLIER, result_limit)
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

            candidates = []
            for hit in hits:
                payload = hit.payload or {}
                raw_score = float(hit.score or 0.0)
                candidates.append(
                    KnowledgeChunk(
                        point_id=str(hit.id),
                        content=payload.get("content", ""),
                        source=payload.get("source", ""),
                        title=payload.get("title", "未知文档"),
                        scope=payload.get("scope", "global"),
                        tags=payload.get("tags", []),
                        chunk_index=payload.get("chunk_index", 0),
                        score=raw_score,
                        raw_score=raw_score,
                    )
                )
            selected, debug_info = self._rerank_and_filter_chunks(
                query=query,
                npc_name=npc_name,
                candidates=candidates,
                limit=result_limit,
                scope=scope,
                allow_cross_npc=allow_cross_npc,
            )
            return selected, debug_info
        except Exception as e:
            print(f"⚠️  外部知识检索失败，已自动降级: {e}")
            return [], {
                "scope": scope,
                "limit": limit or self.top_k,
                "candidate_count": 0,
                "selected_count": 0,
                "selected_or_filtered_reason": f"search_failed:{e}",
                "candidates": [],
            }

    def _rerank_and_filter_chunks(
        self,
        query: str,
        npc_name: str,
        candidates: List[KnowledgeChunk],
        limit: int,
        scope: str,
        allow_cross_npc: bool,
    ) -> Tuple[List[KnowledgeChunk], Dict]:
        """对候选知识块做轻量重排与低价值过滤。"""
        seen_dedup_keys = set()
        reranked: List[KnowledgeChunk] = []
        candidate_logs: List[Dict] = []
        query_keywords = self._extract_query_keywords(query)

        for chunk in candidates:
            final_score, signals = self._score_chunk(
                query=query,
                query_keywords=query_keywords,
                npc_name=npc_name,
                chunk=chunk,
                allow_cross_npc=allow_cross_npc,
            )
            chunk.rerank_score = final_score
            chunk.score = final_score
            chunk.debug_signals = signals

            dedup_key = (chunk.source, self._normalize_text(chunk.content)[:120])
            filtered_reason = self._decide_filter_reason(
                chunk=chunk,
                final_score=final_score,
                signals=signals,
                allow_cross_npc=allow_cross_npc,
                seen_dedup_keys=seen_dedup_keys,
                dedup_key=dedup_key,
            )

            candidate_logs.append(
                {
                    "id": chunk.point_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "raw_score": round(chunk.raw_score, 4),
                    "rerank_score": round(final_score, 4),
                    "signals": signals,
                    "filtered_reason": filtered_reason or "",
                }
            )

            if filtered_reason:
                continue

            seen_dedup_keys.add(dedup_key)
            reranked.append(chunk)

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        selected = reranked[:limit]

        return selected, {
            "scope": scope,
            "limit": limit,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_or_filtered_reason": "reranked_and_filtered",
            "candidates": candidate_logs,
        }

    def _score_chunk(
        self,
        query: str,
        query_keywords: List[str],
        npc_name: str,
        chunk: KnowledgeChunk,
        allow_cross_npc: bool,
    ) -> Tuple[float, Dict[str, object]]:
        """综合角色相关性、关键词命中和原始向量分数。"""
        normalized_title = self._normalize_text(chunk.title)
        normalized_content = self._normalize_text(chunk.content)
        normalized_source = self._normalize_text(chunk.source)
        tag_text = self._normalize_text(" ".join(chunk.tags))

        title_hits = sum(1 for keyword in query_keywords if keyword and keyword in normalized_title)
        content_hits = sum(1 for keyword in query_keywords if keyword and keyword in normalized_content)
        tag_hits = sum(1 for keyword in query_keywords if keyword and keyword in tag_text)

        mentioned_npcs = [
            name for name in self.KNOWN_NPCS
            if name in chunk.content or name in chunk.title or name in chunk.source or name in chunk.tags
        ]
        other_npcs = [name for name in mentioned_npcs if name != npc_name]

        npc_match_bonus = 0.0
        other_npc_penalty = 0.0
        if npc_name and npc_name in mentioned_npcs:
            npc_match_bonus += 0.18
        if npc_name and other_npcs and npc_name not in mentioned_npcs and not allow_cross_npc:
            other_npc_penalty -= min(0.18 * len(other_npcs), 0.36)
        elif npc_name and other_npcs and npc_name in mentioned_npcs:
            other_npc_penalty -= min(0.05 * len(other_npcs), 0.10)

        generic_bonus = 0.03 if not mentioned_npcs else 0.0
        title_bonus = min(title_hits * 0.05, 0.15)
        content_bonus = min(content_hits * 0.025, 0.15)
        tag_bonus = min(tag_hits * 0.03, 0.09)

        final_score = (
            chunk.raw_score
            + npc_match_bonus
            + other_npc_penalty
            + generic_bonus
            + title_bonus
            + content_bonus
            + tag_bonus
        )

        return final_score, {
            "npc_name": npc_name,
            "title_hits": title_hits,
            "content_hits": content_hits,
            "tag_hits": tag_hits,
            "mentioned_npcs": mentioned_npcs,
            "npc_match_bonus": round(npc_match_bonus, 4),
            "other_npc_penalty": round(other_npc_penalty, 4),
            "generic_bonus": round(generic_bonus, 4),
            "title_bonus": round(title_bonus, 4),
            "content_bonus": round(content_bonus, 4),
            "tag_bonus": round(tag_bonus, 4),
        }

    def _decide_filter_reason(
        self,
        chunk: KnowledgeChunk,
        final_score: float,
        signals: Dict[str, object],
        allow_cross_npc: bool,
        seen_dedup_keys: set,
        dedup_key: tuple,
    ) -> str:
        """决定候选知识块是否应被过滤。"""
        if dedup_key in seen_dedup_keys:
            return "duplicate_source_or_content"

        mentioned_npcs = signals.get("mentioned_npcs", [])
        npc_name = signals.get("npc_name", "")
        if (
            final_score < self.CROSS_NPC_FILTER_THRESHOLD
            and npc_name
            and mentioned_npcs
            and npc_name not in mentioned_npcs
            and not allow_cross_npc
        ):
            return "cross_npc_mismatch"

        if final_score < self.MIN_RERANK_SCORE:
            return "score_too_low"

        return ""

    def build_prompt_context(
        self,
        query: str,
        chunks: List[KnowledgeChunk],
        npc_name: str = "",
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
                npc_name=npc_name,
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

    def _extract_hit_snippet(
        self,
        query: str,
        content: str,
        npc_name: str = "",
        max_chars: int = 220,
    ) -> str:
        """根据查询词提取命中片段"""
        text = re.sub(r"\s+", " ", content).strip()
        if len(text) <= max_chars:
            return self._prefer_npc_focused_text(content=content, fallback_text=text, npc_name=npc_name, max_chars=max_chars)

        npc_focused = self._extract_npc_focused_section(content=content, npc_name=npc_name)
        if npc_focused:
            focused_text = re.sub(r"\s+", " ", npc_focused).strip()
            if len(focused_text) <= max_chars:
                return focused_text
            text = focused_text

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

    def _extract_npc_focused_section(self, content: str, npc_name: str) -> str:
        """从多角色文档里优先抽取当前 NPC 对应段落。"""
        if not npc_name:
            return ""

        paragraphs = self._split_paragraphs(content)
        if not paragraphs:
            return ""

        matched = [paragraph for paragraph in paragraphs if npc_name in paragraph]
        if matched:
            return "\n\n".join(matched)
        return ""

    def _prefer_npc_focused_text(self, content: str, fallback_text: str, npc_name: str, max_chars: int) -> str:
        """短文本场景下也优先截取当前 NPC 对应段落。"""
        npc_focused = self._extract_npc_focused_section(content, npc_name)
        chosen = npc_focused or fallback_text
        if len(chosen) <= max_chars:
            return chosen
        return chosen[:max_chars].rstrip()

    def _extract_query_keywords(self, query: str) -> List[str]:
        """提取查询中的关键短语"""
        tokens = [
            token.strip()
            for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", query)
            if token.strip()
        ]
        # 优先保留较长关键词，提升中文片段命中质量
        return sorted(set(tokens), key=len, reverse=True)

    def _normalize_text(self, text: object) -> str:
        """统一做轻量文本归一化，便于命中判断。"""
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

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
