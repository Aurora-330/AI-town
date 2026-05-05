"""外部知识库RAG检索模块

保持与现有记忆链路解耦:
- Memory retrieval 继续由 HelloAgents MemoryManager 负责
- Knowledge retrieval 只读取 knowledge_base 文档并写入独立 Qdrant collection
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import log
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
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_sources: List[str] = field(default_factory=list)
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
            "semantic_score": round(self.semantic_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "retrieval_sources": self.retrieval_sources,
            "debug_signals": self.debug_signals,
        }


class KnowledgeRetriever:
    """独立的外部知识检索器"""

    SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
    KNOWN_NPCS = ("风泠", "郁米", "顾辰")
    CANDIDATE_MULTIPLIER = 4
    MIN_RERANK_SCORE = 0.22
    CROSS_NPC_FILTER_THRESHOLD = 0.28
    BM25_K1 = 1.5
    BM25_B = 0.75
    DOC_REFERENCE_MARKERS = ("里", "里面", "文档", "手册", "说明", "示例", "怎么写", "主要写了什么", "定义")
    QUERY_SUFFIX_HINTS = (
        ("发生在什么时候", ("时间",)),
        ("是在什么时候", ("时间",)),
        ("位于哪里", ("位置",)),
        ("位置在哪里", ("位置",)),
        ("在什么位置", ("位置",)),
        ("在哪里", ("位置",)),
        ("在哪", ("位置",)),
        ("是什么时候", ("时间",)),
        ("什么时候", ("时间",)),
        ("几点", ("时间",)),
        ("何时", ("时间",)),
        ("喜欢什么", ("喜欢",)),
        ("喜欢哪些", ("喜欢",)),
        ("喜欢", ("喜欢",)),
        ("是谁", ("人物",)),
        ("谁", ()),
        ("什么", ()),
        ("多少", ()),
        ("如何", ()),
        ("怎么", ()),
    )
    QUERY_TRAILING_VERBS = ("喜欢", "位于", "位置", "负责", "管理", "主理", "时间", "特点", "规则", "节日")
    QUERY_SHORT_KEYWORDS = {"位置", "时间", "喜欢", "规则", "节日", "人物"}
    QUERY_TAIL_PATTERNS = (
        "发生在什么时候",
        "是在什么时候",
        "是什么时候",
        "什么时候",
        "位于哪里",
        "位置在哪里",
        "在什么位置",
        "在哪里",
        "在哪",
        "喜欢什么",
        "喜欢哪些",
        "由谁负责",
        "谁负责",
        "谁解决的",
        "由谁解决",
        "是谁",
        "是什么",
        "多少",
        "几点",
        "何时",
        "如何",
        "怎么",
    )
    QUERY_LEADING_FILLERS = (
        "请问",
        "想问",
        "我想问",
        "麻烦问下",
        "麻烦问一下",
        "帮我看看",
        "帮我查查",
        "顺便说下",
    )

    def __init__(self):
        self.enabled = settings.KNOWLEDGE_ENABLED
        self.base_dir = Path(settings.KNOWLEDGE_BASE_DIR)
        self.collection_name = settings.KNOWLEDGE_COLLECTION
        self.top_k = settings.KNOWLEDGE_TOP_K
        self.chunk_size = settings.KNOWLEDGE_CHUNK_SIZE
        self.chunk_overlap = settings.KNOWLEDGE_CHUNK_OVERLAP

        self._client: Optional[QdrantClient] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._lexical_chunks: List[KnowledgeChunk] = []
        self._lexical_chunk_map: Dict[str, KnowledgeChunk] = {}
        self._lexical_tokens_by_chunk: Dict[str, List[str]] = {}
        self._lexical_tf_by_chunk: Dict[str, Counter[str]] = {}
        self._lexical_df: Dict[str, int] = {}
        self._lexical_avg_doc_len: float = 0.0

        if not self.enabled:
            return

        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_TIMEOUT,
            check_compatibility=False,
            trust_env=False,
        )
        self._build_lexical_index()

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
            self._build_lexical_index()
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
        self._build_lexical_index(chunks)
        return {"documents": len(self._iter_documents()), "chunks": len(chunks)}

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        scope: str = "global",
        npc_name: str = "",
        allow_cross_npc: bool = False,
        scopes: Optional[List[str]] = None,
    ) -> List[KnowledgeChunk]:
        """兼容旧接口，只返回结果列表。"""
        results, _ = self.search_with_debug(
            query=query,
            limit=limit,
            scope=scope,
            npc_name=npc_name,
            allow_cross_npc=allow_cross_npc,
            scopes=scopes,
        )
        return results

    def search_with_debug(
        self,
        query: str,
        limit: Optional[int] = None,
        scope: str = "global",
        npc_name: str = "",
        allow_cross_npc: bool = False,
        scopes: Optional[List[str]] = None,
    ) -> Tuple[List[KnowledgeChunk], Dict]:
        """检索外部知识块"""
        requested_scopes = self._normalize_scopes(scopes or [scope])
        result_limit = limit or self.top_k
        if not self.available() or not query.strip():
            return [], {
                "scope": requested_scopes[0] if requested_scopes else scope,
                "scopes": requested_scopes,
                "limit": result_limit,
                "candidate_count": 0,
                "selected_count": 0,
                "selected_or_filtered_reason": "retriever_unavailable_or_empty_query",
                "semantic_candidate_count": 0,
                "lexical_candidate_count": 0,
                "candidates": [],
            }

        try:
            search_limit = max(result_limit * self.CANDIDATE_MULTIPLIER, result_limit)
            semantic_candidates: List[KnowledgeChunk] = []
            collection_exists = self._client.collection_exists(self.collection_name)
            if collection_exists:
                semantic_candidates = self._semantic_search(
                    query=query,
                    requested_scopes=requested_scopes,
                    search_limit=search_limit,
                )
            lexical_candidates = self._lexical_search(
                query=query,
                requested_scopes=requested_scopes,
                search_limit=search_limit,
            )
            candidates = self._merge_candidates(semantic_candidates, lexical_candidates)
            selected, debug_info = self._rerank_and_filter_chunks(
                query=query,
                npc_name=npc_name,
                candidates=candidates,
                limit=result_limit,
                scope=requested_scopes[0] if requested_scopes else scope,
                scopes=requested_scopes,
                allow_cross_npc=allow_cross_npc,
                semantic_candidate_count=len(semantic_candidates),
                lexical_candidate_count=len(lexical_candidates),
            )
            if not collection_exists:
                debug_info["selected_or_filtered_reason"] = "collection_missing_lexical_only"
            return selected, debug_info
        except Exception as e:
            print(f"⚠️  外部知识检索失败，已自动降级: {e}")
            return [], {
                "scope": requested_scopes[0] if requested_scopes else scope,
                "scopes": requested_scopes,
                "limit": result_limit,
                "candidate_count": 0,
                "selected_count": 0,
                "selected_or_filtered_reason": f"search_failed:{e}",
                "semantic_candidate_count": 0,
                "lexical_candidate_count": 0,
                "candidates": [],
            }

    def _rerank_and_filter_chunks(
        self,
        query: str,
        npc_name: str,
        candidates: List[KnowledgeChunk],
        limit: int,
        scope: str,
        scopes: List[str],
        allow_cross_npc: bool,
        semantic_candidate_count: int = 0,
        lexical_candidate_count: int = 0,
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
                    "semantic_score": round(chunk.semantic_score, 4),
                    "lexical_score": round(chunk.lexical_score, 4),
                    "rerank_score": round(final_score, 4),
                    "retrieval_sources": chunk.retrieval_sources,
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
            "scopes": scopes,
            "limit": limit,
            "candidate_count": len(candidates),
            "semantic_candidate_count": semantic_candidate_count,
            "lexical_candidate_count": lexical_candidate_count,
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
        query_text = self._normalize_text(query)

        source_aliases = self._extract_source_aliases(chunk.source)
        source_hits = sum(1 for alias in source_aliases if alias and alias in query_text)
        source_exact_match = any(alias and alias in query_text for alias in source_aliases)
        explicit_doc_reference = self._is_explicit_doc_reference(query)
        lexical_dominant = chunk.lexical_score >= max(chunk.semantic_score + 0.08, 0.45)

        title_hits = sum(1 for keyword in query_keywords if keyword and keyword in normalized_title)
        content_hits = sum(1 for keyword in query_keywords if keyword and keyword in normalized_content)
        tag_hits = sum(1 for keyword in query_keywords if keyword and keyword in tag_text)

        mentioned_npcs = [
            name for name in self.KNOWN_NPCS
            if name in chunk.content or name in chunk.title or name in chunk.source or name in chunk.tags
        ]
        target_npcs = [name for name in self.KNOWN_NPCS if name in query]
        query_targets_other_npc = bool(target_npcs and npc_name not in target_npcs)
        other_npcs = [name for name in mentioned_npcs if name != npc_name]

        npc_match_bonus = 0.0
        other_npc_penalty = 0.0
        target_npc_bonus = 0.0
        target_npc_penalty = 0.0
        if npc_name and npc_name in mentioned_npcs:
            npc_match_bonus += 0.18
        if npc_name and other_npcs and npc_name not in mentioned_npcs and not allow_cross_npc:
            other_npc_penalty -= min(0.18 * len(other_npcs), 0.36)
        elif npc_name and other_npcs and npc_name in mentioned_npcs:
            other_npc_penalty -= min(0.05 * len(other_npcs), 0.10)

        if target_npcs:
            if any(name in mentioned_npcs for name in target_npcs):
                target_npc_bonus += 0.24
            elif query_targets_other_npc and npc_name and npc_name in mentioned_npcs:
                target_npc_penalty -= 0.24

        generic_bonus = 0.03 if not mentioned_npcs else 0.0
        scope_bonus = 0.12 if chunk.scope.startswith("npc:") and npc_name and chunk.scope == f"npc:{npc_name}" else 0.0
        if query_targets_other_npc and chunk.scope == f"npc:{npc_name}":
            scope_bonus -= 0.16
        title_bonus = min(title_hits * 0.05, 0.15)
        content_bonus = min(content_hits * 0.025, 0.15)
        tag_bonus = min(tag_hits * 0.03, 0.09)
        source_bonus = min(source_hits * 0.12, 0.36)
        lexical_bonus = min(chunk.lexical_score * 0.42, 0.42)
        explicit_doc_bonus = 0.28 if explicit_doc_reference and source_exact_match else 0.0
        lexical_dominant_bonus = 0.12 if lexical_dominant else 0.0
        lexical_only_bonus = 0.06 if chunk.retrieval_sources == ["lexical"] else 0.0

        final_score = (
            chunk.raw_score
            + scope_bonus
            + npc_match_bonus
            + other_npc_penalty
            + target_npc_bonus
            + target_npc_penalty
            + generic_bonus
            + title_bonus
            + content_bonus
            + tag_bonus
            + source_bonus
            + lexical_bonus
            + explicit_doc_bonus
            + lexical_dominant_bonus
            + lexical_only_bonus
        )

        return final_score, {
            "npc_name": npc_name,
            "semantic_score": round(chunk.semantic_score, 4),
            "lexical_score": round(chunk.lexical_score, 4),
            "retrieval_sources": chunk.retrieval_sources,
            "source_hits": source_hits,
            "source_exact_match": source_exact_match,
            "explicit_doc_reference": explicit_doc_reference,
            "lexical_dominant": lexical_dominant,
            "scope_bonus": round(scope_bonus, 4),
            "title_hits": title_hits,
            "content_hits": content_hits,
            "tag_hits": tag_hits,
            "mentioned_npcs": mentioned_npcs,
            "target_npcs": target_npcs,
            "npc_match_bonus": round(npc_match_bonus, 4),
            "other_npc_penalty": round(other_npc_penalty, 4),
            "target_npc_bonus": round(target_npc_bonus, 4),
            "target_npc_penalty": round(target_npc_penalty, 4),
            "generic_bonus": round(generic_bonus, 4),
            "title_bonus": round(title_bonus, 4),
            "content_bonus": round(content_bonus, 4),
            "tag_bonus": round(tag_bonus, 4),
            "source_bonus": round(source_bonus, 4),
            "lexical_bonus": round(lexical_bonus, 4),
            "explicit_doc_bonus": round(explicit_doc_bonus, 4),
            "lexical_dominant_bonus": round(lexical_dominant_bonus, 4),
            "lexical_only_bonus": round(lexical_only_bonus, 4),
        }

    def _semantic_search(
        self,
        query: str,
        requested_scopes: List[str],
        search_limit: int,
    ) -> List[KnowledgeChunk]:
        """执行现有向量检索主链。"""
        vector = self._get_embedder().encode(query, normalize_embeddings=True).tolist()
        candidates: List[KnowledgeChunk] = []

        for current_scope in requested_scopes:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="scope",
                        match=models.MatchValue(value=current_scope),
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
            for hit in response.points:
                payload = hit.payload or {}
                raw_score = float(hit.score or 0.0)
                candidates.append(
                    KnowledgeChunk(
                        point_id=str(hit.id),
                        content=payload.get("content", ""),
                        source=payload.get("source", ""),
                        title=payload.get("title", "未知文档"),
                        scope=payload.get("scope", current_scope),
                        tags=payload.get("tags", []),
                        chunk_index=payload.get("chunk_index", 0),
                        score=raw_score,
                        raw_score=raw_score,
                        semantic_score=raw_score,
                        retrieval_sources=["semantic"],
                    )
                )
        return candidates

    def _lexical_search(
        self,
        query: str,
        requested_scopes: List[str],
        search_limit: int,
    ) -> List[KnowledgeChunk]:
        """执行轻量 BM25 lexical 检索。"""
        self._ensure_lexical_index()
        query_terms = self._tokenize_query_for_lexical(query)
        if not query_terms or not self._lexical_chunks:
            return []

        scope_set = set(requested_scopes)
        scoped_chunks = [chunk for chunk in self._lexical_chunks if chunk.scope in scope_set]
        if not scoped_chunks:
            return []

        scored_chunks: List[Tuple[float, KnowledgeChunk]] = []
        for chunk in scoped_chunks:
            tokens = self._lexical_tokens_by_chunk.get(chunk.point_id, [])
            if not tokens:
                continue
            score = self._compute_bm25_score(
                query_terms=query_terms,
                term_freqs=self._lexical_tf_by_chunk.get(chunk.point_id, Counter()),
                doc_len=len(tokens),
            )
            if score <= 0:
                continue
            lexical_chunk = KnowledgeChunk(
                point_id=chunk.point_id,
                content=chunk.content,
                source=chunk.source,
                title=chunk.title,
                scope=chunk.scope,
                tags=list(chunk.tags),
                chunk_index=chunk.chunk_index,
                score=0.0,
                raw_score=0.0,
                semantic_score=0.0,
                lexical_score=score,
                retrieval_sources=["lexical"],
            )
            scored_chunks.append((score, lexical_chunk))

        if not scored_chunks:
            return []

        max_score = max(score for score, _ in scored_chunks) or 1.0
        normalized = []
        for score, chunk in scored_chunks:
            chunk.lexical_score = score / max_score
            normalized.append(chunk)

        normalized.sort(key=lambda item: item.lexical_score, reverse=True)
        return normalized[:search_limit]

    def _merge_candidates(
        self,
        semantic_candidates: List[KnowledgeChunk],
        lexical_candidates: List[KnowledgeChunk],
    ) -> List[KnowledgeChunk]:
        """合并 semantic 与 lexical 召回结果。"""
        merged: Dict[str, KnowledgeChunk] = {}

        for chunk in semantic_candidates + lexical_candidates:
            normalized_point_id = self._normalize_point_id(chunk.point_id)
            chunk.point_id = normalized_point_id
            existing = merged.get(normalized_point_id)
            if existing is None:
                merged[normalized_point_id] = chunk
                continue

            existing.raw_score = max(existing.raw_score, chunk.raw_score)
            existing.semantic_score = max(existing.semantic_score, chunk.semantic_score)
            existing.lexical_score = max(existing.lexical_score, chunk.lexical_score)
            existing.score = max(existing.score, chunk.score)
            existing.retrieval_sources = sorted(set(existing.retrieval_sources + chunk.retrieval_sources))

        return list(merged.values())

    def _ensure_lexical_index(self):
        """确保 lexical 索引已建立。"""
        if self._lexical_chunks:
            return
        self._build_lexical_index()

    def _build_lexical_index(self, chunks: Optional[List[KnowledgeChunk]] = None):
        """从知识块构建轻量 BM25 索引。"""
        lexical_chunks = chunks
        if lexical_chunks is None:
            lexical_chunks = []
            for path in self._iter_documents():
                lexical_chunks.extend(self._load_document_chunks(path))

        self._lexical_chunks = lexical_chunks
        self._lexical_chunk_map = {chunk.point_id: chunk for chunk in lexical_chunks}
        self._lexical_tokens_by_chunk = {}
        self._lexical_tf_by_chunk = {}

        document_count = len(lexical_chunks)
        doc_freq_counter: defaultdict[str, int] = defaultdict(int)
        total_doc_len = 0

        for chunk in lexical_chunks:
            tokens = self._tokenize_for_lexical(
                " ".join([chunk.title, " ".join(chunk.tags), chunk.source, chunk.content])
            )
            self._lexical_tokens_by_chunk[chunk.point_id] = tokens
            term_freqs = Counter(tokens)
            self._lexical_tf_by_chunk[chunk.point_id] = term_freqs
            total_doc_len += len(tokens)
            for term in term_freqs:
                doc_freq_counter[term] += 1

        self._lexical_df = dict(doc_freq_counter)
        self._lexical_avg_doc_len = (total_doc_len / document_count) if document_count else 0.0

    def _tokenize_for_lexical(self, text: str) -> List[str]:
        """为中文/英文混合文本生成 lexical token。"""
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        tokens: List[str] = []
        for term in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]{2,}", normalized):
            cleaned = term.strip()
            if cleaned:
                tokens.append(cleaned)
                if len(cleaned) >= 4:
                    tokens.extend(self._generate_ngrams(cleaned, 2))
        return tokens

    def _tokenize_query_for_lexical(self, query: str) -> List[str]:
        """为 query 生成更适合中文事实问答的 lexical token。"""
        normalized = self._normalize_text(query)
        if not normalized:
            return []

        tokens: List[str] = []
        segments = [
            segment.strip()
            for segment in re.split(r"[\s,，。！？?、；;：:()（）\[\]【】]+", normalized)
            if segment.strip()
        ]
        for segment in segments:
            for term in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]{2,}", segment):
                cleaned = term.strip()
                if not cleaned:
                    continue
                tokens.extend(self._expand_query_terms(cleaned))

        deduped: List[str] = []
        seen = set()
        for token in tokens:
            cleaned = str(token or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

    def _generate_ngrams(self, text: str, n: int) -> List[str]:
        """生成字符 n-gram，增强专有词和标题命中。"""
        if len(text) < n:
            return []
        return [text[index : index + n] for index in range(0, len(text) - n + 1)]

    def _extract_source_aliases(self, source: str) -> List[str]:
        """从 source 路径提取可用于 query 匹配的文件别名。"""
        source_path = Path(str(source or ""))
        aliases = []

        stem = self._normalize_text(source_path.stem)
        if stem:
            aliases.append(stem)
            aliases.extend(part for part in stem.split("_") if len(part) >= 3)

        normalized_source = self._normalize_text(str(source or ""))
        if normalized_source:
            aliases.extend(
                part for part in re.split(r"[/._-]+", normalized_source)
                if len(part) >= 3
            )

        deduped = []
        seen = set()
        for alias in aliases:
            if not alias or alias in seen:
                continue
            seen.add(alias)
            deduped.append(alias)
        return deduped

    def _is_explicit_doc_reference(self, query: str) -> bool:
        """判断 query 是否显式在问某个文档/规则名。"""
        raw_query = str(query or "")
        if any(marker in raw_query for marker in self.DOC_REFERENCE_MARKERS):
            return True
        return bool(re.search(r"[A-Za-z0-9_./-]{4,}", raw_query))

    def _compute_bm25_score(
        self,
        query_terms: List[str],
        term_freqs: Counter[str],
        doc_len: int,
    ) -> float:
        """计算单文档 BM25 分数。"""
        if not query_terms or not term_freqs or doc_len <= 0 or self._lexical_avg_doc_len <= 0:
            return 0.0

        score = 0.0
        unique_terms = set(query_terms)
        for term in unique_terms:
            term_freq = term_freqs.get(term, 0)
            if term_freq <= 0:
                continue
            doc_freq = self._lexical_df.get(term, 0)
            idf = log(1 + ((len(self._lexical_chunks) - doc_freq + 0.5) / (doc_freq + 0.5)))
            numerator = term_freq * (self.BM25_K1 + 1.0)
            denominator = term_freq + self.BM25_K1 * (
                1.0 - self.BM25_B + self.BM25_B * (doc_len / self._lexical_avg_doc_len)
            )
            score += idf * (numerator / denominator)
        return score

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
        keywords = [
            token
            for token in self._tokenize_query_for_lexical(query)
            if (
                len(token) >= 3
                or token in self.KNOWN_NPCS
                or token in self.QUERY_SHORT_KEYWORDS
            )
        ]
        # 优先保留较长关键词，提升中文片段命中质量。
        # 这里避免把纯 2-gram 当作 rerank 主关键词，降低误命中。
        return sorted(set(keywords), key=len, reverse=True)

    def _expand_query_terms(self, term: str) -> List[str]:
        """把中文 query 片段扩展成更适合 lexical/rerank 的词项。"""
        cleaned = str(term or "").strip()
        if not cleaned:
            return []

        expansions = [cleaned]
        normalized_term = self._strip_query_fillers(cleaned)
        if normalized_term and normalized_term != cleaned:
            expansions.append(normalized_term)

        stripped_stem = self._strip_query_tail(normalized_term or cleaned)
        if len(stripped_stem) >= 2:
            expansions.append(stripped_stem)

        for suffix, hint_terms in self.QUERY_SUFFIX_HINTS:
            if len(cleaned) <= len(suffix):
                continue
            if cleaned.endswith(suffix):
                stem = self._normalize_query_stem(cleaned[: -len(suffix)])
                if len(stem) >= 2:
                    expansions.append(stem)
                    for verb in self.QUERY_TRAILING_VERBS:
                        if stem.endswith(verb) and len(stem) > len(verb):
                            prefix = stem[: -len(verb)].strip()
                            if len(prefix) >= 2:
                                expansions.append(prefix)
                            expansions.append(verb)
                    expansions.extend(hint_terms)

        for verb in self.QUERY_TRAILING_VERBS:
            if cleaned.endswith(verb) and len(cleaned) > len(verb):
                prefix = self._normalize_query_stem(cleaned[: -len(verb)])
                if len(prefix) >= 2:
                    expansions.append(prefix)
                expansions.append(verb)

        tokens: List[str] = []
        seen = set()
        for item in expansions:
            piece = str(item or "").strip()
            if not piece or piece in seen:
                continue
            seen.add(piece)
            tokens.append(piece)
            if len(piece) >= 4:
                tokens.extend(self._generate_ngrams(piece, 2))
        return tokens

    def _strip_query_fillers(self, text: str) -> str:
        """去掉 query 开头礼貌/口语填充词，保留核心语义。"""
        cleaned = str(text or "").strip()
        for filler in self.QUERY_LEADING_FILLERS:
            if cleaned.startswith(filler) and len(cleaned) > len(filler):
                cleaned = cleaned[len(filler) :].strip()
                break
        return cleaned

    def _strip_query_tail(self, text: str) -> str:
        """把中文事实问句尾部的疑问结构裁掉，保留实体/事件主体。"""
        cleaned = self._normalize_query_stem(self._strip_query_fillers(text))
        if len(cleaned) < 2:
            return cleaned

        for pattern in self.QUERY_TAIL_PATTERNS:
            if len(cleaned) <= len(pattern):
                continue
            if cleaned.endswith(pattern):
                stem = self._normalize_query_stem(cleaned[: -len(pattern)])
                if len(stem) >= 2:
                    return stem
        return cleaned

    def _normalize_query_stem(self, text: str) -> str:
        """清理 query 截断后残留的中文功能词。"""
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^[的了呢呀吗嘛吧啊哦请问想问]+", "", cleaned)
        cleaned = re.sub(r"[的是了呢呀吗嘛吧啊哦]+$", "", cleaned)
        return cleaned.strip()

    def _normalize_text(self, text: object) -> str:
        """统一做轻量文本归一化，便于命中判断。"""
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

    def _normalize_point_id(self, point_id: object) -> str:
        """统一 point_id 形式，兼容 Qdrant 返回的 UUID 样式字符串。"""
        raw = str(point_id or "").strip().lower()
        if not raw:
            return ""
        collapsed = re.sub(r"[^0-9a-f]", "", raw)
        if len(collapsed) == 32:
            return collapsed
        return raw

    def _normalize_scopes(self, scopes: List[str]) -> List[str]:
        """归一化 scope 列表，保持顺序并去重。"""
        normalized = []
        seen = set()
        for scope in scopes:
            cleaned = str(scope or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized or ["global"]

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
        return self._normalize_point_id(hashlib.md5(raw.encode("utf-8")).hexdigest())
