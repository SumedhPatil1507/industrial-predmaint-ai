"""
RAG indexing and similarity search for industrial equipment manuals and SOPs.
Provides TF-IDF indexing, token-similarity retrieval, and novelty scoring.
"""

import os
import re
import math
import logging
from typing import Optional, Any, List
from pathlib import Path
from collections import Counter

from backend.copilot.manuals_data import get_all_manuals, get_manuals_for_machine
from backend.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

VECTOR_STORE_DIR = Path("models/vector_store")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Similarity threshold for novel anomaly classification
NOVELTY_SIMILARITY_THRESHOLD = 0.52


class ManualChunk:
    def __init__(self, doc_id: str, machine_type: str, subsystem: str, title: str,
                 content: str, root_cause_analysis: str, sop_steps: list[str], safety_protocol: str,
                 required_parts: list[str], estimated_mttr: float, tolerances: str):
        self.doc_id = doc_id
        self.machine_type = machine_type
        self.subsystem = subsystem
        self.title = title
        self.content = content
        self.root_cause_analysis = root_cause_analysis
        self.sop_steps = sop_steps
        self.safety_protocol = safety_protocol
        self.required_parts = required_parts
        self.estimated_mttr = estimated_mttr
        self.tolerances = tolerances

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "machine_type": self.machine_type,
            "subsystem": self.subsystem,
            "title": self.title,
            "content": self.content,
            "root_cause_analysis": self.root_cause_analysis,
            "sop_steps": self.sop_steps,
            "safety_protocol": self.safety_protocol,
            "required_parts": self.required_parts,
            "estimated_mttr_hours": self.estimated_mttr,
            "tolerances": self.tolerances,
        }


class EquipmentRAGIndexer:
    """
    RAG Indexer that creates vector embeddings for equipment SOPs and provides
    fast semantic search with novelty-gating capabilities.
    """

    def __init__(self):
        self.chunks: list[ManualChunk] = []
        self._build_chunks()
        self._init_vector_store()

    def _build_chunks(self):
        """Transform equipment manuals into searchable structured text chunks."""
        self.chunks = []
        for man in get_all_manuals():
            full_text = (
                f"Machine: {man['machine_type']}\n"
                f"Subsystem: {man['subsystem']}\n"
                f"Title: {man['title']}\n"
                f"Failure Modes: {', '.join(man['failure_modes'])}\n"
                f"Symptoms: {man['symptoms']}\n"
                f"Root Cause: {man['root_cause_analysis']}\n"
                f"Tolerances: {man['clearance_tolerances']}\n"
                f"SOP: {' '.join(man['sop_steps'])}"
            )
            chunk = ManualChunk(
                doc_id=man["manual_id"],
                machine_type=man["machine_type"],
                subsystem=man["subsystem"],
                title=man["title"],
                content=full_text,
                root_cause_analysis=man["root_cause_analysis"],
                sop_steps=man["sop_steps"],
                safety_protocol=man["safety_protocol"],
                required_parts=man["required_parts"],
                estimated_mttr=man["estimated_mttr_hours"],
                tolerances=man["clearance_tolerances"],
            )
            self.chunks.append(chunk)

    def _init_vector_store(self):
        """Build the in-memory TF-IDF index used by manual retrieval."""
        with tracer.start_as_current_span("copilot.tfidf.index") as span:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.corpus = [c.content for c in self.chunks]
                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True
                )
                self.doc_vectors = self.vectorizer.fit_transform(self.corpus)
                span.set_attribute("retrieval.document_count", len(self.chunks))
                logger.info(f"Vectorized {len(self.chunks)} SOP chunks into vector store.")
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"TFIDF Vectorizer initialization note: {e}")

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9_\-\.]{2,}\b", text)
        stopwords = {
            "the", "and", "is", "in", "to", "of", "for", "with", "a", "an", "on", "at",
            "by", "this", "that", "from", "as", "be", "or", "are", "it", "if", "not"
        }
        return [t for t in tokens if t not in stopwords]

    def _compute_cosine_sim(self, query: str, chunk: ManualChunk) -> float:
        """Compute cosine similarity between query and chunk using vector space & keywords."""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(chunk.content)
        
        if not query_tokens or not doc_tokens:
            return 0.0

        q_count = Counter(query_tokens)
        d_count = Counter(doc_tokens)
        intersection = set(q_count.keys()) & set(d_count.keys())
        
        if not intersection:
            return 0.0

        numerator = sum(q_count[t] * d_count[t] for t in intersection)
        norm_q = math.sqrt(sum(v ** 2 for v in q_count.values()))
        norm_d = math.sqrt(sum(v ** 2 for v in d_count.values()))
        
        if norm_q == 0 or norm_d == 0:
            return 0.0

        # Vector cosine
        cosine = numerator / (norm_q * norm_d)
        
        # Keyword match ratio
        overlap_ratio = len(intersection) / max(len(set(query_tokens)), 1)
        
        sim = (cosine * 0.6) + (overlap_ratio * 0.4)

        # Boost when specific subsystem matches
        if any(token in chunk.subsystem.lower() for token in query_tokens):
            sim += 0.15

        return min(round(float(sim), 4), 1.0)

    def search_manuals(
        self,
        machine_type: str,
        query: str,
        top_k: int = 3,
        min_threshold: float = NOVELTY_SIMILARITY_THRESHOLD
    ) -> dict:
        """
        Search equipment manuals for the given machine type and query text.
        Cross-references similarity and returns retrieval results with novelty gating.
        """
        with tracer.start_as_current_span("copilot.manual_retrieval") as span:
            span.set_attribute("retrieval.machine_type", machine_type)
            span.set_attribute("retrieval.top_k", top_k)
            scored_results = []

            # Filter candidate chunks matching machine_type
            candidates = [c for c in self.chunks if not machine_type or c.machine_type.lower() == machine_type.lower()]
            if not candidates:
                candidates = self.chunks

            with tracer.start_as_current_span("copilot.manual_similarity_scoring") as score_span:
                score_span.set_attribute("retrieval.algorithm", "token_cosine_keyword_overlap")
                score_span.set_attribute("retrieval.candidate_count", len(candidates))
                for chunk in candidates:
                    sim = self._compute_cosine_sim(query, chunk)
                    scored_results.append({
                        "chunk": chunk,
                        "similarity_score": round(sim, 4),
                    })

        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_matches = scored_results[:top_k]

        top_similarity = top_matches[0]["similarity_score"] if top_matches else 0.0
        is_novel = top_similarity < min_threshold

        retrieved_docs = []
        for item in top_matches:
            c = item["chunk"]
            retrieved_docs.append({
                "manual_id": c.doc_id,
                "vector_chunk_id": f"{c.doc_id}#chunk-000",
                "machine_type": c.machine_type,
                "subsystem": c.subsystem,
                "title": c.title,
                "root_cause_analysis": c.root_cause_analysis,
                "similarity_score": item["similarity_score"],
                "sop_steps": c.sop_steps,
                "safety_protocol": c.safety_protocol,
                "required_parts": c.required_parts,
                "estimated_mttr_hours": c.estimated_mttr,
                "tolerances": c.tolerances,
                "snippet": c.content[:300] + "...",
            })

        novelty_reason = None
        if is_novel:
            novelty_reason = (
                f"Symptom signature (Top similarity {top_similarity:.2f} < threshold {min_threshold:.2f}) "
                f"does not sufficiently match standard known OEM SOPs for {machine_type}. "
                "Possible novel failure mode, multi-component interaction, or structural fatigue."
            )

        return {
            "machine_type": machine_type,
            "query": query,
            "top_similarity": top_similarity,
            "is_novel_anomaly": is_novel,
            "novelty_threshold": min_threshold,
            "novelty_reason": novelty_reason,
            "retrieved_docs": retrieved_docs,
        }


# Global singleton indexer
_rag_indexer: Optional[EquipmentRAGIndexer] = None


def get_rag_indexer() -> EquipmentRAGIndexer:
    global _rag_indexer
    if _rag_indexer is None:
        _rag_indexer = EquipmentRAGIndexer()
    return _rag_indexer
