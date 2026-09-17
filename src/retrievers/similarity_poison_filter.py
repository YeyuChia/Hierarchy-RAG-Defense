"""Query–document similarity relevance gate (standard RAG logic).

Keep chunks whose query–content cosine is **at or above** a threshold; drop
low-similarity chunks as likely irrelevant.  This is *not* a concat_q-specific
anti-poison rule (no dropping of suspiciously *high* similarity).

Typical pipeline:
    retrieve → ACL → SimilarityPoisonFilter → TrustFilter → LLM
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SimilarityPoisonFilter:
    """Keep ACL docs with cos(query, content) >= min_cosine; drop lower-sim chunks."""

    def __init__(
        self,
        min_cosine: float = 0.45,
        max_content_chars: int = 2000,
        # legacy alias — do not use (was concat_q-specific high-sim drop)
        max_cosine: Optional[float] = None,
    ):
        if max_cosine is not None:
            raise ValueError(
                "max_cosine (drop-if-too-high) is removed; use min_cosine "
                "(keep-if-similar-enough) instead."
            )
        self.min_cosine = float(min_cosine)
        self.max_content_chars = int(max_content_chars)

    def _content(self, doc: Dict[str, Any], meta: Dict[str, Any]) -> str:
        text = doc.get("content") or meta.get("content") or ""
        return str(text)[: self.max_content_chars]

    def score_one(
        self,
        query: str,
        doc: Dict[str, Any],
        meta: Dict[str, Any],
        encoder=None,
        query_emb: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        content = self._content(doc, meta)
        cos_sim: Optional[float] = None

        if encoder is not None and content:
            qe = query_emb
            if qe is None:
                qe = encoder.encode([query])[0]
            ce = encoder.encode([content])[0]
            cos_sim = _cosine(np.asarray(qe, dtype=np.float32), np.asarray(ce, dtype=np.float32))

        # Standard gate: drop if similarity is below threshold
        cos_drop = cos_sim is not None and cos_sim < self.min_cosine
        # If encoder/content missing, do not drop (same as before)
        if cos_sim is None:
            cos_drop = False

        return {
            "cos_sim": -1.0 if cos_sim is None else float(cos_sim),
            "cos_drop": float(cos_drop),
            "suspicious": float(cos_drop),
            "kept": not cos_drop,
        }

    def filter_docs(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        metadata_store,
        encoder=None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        query_emb = None
        if encoder is not None and docs:
            query_emb = encoder.encode([query])[0]

        for doc in docs:
            uid = doc.get("global_uuid")
            meta = metadata_store.get_by_uuid(uid) or {}
            sc = self.score_one(query, doc, meta, encoder=encoder, query_emb=query_emb)
            row = {
                "file_name": meta.get("file_name", ""),
                "global_uuid": uid,
                **sc,
            }
            rows.append(row)
            if sc["kept"]:
                kept.append(doc)
        return kept, rows


def calibrate_min_cosine(
    attack_docs: List[Dict[str, Any]],
    vector_store,
    permission_retriever,
    ac_manager,
    metadata_store,
    poison_names: set,
    grid: Optional[List[float]] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Pick min_cosine on VAL: lower Poison@k / higher clean retention where possible."""
    grid = grid or [round(x, 2) for x in np.arange(0.40, 0.71, 0.05)]
    vector_store._ensure_model()
    encoder = vector_store.model

    pairs: List[Tuple[float, bool]] = []
    for d in attack_docs:
        q = d["question"]
        retrieved = vector_store.search(q, top_k=5)
        acl, _, _, _ = permission_retriever.filter_docs(retrieved, ac_manager)
        qe = encoder.encode([q])[0]
        for doc in acl:
            meta = metadata_store.get_by_uuid(doc.get("global_uuid")) or {}
            content = doc.get("content") or meta.get("content") or ""
            if not content:
                continue
            ce = encoder.encode([content[:2000]])[0]
            cos = _cosine(np.asarray(qe), np.asarray(ce))
            is_poison = bool(meta.get("is_poisoned") or meta.get("file_name") in poison_names)
            pairs.append((cos, is_poison))

    sweep = []
    best_tau = grid[0]
    best_key = None
    for tau in grid:
        tp = fp = fn = tn = 0
        for cos, is_p in pairs:
            drop = cos < tau
            if is_p and drop:
                tp += 1
            elif is_p and not drop:
                fn += 1
            elif (not is_p) and drop:
                fp += 1
            else:
                tn += 1
        atk_recall = tp / (tp + fn) if tp + fn else 0.0
        gtr = tn / (tn + fp) if tn + fp else 0.0
        row = {
            "min_cosine": tau,
            "Attack_Recall": round(atk_recall, 4),
            "GT_Recall": round(gtr, 4),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }
        sweep.append(row)
        key = (fn, -gtr)  # prefer dropping poison (low fn), then clean retention
        if best_key is None or key < best_key:
            best_key = key
            best_tau = tau
    return best_tau, sweep


# Backward-compat name for imports; prefer calibrate_min_cosine
calibrate_max_cosine = calibrate_min_cosine
