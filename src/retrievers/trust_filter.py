"""Hierarchy-aware trust filter (non-linear by default).

Levels (unchanged):
- L_doc    = 1 - openness
- L_user   = max L_doc among buckets the querier can read (same scale as L_doc)
- L_writer = author privilege (from written_by / writer_level); unknown → no writer term
- gap      = max(0, L_user - L_doc)

Components:
- T_doc    = L_doc
- T_hier   = 1 - σ((gap - δ0) / k)     [default, sigmoid]
           or 1 - gap                  [legacy linear]
- T_writer = L_writer when include_writer=True (main); unused (=1) when False

Final score (default — multiplicative):
    T = T_doc × T_hier × T_writer   [include_writer=True, main]
    T = T_doc × T_hier              [include_writer=False, ablation; T_writer_eff=1]

- T_writer = L_writer when writer_mode=level (default when writer on)

Legacy additive mode kept only for ablation:
    T = w_doc·T_doc + w_hier·T_hier + w_writer·T_writer
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple


def _resolve_provider_scope(provider_scope: Optional[str] = None) -> str:
    """Return 'gcp' when SKIP_AWS=1 or provider_scope='gcp'; else 'all'."""
    if provider_scope:
        return provider_scope.strip().lower()
    if os.environ.get("SKIP_AWS", "0").strip().lower() in {"1", "true", "yes"}:
        return "gcp"
    return "all"


def _sigmoid(x: float) -> float:
    # numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _user_can_read_gcp(user: Dict[str, Any], bucket: str, gcp_rbac: Dict[str, str]) -> bool:
    if user.get("full_access"):
        return True
    roles = user.get("gcp_roles") or []
    return any(gcp_rbac.get(r) == bucket for r in roles)


def _user_can_read_aws(user: Dict[str, Any], bucket: str, aws_abac: Dict[str, Dict[str, str]]) -> bool:
    if user.get("full_access"):
        return True
    required = aws_abac.get(bucket) or {}
    if not required:
        return False
    attrs = user.get("aws_attributes") or {}
    return all(attrs.get(k) == v for k, v in required.items())


def compute_bucket_openness(
    access_model: Dict[str, Any],
    provider_scope: Optional[str] = None,
) -> Dict[Tuple[str, str], float]:
    users = access_model.get("USERS") or []
    n = max(len(users), 1)
    gcp_rbac = access_model.get("GCP_RBAC") or {}
    aws_abac = access_model.get("AWS_ABAC") or {}
    scope = _resolve_provider_scope(provider_scope)

    openness: Dict[Tuple[str, str], float] = {}
    for bucket in set(gcp_rbac.values()):
        cnt = sum(1 for u in users if _user_can_read_gcp(u, bucket, gcp_rbac))
        openness[("gcp", bucket)] = cnt / n
    if scope != "gcp":
        for bucket in aws_abac.keys():
            cnt = sum(1 for u in users if _user_can_read_aws(u, bucket, aws_abac))
            openness[("aws", bucket)] = cnt / n
    return openness


def compute_user_levels(
    access_model: Dict[str, Any],
    provider_scope: Optional[str] = None,
) -> Dict[str, float]:
    """Clearance = exclusivity of the most locked bucket the user can read.

    Same units as L_doc (1 - readers/|USERS|). Empty access → 0.
    """
    users = access_model.get("USERS") or []
    gcp_rbac = access_model.get("GCP_RBAC") or {}
    aws_abac = access_model.get("AWS_ABAC") or {}
    gcp_buckets = list(set(gcp_rbac.values()))
    aws_buckets = list(aws_abac.keys())
    scope = _resolve_provider_scope(provider_scope)
    openness = compute_bucket_openness(access_model, provider_scope)

    levels: Dict[str, float] = {}
    for u in users:
        name = u.get("name") or ""
        ldocs = []
        for b in gcp_buckets:
            if _user_can_read_gcp(u, b, gcp_rbac):
                ldocs.append(1.0 - float(openness.get(("gcp", b), 0.5)))
        if scope != "gcp":
            for b in aws_buckets:
                if _user_can_read_aws(u, b, aws_abac):
                    ldocs.append(1.0 - float(openness.get(("aws", b), 0.5)))
        levels[name] = max(ldocs) if ldocs else 0.0
    return levels


class TrustFilter:
    def __init__(
        self,
        access_model: Dict[str, Any],
        shared_bucket: Optional[str] = None,
        shared_provider: str = "gcp",
        w_doc: float = 0.4,
        w_hier: float = 0.6,
        w_writer: float = 0.0,
        threshold: float = 0.15,
        shared_doc_cap: float = 0.35,
        # non-linear defaults (no-writer ablation selected):
        # T = T_doc * T_hier, T_hier = 1 - sigmoid((gap-0.10)/0.08), thr=0.15
        combine_mode: str = "mul",  # "mul" | "add" | "logistic"
        hier_mode: str = "sigmoid",  # "sigmoid" | "linear"
        writer_mode: str = "gap",  # only used when include_writer=True
        include_writer: Optional[bool] = None,
        sigmoid_delta0: float = 0.10,
        sigmoid_k: float = 0.08,
        logistic_b: float = 0.5,
        logistic_t: float = 0.2,
        # backward-compat aliases
        w_src: Optional[float] = None,
        w_cnt: Optional[float] = None,
        shared_src_cap: Optional[float] = None,
        dist_ref: float = 1.0,
        provider_scope: Optional[str] = None,
    ):
        if w_src is not None:
            w_doc = w_src
        if w_cnt is not None and w_src is not None:
            w_hier = w_cnt
        if shared_src_cap is not None:
            shared_doc_cap = shared_src_cap

        self.access_model = access_model
        self.provider_scope = _resolve_provider_scope(provider_scope)
        self.openness = compute_bucket_openness(access_model, self.provider_scope)
        self.user_levels = compute_user_levels(access_model, self.provider_scope)
        self.users_by_name = {u.get("name"): u for u in (access_model.get("USERS") or [])}
        self.shared_bucket = shared_bucket
        self.shared_provider = (shared_provider or "gcp").lower()
        self.w_doc = float(w_doc)
        self.w_hier = float(w_hier)
        self.w_writer = float(w_writer)
        self.threshold = float(threshold)
        self.shared_doc_cap = shared_doc_cap
        self.combine_mode = (combine_mode or "mul").lower()
        self.hier_mode = (hier_mode or "sigmoid").lower()
        self.writer_mode = (writer_mode or "gap").lower()
        if include_writer is None:
            # default OFF for mul (current main ablation); ON for add only if w_writer>0
            self.include_writer = bool(self.w_writer > 0)
        else:
            self.include_writer = bool(include_writer)
        self.sigmoid_delta0 = float(sigmoid_delta0)
        self.sigmoid_k = max(float(sigmoid_k), 1e-6)
        self.logistic_b = float(logistic_b)
        self.logistic_t = max(float(logistic_t), 1e-6)
        # aliases
        self.w_src = self.w_doc
        self.w_cnt = self.w_hier
        self.shared_src_cap = shared_doc_cap
        self.dist_ref = dist_ref

    def doc_level(self, meta: Dict[str, Any]) -> float:
        provider = (meta.get("provider") or "").lower()
        bucket = (meta.get("parameters") or {}).get("bucket") or ""
        if not provider or not bucket:
            return 0.5
        open_frac = self.openness.get((provider, bucket), 0.5)
        level = 1.0 - float(open_frac)
        if (
            self.shared_bucket
            and provider == self.shared_provider
            and bucket == self.shared_bucket
        ):
            level = min(level, self.shared_doc_cap)
        return max(0.0, min(1.0, level))

    def user_level(self, user_name: Optional[str]) -> float:
        if not user_name:
            return 0.5
        if user_name in self.user_levels:
            return float(self.user_levels[user_name])
        u = self.users_by_name.get(user_name)
        if u and u.get("full_access"):
            return 1.0
        return 0.5

    def writer_level(self, meta: Dict[str, Any]) -> Optional[float]:
        """Author clearance: same definition as L_user (max L_doc they can read)."""
        writer = meta.get("written_by") or meta.get("writer_user")
        if writer:
            return self.user_level(str(writer))
        if "writer_level" in meta and meta["writer_level"] is not None:
            return max(0.0, min(1.0, float(meta["writer_level"])))
        return None

    def hierarchy_trust(self, user_name: Optional[str], meta: Dict[str, Any]) -> Tuple[float, float, float]:
        """Return (T_hier, L_user, gap)."""
        l_user = self.user_level(user_name)
        l_doc = self.doc_level(meta)
        gap = max(0.0, l_user - l_doc)
        if self.hier_mode == "linear":
            t_hier = max(0.0, min(1.0, 1.0 - gap))
        else:
            # sigmoid: large gap → low trust; small/zero gap → high trust
            t_hier = 1.0 - _sigmoid((gap - self.sigmoid_delta0) / self.sigmoid_k)
            t_hier = max(0.0, min(1.0, t_hier))
        return t_hier, l_user, gap

    def writer_trust(self, user_name: Optional[str], meta: Dict[str, Any]) -> Tuple[float, Optional[float], float]:
        """Return (T_writer, L_writer|None, gap_w).

        Default writer_mode='level': T_writer = L_writer (author privilege only).
        Unknown writer → T_writer = 1.0. gap_w kept for diagnostics only.
        """
        l_user = self.user_level(user_name)
        l_writer = self.writer_level(meta)
        if l_writer is None:
            return 1.0, None, 0.0
        gap_w = max(0.0, l_user - l_writer)
        if self.writer_mode == "gap":
            t_writer = max(0.0, min(1.0, 1.0 - gap_w))
        else:
            # level: trust scales with author privilege alone, not reader–writer gap
            t_writer = max(0.0, min(1.0, float(l_writer)))
        return t_writer, l_writer, gap_w

    def source_trust(self, meta: Dict[str, Any]) -> float:
        return self.doc_level(meta)

    def content_trust(self, distance: Optional[float] = None) -> float:
        return 0.5

    def score(
        self,
        meta: Dict[str, Any],
        distance: Optional[float] = None,
        user_name: Optional[str] = None,
    ) -> Dict[str, float]:
        t_doc = self.doc_level(meta)
        t_hier, l_user, gap = self.hierarchy_trust(user_name, meta)
        t_writer, l_writer, gap_w = self.writer_trust(user_name, meta)

        if not self.include_writer:
            t_writer_eff = 1.0
        else:
            t_writer_eff = t_writer

        if self.combine_mode == "add":
            if self.include_writer and self.w_writer > 0:
                t = self.w_doc * t_doc + self.w_hier * t_hier + self.w_writer * t_writer_eff
            else:
                # renormalize if writer off
                s = self.w_doc + self.w_hier
                if s <= 0:
                    t = 0.0
                else:
                    t = (self.w_doc / s) * t_doc + (self.w_hier / s) * t_hier
        elif self.combine_mode == "logistic":
            z = self.w_doc * t_doc + self.w_hier * t_hier + self.w_writer * t_writer_eff
            t = _sigmoid((z - self.logistic_b) / self.logistic_t)
        elif self.combine_mode == "logistic_max":
            t_src = max(t_doc, t_writer_eff)
            z = self.w_doc * t_src + self.w_hier * t_hier
            t = _sigmoid((z - self.logistic_b) / self.logistic_t)
        else:
            # multiplicative (default)
            t = t_doc * t_hier * t_writer_eff

        out = {
            "L_user": l_user,
            "L_doc": t_doc,
            "L_writer": -1.0 if l_writer is None else float(l_writer),
            "gap": gap,
            "gap_w": gap_w,
            "T_doc": t_doc,
            "T_hier": t_hier,
            "T_writer": t_writer,
            "T_src": t_doc,
            "T_cnt": t_hier,
            "T": float(t),
            "w_doc": self.w_doc,
            "w_hier": self.w_hier,
            "w_writer": self.w_writer,
            "combine_mode": 1.0 if self.combine_mode == "mul" else 0.0,
            "hier_mode_sigmoid": 1.0 if self.hier_mode == "sigmoid" else 0.0,
        }
        return out

    def filter_docs(
        self,
        docs: List[Dict[str, Any]],
        metadata_store,
        user_name: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept: List[Dict[str, Any]] = []
        score_rows: List[Dict[str, Any]] = []
        for d in docs:
            meta = metadata_store.get_by_uuid(d.get("global_uuid")) or {}
            scores = self.score(meta, d.get("distance"), user_name=user_name)
            row = {
                "file_name": meta.get("file_name", ""),
                "global_uuid": d.get("global_uuid"),
                "user": user_name,
                **scores,
                "kept": scores["T"] >= self.threshold,
            }
            score_rows.append(row)
            if row["kept"]:
                kept.append(d)
        return kept, score_rows
