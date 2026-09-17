"""Shared helpers for the cloud ladder eval (imported by script 18).

Provides victim/placements, VAL/TEST split, retrieve+ACL cache, TrustFilter
grid, poison remap, and metric helpers. GCP-only: LOCAL_MOCK_IAM=0, SKIP_AWS=1.

The __main__ path is an older GTR / similarity-vs-trust protocol.
Current experiment entry: python scripts/18_cloud_ladder_main_k5.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

from src.managers.credential_manager import CredentialManager
from src.retrievers.permission_retriever import PermissionRetriever, PermissionRetrieverConfig
from src.retrievers.similarity_poison_filter import SimilarityPoisonFilter
from src.retrievers.trust_filter import TrustFilter
from src.retrievers.trust_param_loss import (
    LossWeights,
    compute_val_loss,
    loss_formula_markdown,
    select_best_params,
)
from src.stores.metadata_store import MetadataStore
from src.stores.vector_store import VectorStore

ARTIFACTS = ROOT / "artifacts"
OUT = ARTIFACTS / "results"
MANIFEST = ARTIFACTS / "poison" / "manifest.json"
ASSIGNMENT = ARTIFACTS / "poison" / "question_bucket_assignment.json"
ACCESS_MODEL = ARTIFACTS / "generated_access_model.json"
QUESTION_LIST = ARTIFACTS / "question_list.json"

TOP_K = 5
SEED = 42
N_CLEAN_VAL = 40
N_CLEAN_TEST = 40


def _load_victim_and_placements():
    """Read querying user + low/mid/high buckets from access model (tier_rank)."""
    am = json.loads(ACCESS_MODEL.read_text(encoding="utf-8"))
    victim = (am.get("DESIGN") or {}).get("querying_user") or "user-31"
    tiers = am.get("BUCKET_TIERS") or {}
    ranked = sorted(tiers, key=lambda b: int((tiers[b] or {}).get("tier_rank") or 99))
    if len(ranked) < 3:
        raise SystemExit("BUCKET_TIERS needs >=3 buckets for low/mid/high placements")
    return victim, {
        "low": {"provider": "gcp", "bucket": ranked[0]},
        "mid": {"provider": "gcp", "bucket": ranked[1]},
        "high": {"provider": "gcp", "bucket": ranked[-1]},
    }


VICTIM, PLACEMENTS = _load_victim_and_placements()

# Joint grid for blocking low+mid on cloud VAL (user-31 L_user ≈ 0.906)
DELTA0S = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
KS = [0.08, 0.10, 0.12, 0.16, 0.20]
THRS = [0.10, 0.12, 0.15, 0.18, 0.20]
# min_cosine: keep if cos(query,content) >= tau; drop low-sim (standard RAG gate)
SIM_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def unique_parent_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for d in docs:
        pid = d.get("parent_id") or d["id"]
        if pid in seen:
            continue
        seen.add(pid)
        out.append(d)
    return out


def set_bucket(meta: Dict[str, Any], provider: str, bucket: str) -> None:
    meta["provider"] = provider
    params = dict(meta.get("parameters") or {})
    params["bucket"] = bucket
    params.setdefault("iam", "google iam")
    params.setdefault("region", "asia-northeast3")
    meta["parameters"] = params


def is_poison(meta: Dict[str, Any], poison_names: set) -> bool:
    return bool(meta.get("is_poisoned") or meta.get("file_name") in poison_names)


def remap_poison(ms: MetadataStore, provider: str, bucket: str) -> None:
    """Point poison at a physical bucket copy and re-stamp eligible writers.

    Blobs already exist in every tier bucket (script 13). This selects which
    copy we score and assigns written_by among users allowed on that bucket so
    T_writer stays write-ACL consistent.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s14", ROOT / "scripts" / "14_stamp_writer_levels.py"
    )
    s14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s14)
    am = json.loads(ACCESS_MODEL.read_text(encoding="utf-8"))
    levels = s14.gcp_user_levels(am)
    pool = s14.clean_pool(am)
    for meta in ms.all():
        if meta.get("is_poisoned"):
            set_bucket(meta, provider, bucket)
            s14.stamp_meta_for_bucket(meta, am, bucket, levels, pool)


def _poison_variant(meta: Dict[str, Any]) -> int:
    label = str(meta.get("poison_id") or meta.get("file_name") or "")
    for v in (1, 2, 3):
        if f"_v{v}" in label:
            return v
    return 1


def remap_poison_mixed(ms: MetadataStore) -> None:
    """Spread poison variants over low/mid/high; re-stamp writers per bucket."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s14", ROOT / "scripts" / "14_stamp_writer_levels.py"
    )
    s14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s14)
    am = json.loads(ACCESS_MODEL.read_text(encoding="utf-8"))
    levels = s14.gcp_user_levels(am)
    pool = s14.clean_pool(am)
    variant_place = {1: "low", 2: "mid", 3: "high"}
    for meta in ms.all():
        if not meta.get("is_poisoned"):
            continue
        place = variant_place.get(_poison_variant(meta), "low")
        provider, bucket = PLACEMENTS[place]["provider"], PLACEMENTS[place]["bucket"]
        set_bucket(meta, provider, bucket)
        s14.stamp_meta_for_bucket(meta, am, bucket, levels, pool)


def split_val_test(docs: List[Dict[str, Any]], seed: int = SEED):
    """Half split; stratified by assigned bucket when question_bucket_assignment.json exists."""
    from collections import defaultdict

    rng = random.Random(seed)
    assignment = {}
    if ASSIGNMENT.exists():
        assignment = (json.loads(ASSIGNMENT.read_text(encoding="utf-8")) or {}).get("assignment") or {}
    groups = defaultdict(list)
    for d in docs:
        pid = d.get("parent_id") or d.get("id")
        groups[assignment.get(pid, "_")].append(d)
    val, test = [], []
    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        mid = len(items) // 2
        val.extend(items[:mid])
        test.extend(items[mid:])
    rng.shuffle(val)
    rng.shuffle(test)
    return val, test


def make_tf(
    access_model,
    delta0,
    k,
    thr,
    provider_scope: str = "gcp",
    include_writer: bool = True,
    combine_mode: str = "mul",
    w_doc: float = 0.4,
    w_hier: float = 0.6,
    w_writer: float = 0.0,
    logistic_b: float = 0.5,
    logistic_t: float = 0.2,
) -> TrustFilter:
    kw: dict = dict(
        access_model=access_model,
        combine_mode=combine_mode,
        hier_mode="sigmoid",
        include_writer=include_writer,
        threshold=thr,
        sigmoid_delta0=delta0,
        sigmoid_k=k,
        w_doc=w_doc,
        w_hier=w_hier,
        w_writer=w_writer,
        logistic_b=logistic_b,
        logistic_t=logistic_t,
        shared_bucket=None,
        shared_doc_cap=1.0,
        provider_scope=provider_scope,
    )
    if include_writer:
        kw["writer_mode"] = "level"
    return TrustFilter(**kw)


def filter_metrics(acl_docs_list, kept_lists, ms, poison_names, top_k: int = TOP_K):
    tp = fp = fn = tn = 0
    asr_hits = poison_at_k_sum = 0
    for (d, acl_docs), kept in zip(acl_docs_list, kept_lists):
        kept_ids = {r["global_uuid"] for r in kept}
        poison_in_kept = 0
        for row in acl_docs:
            uid = row["global_uuid"]
            meta = ms.get_by_uuid(uid) or {}
            poison = is_poison(meta, poison_names)
            dropped = uid not in kept_ids
            if poison and dropped:
                tp += 1
            elif (not poison) and dropped:
                fp += 1
            elif poison and (not dropped):
                fn += 1
                poison_in_kept += 1
            else:
                tn += 1
        poison_at_k_sum += poison_in_kept / max(top_k, 1)
        target = (d.get("target_answer") or "").lower()
        ctx = "\n\n".join((r.get("content") or "") for r in kept).lower()
        asr_hits += int(bool(target) and target in ctx)
    n = max(len(acl_docs_list), 1)
    return {
        "Poison_at_k": round(poison_at_k_sum / n, 4),
        "ASR_star": round(asr_hits / n, 4),
        "Attack_Recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "GT_Recall": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "Precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n": n,
    }


def load_vector_store() -> VectorStore:
    faiss_dir = Path(os.environ.get("FAISS_ASCII_DIR", r"C:\temp\ref_rag_artifacts"))
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss_index = faiss_dir / "faiss.index"
    faiss_meta = faiss_dir / "faiss_metadata.json"
    src_index = ARTIFACTS / "faiss.index"
    src_meta = ARTIFACTS / "faiss_metadata.json"
    for src, dst in ((src_index, faiss_index), (src_meta, faiss_meta)):
        if not src.exists():
            continue
        if (not dst.exists()) or src.stat().st_mtime > dst.stat().st_mtime or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
    vs = VectorStore(model_name="all-MiniLM-L6-v2")
    vs.load(str(faiss_index), str(faiss_meta))
    return vs


def precompute_acl(
    docs: List[Dict[str, Any]],
    vs: VectorStore,
    pr: PermissionRetriever,
    ac,
    label: str,
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Once: retrieve + ACL. Trust/sim grids reuse this."""
    pr.clear_cache()
    out = []
    for i, d in enumerate(docs):
        q = d.get("question") if "question" in d else d["question"]
        retrieved = vs.search(q, top_k=TOP_K)
        acl_docs, _, _, _ = pr.filter_docs(retrieved, ac)
        out.append((d, acl_docs))
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [ACL/{label}] {i+1}/{len(docs)} acl={len(acl_docs)}", flush=True)
    return out


def metrics_from_cache(
    cached: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
    ms: MetadataStore,
    poison_names: set,
    keep_fn,
) -> Dict[str, float]:
    pairs = []
    kept_lists = []
    for d, acl_docs in cached:
        kept = keep_fn(d, acl_docs, ms)
        pairs.append((d, acl_docs))
        kept_lists.append(kept)
    return filter_metrics(pairs, kept_lists, ms, poison_names)


def clean_keep_rate(
    cached_clean: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
    ms: MetadataStore,
    keep_fn,
) -> float:
    tot_acl = tot_kept = 0
    for d, acl_docs in cached_clean:
        kept = keep_fn(d, acl_docs, ms)
        tot_acl += len(acl_docs)
        tot_kept += len(kept)
    return (tot_kept / tot_acl) if tot_acl else 0.0


def step1_select_trust(
    access_model,
    cached_val,
    cached_clean_val,
    ms0: MetadataStore,
    poison_names: set,
    loss_weights: LossWeights,
    provider_scope: str,
    include_writer: bool = False,
):
    grid = []
    n_cfg = len(DELTA0S) * len(KS) * len(THRS)
    print(f"[STEP1] joint grid size={n_cfg} (δ0×k×thr) on cloud VAL", flush=True)
    done = 0
    for delta0, k, thr in product(DELTA0S, KS, THRS):
        done += 1
        tf = make_tf(access_model, delta0, k, thr, provider_scope, include_writer=include_writer)

        def trust_keep(d, acl_docs, ms, _tf=tf):
            kept, _ = _tf.filter_docs(acl_docs, ms, user_name=VICTIM)
            return kept

        ms_mixed = deepcopy(ms0)
        remap_poison_mixed(ms_mixed)
        val_mixed = metrics_from_cache(cached_val, ms_mixed, poison_names, trust_keep)

        per = {}
        for place_name, place in PLACEMENTS.items():
            ms = deepcopy(ms0)
            remap_poison(ms, place["provider"], place["bucket"])
            per[place_name] = metrics_from_cache(cached_val, ms, poison_names, trust_keep)

        clean_keep = clean_keep_rate(cached_clean_val, ms0, trust_keep)
        bd = compute_val_loss(
            val_mixed,
            loss_weights,
            clean_keep=clean_keep,
            placement_breakdown=per,
        )
        row = {
            "delta0": delta0,
            "k": k,
            "threshold": thr,
            "val_mixed": val_mixed,
            "val_placement": per,
            "clean_keep": round(clean_keep, 4),
            "loss_breakdown": bd.to_dict(),
        }
        grid.append(row)
        if done % 10 == 0 or done == 1 or done == n_cfg:
            print(
                f"  [{done}/{n_cfg}] d0={delta0:.2f} k={k:.2f} thr={thr:.2f} "
                f"P@k L/M/H={bd.poison_at_k_low:.2f}/{bd.poison_at_k_mid:.2f}/{bd.poison_at_k_high:.2f} "
                f"AR={bd.attack_recall:.2f} GTR={val_mixed['GT_Recall']:.2f} "
                f"clean={clean_keep:.2f} sec={bd.security_feasible} ok={bd.feasible} L={bd.loss:.4f}",
                flush=True,
            )

    best, ranked = select_best_params(grid, loss_weights)
    return best, ranked


def calibrate_sim_tau(
    cached_val,
    ms0: MetadataStore,
    poison_names: set,
    vs: VectorStore,
) -> Tuple[float, List[Dict[str, Any]]]:
    vs._ensure_model()
    encoder = vs.model
    ms = deepcopy(ms0)
    remap_poison_mixed(ms)

    sweep = []
    best_tau = SIM_GRID[0]
    best_key = None
    for tau in SIM_GRID:
        sim = SimilarityPoisonFilter(min_cosine=tau)

        def keep_fn(d, acl_docs, ms_local, _sim=sim):
            kept, _ = _sim.filter_docs(d["question"], acl_docs, ms_local, encoder=encoder)
            return kept

        stats = metrics_from_cache(cached_val, ms, poison_names, keep_fn)
        row = {
            "min_cosine": tau,
            "Poison_at_k": stats["Poison_at_k"],
            "GT_Recall": stats["GT_Recall"],
        }
        sweep.append(row)
        key = (stats["Poison_at_k"], -stats["GT_Recall"])
        if best_key is None or key < best_key:
            best_key = key
            best_tau = tau
    return best_tau, sweep


def eval_modes_on_test(
    cached_test,
    ms0: MetadataStore,
    poison_names: set,
    access_model,
    tf: TrustFilter,
    sim: SimilarityPoisonFilter,
    vs: VectorStore,
):
    vs._ensure_model()
    encoder = vs.model
    modes = ("acl", "similarity_only", "trust", "similarity_trust")
    results = {"by_placement": {}, "mixed": {}}

    for mode in modes:
        results["by_placement"][mode] = {}

        def keep_fn(d, acl_docs, ms_local, _mode=mode):
            if _mode == "acl":
                return list(acl_docs)
            if _mode == "similarity_only":
                kept, _ = sim.filter_docs(d["question"], acl_docs, ms_local, encoder=encoder)
                return kept
            if _mode == "trust":
                kept, _ = tf.filter_docs(acl_docs, ms_local, user_name=VICTIM)
                return kept
            after_sim, _ = sim.filter_docs(d["question"], acl_docs, ms_local, encoder=encoder)
            kept, _ = tf.filter_docs(after_sim, ms_local, user_name=VICTIM)
            return kept

        ms_mixed = deepcopy(ms0)
        remap_poison_mixed(ms_mixed)
        results["mixed"][mode] = metrics_from_cache(cached_test, ms_mixed, poison_names, keep_fn)
        print(
            f"[TEST {mode:18} mixed] P@k={results['mixed'][mode]['Poison_at_k']:.2f} "
            f"GTR={results['mixed'][mode]['GT_Recall']:.2f}",
            flush=True,
        )

        for place_name, place in PLACEMENTS.items():
            ms = deepcopy(ms0)
            remap_poison(ms, place["provider"], place["bucket"])
            stats = metrics_from_cache(cached_test, ms, poison_names, keep_fn)
            results["by_placement"][mode][place_name] = stats
            print(
                f"[TEST {mode:18} {place_name}] P@k={stats['Poison_at_k']:.2f} "
                f"GTR={stats['GT_Recall']:.2f}",
                flush=True,
            )
    return results


def write_select_md(best, ranked, loss_weights, path: Path):
    bd = best["loss_breakdown"]
    vm = best.get("val_mixed") or {}
    lines = [
        "# Cloud Trust parameter selection (VAL loss)",
        "",
        loss_formula_markdown(loss_weights),
        "",
        f"## Selected: δ0={best['delta0']}, k={best['k']}, thr={best['threshold']}",
        "",
        f"- Loss L = **{bd['loss']:.4f}** (feasible={bd['feasible']}, security_ok={bd.get('security_feasible', False)})",
        f"- Mixed VAL: Poison@k = {bd['poison_at_k']:.2f}, "
        f"GTR = {bd['gtr_utility']:.3f}, clean_keep = {bd['clean_keep']:.3f}",
        f"- Single-bucket VAL Poison@k low/mid/high = "
        f"{bd.get('poison_at_k_low', 0):.2f}/{bd.get('poison_at_k_mid', 0):.2f}/{bd.get('poison_at_k_high', 0):.2f}",
        f"- Mixed VAL AttackRecall = {bd.get('attack_recall', 0):.2f}",
        f"- Single-bucket VAL AttackRecall low/mid/high = "
        f"{bd.get('attack_recall_low', 0):.2f}/"
        f"{bd.get('attack_recall_mid', 0):.2f}/"
        f"{bd.get('attack_recall_high', 0):.2f}",
        f"- provider_scope = gcp (LOCAL_MOCK_IAM=0, SKIP_AWS=1)",
        "",
        "## Top-15 by cloud selection",
        "",
        "| rank | δ0 | k | thr | L | AR | P@k L/M/H | GTR | clean | sec | ok |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|:---:|:---:|",
    ]
    for i, row in enumerate(ranked[:15], 1):
        b = row["loss_breakdown"]
        vm_row = row.get("val_mixed") or {}
        lines.append(
            f"| {i} | {row['delta0']:.2f} | {row['k']:.2f} | {row['threshold']:.2f} | "
            f"{b['loss']:.4f} | {b.get('attack_recall', 0):.2f} | "
            f"{b.get('poison_at_k_low', 0):.2f}/{b.get('poison_at_k_mid', 0):.2f}/{b.get('poison_at_k_high', 0):.2f} | "
            f"{vm_row.get('GT_Recall', b['gtr_utility']):.2f} | {b['clean_keep']:.2f} | "
            f"{b.get('security_feasible', False)} | {b['feasible']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_compare_md(best, max_cos, sim_sweep, test_results, path: Path):
    by_place = test_results.get("by_placement") or test_results
    mixed = test_results.get("mixed") or {}
    lines = [
        "# Cloud: similarity vs Trust (TEST)",
        "",
        f"- Trust frozen from cloud VAL: δ0={best['delta0']}, k={best['k']}, thr={best['threshold']}",
        f"- Retrieval top_k={TOP_K}",
        f"- Similarity min_cosine (mixed VAL): **{max_cos:.2f}** (keep if cos ≥ τ)",
        f"- Mixed placement: v1→low, v2→mid, v3→high",
        "",
        "## TEST — mixed placement",
        "",
        "| method | Poison@k | GTR |",
        "|---|---:|---:|",
    ]
    for mode in ("acl", "similarity_only", "trust", "similarity_trust"):
        s = mixed.get(mode) or by_place.get(mode, {}).get("low", {})
        lines.append(
            f"| {mode} | {s.get('Poison_at_k', 0):.2f} | {s.get('GT_Recall', 0):.2f} |"
        )
    lines += [
        "",
        "## TEST — single-bucket ablation (Poison@k)",
        "",
        "| method | low | mid | high |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("acl", "similarity_only", "trust", "similarity_trust"):
        r = by_place.get(mode) or test_results.get(mode, {})
        lines.append(
            f"| {mode} | {r.get('low', {}).get('Poison_at_k', 0):.2f} | "
            f"{r.get('mid', {}).get('Poison_at_k', 0):.2f} | {r.get('high', {}).get('Poison_at_k', 0):.2f} |"
        )
    lines += [
        "",
        "## VAL similarity min_cosine sweep (mixed)",
        "",
        "Rule: **keep** chunk if cos(query, content) ≥ τ; drop low-sim.",
        "",
        "| min_cos | Poison@k | GTR |",
        "|---:|---:|---:|",
    ]
    for row in sim_sweep:
        lines.append(
            f"| {row['min_cosine']:.2f} | {row.get('Poison_at_k', 0):.2f} | "
            f"{row.get('GT_Recall', 0):.2f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    global TOP_K
    ap = argparse.ArgumentParser(description="Cloud Trust param select + sim compare")
    ap.add_argument("--skip-select", action="store_true", help="Load selected_trust_cloud.json")
    ap.add_argument("--top-k", type=int, default=5, help="Retrieval top-k (default 5)")
    ap.add_argument(
        "--loss-only",
        action="store_true",
        help="Pure L selection: no Stage1 Poison@k floors, no Stage2 utility floor",
    )
    ap.add_argument(
        "--soft",
        action="store_true",
        help="Soft constraints: L uses Poison@k + soft GTR/clean targets (no +100 hard gate)",
    )
    ap.add_argument(
        "--hard",
        action="store_true",
        help="Legacy hard Stage1/2 selection (default if neither --soft nor --loss-only)",
    )
    ap.add_argument(
        "--no-writer",
        action="store_true",
        help="Ablation: T = T_doc × T_hier only (default main includes T_writer)",
    )
    ap.add_argument(
        "--include-writer",
        action="store_true",
        help="Deprecated: T_writer is on by default; use --no-writer to turn off",
    )
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Output filename suffix, e.g. k10_soft → cloud_param_select_loss_k10_soft.md",
    )
    ap.add_argument(
        "--alpha-ar",
        type=float,
        default=1.0,
        help="Loss weight on (1−AttackRecall) on mixed VAL (hard/pure modes)",
    )
    ap.add_argument(
        "--alpha-pak",
        type=float,
        default=1.0,
        help="Soft-mode weight on mixed Poison@k",
    )
    ap.add_argument("--beta-gtr", type=float, default=1.0, help="Utility loss weight on (1-GTR)")
    ap.add_argument("--zeta-clean", type=float, default=0.5, help="Utility loss weight on (1-clean_keep)")
    ap.add_argument("--lambda-rsr", type=float, default=0.0, help="Unused (RSR dropped; kept for CLI compat)")
    ap.add_argument("--lambda-util", type=float, default=1.0, help="Soft weight on utility shortfall")
    ap.add_argument("--min-gtr", type=float, default=0.15)
    ap.add_argument("--min-clean", type=float, default=0.15)
    ap.add_argument(
        "--no-block-low-mid",
        action="store_true",
        help="Disable Poison@k_low=Poison@k_mid=0 security constraint (hard mode)",
    )
    args = ap.parse_args()
    # Main formula includes T_writer; --no-writer is ablation only.
    args.include_writer = not bool(args.no_writer)
    TOP_K = int(args.top_k)

    os.environ["LOCAL_MOCK_IAM"] = "0"
    os.environ["SKIP_AWS"] = "1"
    random.seed(SEED)

    if args.soft:
        mode_name = "soft"
    elif args.loss_only:
        mode_name = "pure"
    else:
        mode_name = "hard"

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    select_md = OUT / f"cloud_param_select_loss{suffix}.md"
    select_json = OUT / f"cloud_param_select_loss{suffix}.json"
    selected_path = OUT / f"selected_trust_cloud{suffix}.json"
    compare_md = OUT / f"cloud_sim_vs_trust{suffix}.md"
    compare_json = OUT / f"cloud_sim_vs_trust{suffix}.json"

    if mode_name == "soft":
        loss_weights = LossWeights(
            soft_constraint=True,
            pure_loss=False,
            require_block_low_mid=False,
            alpha_poison_at_k=args.alpha_pak,
            alpha_attack_recall=0.0,
            beta_gtr=args.beta_gtr,
            zeta_clean=args.zeta_clean,
            lambda_rsr=args.lambda_rsr,
            lambda_util=args.lambda_util,
            min_gtr=args.min_gtr,
            min_clean_keep=args.min_clean,
        )
    elif mode_name == "pure":
        loss_weights = LossWeights(
            soft_constraint=False,
            pure_loss=True,
            require_block_low_mid=False,
            alpha_attack_recall=args.alpha_ar,
            alpha_poison_at_k=0.0,
            beta_gtr=args.beta_gtr,
            zeta_clean=args.zeta_clean,
            min_gtr=0.0,
            min_clean_keep=0.0,
        )
    else:
        loss_weights = LossWeights(
            soft_constraint=False,
            pure_loss=False,
            alpha_attack_recall=args.alpha_ar,
            alpha_poison_at_k=0.0,
            beta_gtr=args.beta_gtr,
            zeta_clean=args.zeta_clean,
            min_gtr=args.min_gtr,
            min_clean_keep=args.min_clean,
            require_block_low_mid=not args.no_block_low_mid,
        )

    print(
        f"[INFO] top_k={TOP_K} mode={mode_name} include_writer={args.include_writer} "
        f"out=*{suffix or '(default)'}",
        flush=True,
    )

    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    with open(ACCESS_MODEL, encoding="utf-8") as f:
        access_model = json.load(f)
    with open(QUESTION_LIST, encoding="utf-8") as f:
        all_qs = json.load(f)

    poison_names = {d["file_name"] for d in man["docs"]}
    docs = unique_parent_docs(man["docs"])
    poison_qs = {d["question"] for d in docs}
    val_docs, test_docs = split_val_test(docs, SEED)
    clean_pool = [q for q in all_qs if q.get("question") not in poison_qs]
    rng = random.Random(SEED)
    clean_val = rng.sample(clean_pool, min(N_CLEAN_VAL, len(clean_pool)))
    clean_test = rng.sample(clean_pool, min(N_CLEAN_TEST, len(clean_pool)))

    vs = load_vector_store()
    ms0 = MetadataStore()
    ms0.load(str(ARTIFACTS / "metadata.json"))
    um = CredentialManager(base_path=str(ARTIFACTS), file_name="test_credential.txt")
    pr = PermissionRetriever(ms0, PermissionRetrieverConfig(max_workers=8, use_cache=True))
    ac = um.get_ac_manager(VICTIM)

    print(f"[INFO] val={len(val_docs)} test={len(test_docs)} clean_val={len(clean_val)}", flush=True)
    print("[INFO] precompute retrieve+ACL (real GCP, once)...", flush=True)
    cached_val = precompute_acl(val_docs, vs, pr, ac, "val")
    cached_test = precompute_acl(test_docs, vs, pr, ac, "test")
    cached_clean_val = precompute_acl(clean_val, vs, pr, ac, "clean_val")

    OUT.mkdir(parents=True, exist_ok=True)

    if args.skip_select and selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        best = selected["best"]
        ranked = selected.get("ranked_top30", [best])
        print(f"[STEP1] loaded {selected_path}", flush=True)
    else:
        print("[STEP1] joint (δ0,k,thr) selection on VAL via loss...", flush=True)
        best, ranked = step1_select_trust(
            access_model,
            cached_val,
            cached_clean_val,
            ms0,
            poison_names,
            loss_weights,
            "gcp",
            include_writer=args.include_writer,
        )
        assert best is not None
        bd_sel = best["loss_breakdown"]
        print(
            f"[SELECTED] d0={best['delta0']} k={best['k']} thr={best['threshold']} "
            f"P@k L/M/H={bd_sel.get('poison_at_k_low', 0):.2f}/"
            f"{bd_sel.get('poison_at_k_mid', 0):.2f}/{bd_sel.get('poison_at_k_high', 0):.2f} "
            f"AR={bd_sel.get('attack_recall', 0):.2f} "
            f"GTR={bd_sel.get('gtr_utility', 0):.2f} L={bd_sel['loss']:.4f}",
            flush=True,
        )
        selected = {
            "best": {
                "delta0": best["delta0"],
                "k": best["k"],
                "threshold": best["threshold"],
                "loss_breakdown": best["loss_breakdown"],
                "val_mixed": best.get("val_mixed"),
                "val_placement": best.get("val_placement"),
                "clean_keep": best["clean_keep"],
            },
            "loss_weights": loss_weights.to_dict(),
            "provider_scope": "gcp",
            "placement_protocol": {
                "soft": "soft_poison_at_k_plus_util",
                "pure": "pure_loss_no_constraints",
                "hard": "cloud_val_pak_low_mid_zero_then_max_utility",
            }[mode_name],
            "selection_mode": mode_name,
            "top_k": TOP_K,
            "include_writer": args.include_writer,
            "writer_mode": "level" if args.include_writer else None,
            "grid": {"DELTA0S": DELTA0S, "KS": KS, "THRS": THRS},
            "ranked_top30": [
                {
                    "delta0": r["delta0"],
                    "k": r["k"],
                    "threshold": r["threshold"],
                    "loss_breakdown": r["loss_breakdown"],
                    "clean_keep": r["clean_keep"],
                }
                for r in ranked[:30]
            ],
        }
        selected_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
        write_select_md(best, ranked, loss_weights, select_md)
        select_json.write_text(
            json.dumps(
                {"selected": selected, "val_size": len(val_docs), "test_size": len(test_docs), "top_k": TOP_K},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[DONE] {select_md}", flush=True)

    # STEP2: similarity calibrate + compare
    print("[STEP2] calibrate similarity τ on VAL...", flush=True)
    max_cos, sim_sweep = calibrate_sim_tau(cached_val, ms0, poison_names, vs)
    print(f"[CALIB] min_cosine={max_cos:.2f} (keep if cos >= tau)", flush=True)

    tf = make_tf(
        access_model,
        best["delta0"],
        best["k"],
        best["threshold"],
        "gcp",
        include_writer=args.include_writer,
    )
    sim = SimilarityPoisonFilter(min_cosine=max_cos)

    print("[STEP2] TEST compare: acl / sim / trust / sim+trust...", flush=True)
    test_results = eval_modes_on_test(cached_test, ms0, poison_names, access_model, tf, sim, vs)

    compare_payload = {
        "trust": {
            "delta0": best["delta0"],
            "k": best["k"],
            "threshold": best["threshold"],
            "val_loss": best.get("loss_breakdown") or best.get("loss_breakdown"),
        },
        "similarity": {
            "min_cosine": max_cos,
            "rule": "keep if cos(query,content) >= min_cosine",
            "val_sweep": sim_sweep,
        },
        "test_results": test_results,
        "protocol": {
            "victim": VICTIM,
            "seed": SEED,
            "top_k": TOP_K,
            "include_writer": args.include_writer,
            "selection_mode": mode_name,
            "loss_only": args.loss_only,
            "soft": args.soft,
        },
    }
    compare_json.write_text(
        json.dumps(compare_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_compare_md(best, max_cos, sim_sweep, test_results, compare_md)
    print(f"[DONE] {compare_md}", flush=True)


if __name__ == "__main__":
    main()
