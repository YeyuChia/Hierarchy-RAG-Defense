"""Audit Trust score ladder under the access model (no GCP calls).

For the querying user (DESIGN.querying_user), L_user is max L_doc among
buckets they can read. T depends only on (bucket, writer_level).
low/mid/high placements follow BUCKET_TIERS.tier_rank.
"""
from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ["SKIP_AWS"] = "1"
os.environ.setdefault("PYTHONPATH", str(ROOT))

from src.retrievers.trust_filter import TrustFilter

ACCESS_MODEL = ROOT / "artifacts" / "generated_access_model.json"
METADATA = ROOT / "artifacts" / "metadata.json"
OUT_MD = ROOT / "artifacts" / "results" / "trust_ladder_audit.md"

DELTA0S = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
KS = [0.08, 0.10, 0.12, 0.16, 0.20]


def make_tf(am, d0, k, include_writer):
    kw = dict(
        access_model=am,
        combine_mode="mul",
        hier_mode="sigmoid",
        include_writer=include_writer,
        threshold=0.0,
        sigmoid_delta0=d0,
        sigmoid_k=k,
        shared_bucket=None,
        shared_doc_cap=1.0,
        provider_scope="gcp",
    )
    if include_writer:
        kw["writer_mode"] = "level"
    return TrustFilter(**kw)


def main():
    am = json.loads(ACCESS_MODEL.read_text(encoding="utf-8"))
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    tiers = am.get("BUCKET_TIERS") or {}
    victim = (am.get("DESIGN") or {}).get("querying_user") or "user-31"
    ranked = sorted(tiers, key=lambda b: int((tiers[b] or {}).get("tier_rank") or 99))
    placements = {
        "low": ranked[0],
        "mid": ranked[1],
        "high": ranked[-1],
    }

    clean = [m for m in meta if not m.get("is_poisoned")]
    poison = [m for m in meta if m.get("is_poisoned")]
    attacker_wl = float(poison[0].get("writer_level", 0.2)) if poison else 0.2

    lines = [f"# Trust ladder audit (victim={victim})", ""]

    tf0 = make_tf(am, 0.10, 0.08, False)
    lines += [
        "## Bucket ladder",
        "",
        "| bucket | tier | openness | L_doc | gap(victim) |",
        "|--------|------|----------|-------|-------------|",
    ]
    for b in ranked:
        m = {"provider": "gcp", "parameters": {"bucket": b}}
        ldoc = tf0.doc_level(m)
        _, luser, gap = tf0.hierarchy_trust(victim, m)
        lines.append(
            f"| {b} | {tiers[b].get('tier')} | {1 - ldoc:.3f} | {ldoc:.3f} | {gap:.3f} |"
        )
    lines.append("")

    print(f"[INFO] victim={victim} placements={placements}", flush=True)
    print("[INFO] scanning grid...", flush=True)
    rows = []
    for d0, k in product(DELTA0S, KS):
        for include_writer in (False, True):
            tf = make_tf(am, d0, k, include_writer)

            def T(m, _tf=tf):
                return float(_tf.score(m, None, user_name=victim)["T"])

            p = {}
            for name, b in placements.items():
                m = {
                    "provider": "gcp",
                    "parameters": {"bucket": b},
                    "writer_level": attacker_wl,
                }
                p[name] = T(m)

            clean_T = np.array([T(m) for m in clean])
            L = max(p["low"], p["mid"])
            rows.append(
                {
                    "delta0": d0,
                    "k": k,
                    "writer": include_writer,
                    "T_low": p["low"],
                    "T_mid": p["mid"],
                    "T_high": p["high"],
                    "L": L,
                    "clean_mean": float(clean_T.mean()),
                    "clean_median": float(np.median(clean_T)),
                    "clean_q25": float(np.percentile(clean_T, 25)),
                    "clean_q50": float(np.percentile(clean_T, 50)),
                    "frac_clean_above_L": float((clean_T > L).mean()),
                }
            )

    for wr in (False, True):
        lines += [
            f"## T grid — include_writer={wr}",
            "",
            "| δ0 | k | T_low | T_mid | T_high | L=max(low,mid) | clean med | clean mean | clean>L |",
            "|----|---|-------|-------|--------|----------------|-----------|------------|---------|",
        ]
        for r in rows:
            if r["writer"] != wr:
                continue
            lines.append(
                f"| {r['delta0']} | {r['k']} | {r['T_low']:.4f} | {r['T_mid']:.4f} | "
                f"{r['T_high']:.4f} | {r['L']:.4f} | {r['clean_median']:.4f} | "
                f"{r['clean_mean']:.4f} | {r['frac_clean_above_L']:.1%} |"
            )
        lines.append("")

    best = max((r for r in rows if not r["writer"]), key=lambda r: r["frac_clean_above_L"])
    lines += [
        "## Best separability (no writer)",
        f"- δ0={best['delta0']} k={best['k']}: L={best['L']:.4f}, "
        f"{best['frac_clean_above_L']:.1%} of clean chunks score above L "
        f"(these survive a thr just above L)",
        f"- T_high={best['T_high']:.4f} → high poison "
        f"{'ALSO blocked' if best['T_high'] <= best['L'] else 'still passes'}",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {OUT_MD}", flush=True)
    print(
        f"[BEST no-writer] d0={best['delta0']} k={best['k']} "
        f"L={best['L']:.4f} clean_above_L={best['frac_clean_above_L']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
