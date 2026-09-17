"""VAL loss for trust-filter hyper-parameter (δ0, k, thr) selection on real GCP.

Modes
-----
1) soft (default for new runs): no Stage 1/2 hard gates; everything enters ranking.

    L = α·Poison@k
      + β·(1−GTR) + ζ·(1−clean_keep)
      + λ_u·[relu(min_gtr−GTR) + relu(min_clean−clean_keep)]

2) hard (legacy): Stage1 Poison@k_low=Poison@k_mid=0, Stage2 GTR/clean floors, else +100.

3) pure_loss: L = (1−AR)+β(1−GTR)+ζ(1−clean) only.

Joint grid on cloud VAL → freeze → report TEST.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


PlacementMetrics = Mapping[str, Mapping[str, float]]


@dataclass
class LossWeights:
    """Weights for cloud VAL selection loss."""

    alpha_poison_at_k: float = 1.0
    alpha_attack_recall: float = 0.0
    beta_gtr: float = 1.0
    zeta_clean: float = 0.5
    lambda_rsr: float = 0.0  # unused; RSR dropped
    lambda_util: float = 1.0
    min_gtr: float = 0.15
    min_clean_keep: float = 0.15
    hard_penalty: float = 100.0
    require_block_low_mid: bool = False
    pure_loss: bool = False
    soft_constraint: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LossBreakdown:
    loss: float
    poison_at_k: float
    gtr_utility: float
    clean_keep: float
    attack_recall: float
    feasible: bool
    security_feasible: bool
    hard_penalty: float
    poison_at_k_low: float = 0.0
    poison_at_k_mid: float = 0.0
    poison_at_k_high: float = 0.0
    attack_recall_low: float = 0.0
    attack_recall_mid: float = 0.0
    attack_recall_high: float = 0.0
    gtr_low: float = 0.0
    gtr_mid: float = 0.0
    gtr_high: float = 0.0
    mode: str = "soft"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get(metrics: PlacementMetrics, placement: str, key: str) -> float:
    return float(metrics[placement].get(key, 0.0))


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def compute_val_loss(
    stats: Mapping[str, float],
    weights: Optional[LossWeights] = None,
    clean_keep: Optional[float] = None,
    placement_breakdown: Optional[PlacementMetrics] = None,
) -> LossBreakdown:
    """Compute scalar VAL loss from mixed attack metrics + per-bucket breakdown."""
    w = weights or LossWeights()

    poison_at_k = float(stats.get("Poison_at_k", stats.get("poison_at_k", 0.0)))
    gtr_utility = float(stats.get("GT_Recall", 0.0))
    clean_keep_rate = 0.0 if clean_keep is None else float(clean_keep)

    pak_low = pak_mid = pak_high = 0.0
    ar_low = ar_mid = ar_high = 0.0
    gtr_low = gtr_mid = gtr_high = 0.0
    if placement_breakdown:
        pak_low = _get(placement_breakdown, "low", "Poison_at_k")
        pak_mid = _get(placement_breakdown, "mid", "Poison_at_k")
        pak_high = _get(placement_breakdown, "high", "Poison_at_k")
        ar_low = _get(placement_breakdown, "low", "Attack_Recall")
        ar_mid = _get(placement_breakdown, "mid", "Attack_Recall")
        ar_high = _get(placement_breakdown, "high", "Attack_Recall")
        gtr_low = _get(placement_breakdown, "low", "GT_Recall")
        gtr_mid = _get(placement_breakdown, "mid", "GT_Recall")
        gtr_high = _get(placement_breakdown, "high", "GT_Recall")

    attack_recall = float(stats.get("Attack_Recall", stats.get("attack_recall", 0.0)))

    if w.require_block_low_mid and placement_breakdown is not None:
        security_feasible = pak_low == 0.0 and pak_mid == 0.0
    elif w.require_block_low_mid:
        security_feasible = False
    else:
        security_feasible = True

    utility_feasible = gtr_utility >= w.min_gtr and clean_keep_rate >= w.min_clean_keep
    feasible = security_feasible and utility_feasible

    util_shortfall = _relu(w.min_gtr - gtr_utility) + _relu(w.min_clean_keep - clean_keep_rate)

    if w.soft_constraint and not w.pure_loss:
        loss = (
            w.alpha_poison_at_k * poison_at_k
            + w.beta_gtr * (1.0 - gtr_utility)
            + w.zeta_clean * (1.0 - clean_keep_rate)
            + w.lambda_util * util_shortfall
        )
        hard_penalty = 0.0
        feasible = True
        security_feasible = True
        mode = "soft"
    elif w.pure_loss:
        loss = (
            w.alpha_attack_recall * (1.0 - attack_recall)
            + w.beta_gtr * (1.0 - gtr_utility)
            + w.zeta_clean * (1.0 - clean_keep_rate)
        )
        hard_penalty = 0.0
        feasible = True
        mode = "pure"
    elif feasible:
        loss = (
            w.alpha_attack_recall * (1.0 - attack_recall)
            + w.beta_gtr * (1.0 - gtr_utility)
            + w.zeta_clean * (1.0 - clean_keep_rate)
        )
        hard_penalty = 0.0
        mode = "hard_ok"
    else:
        hard_penalty = w.hard_penalty
        loss = (
            hard_penalty
            + w.alpha_attack_recall * (1.0 - attack_recall)
            + pak_low
            + pak_mid
            + util_shortfall
        )
        mode = "hard_infeasible"

    return LossBreakdown(
        loss=round(loss, 6),
        poison_at_k=round(poison_at_k, 6),
        gtr_utility=round(gtr_utility, 6),
        clean_keep=round(clean_keep_rate, 6),
        attack_recall=round(attack_recall, 6),
        feasible=feasible,
        security_feasible=security_feasible,
        hard_penalty=hard_penalty,
        poison_at_k_low=round(pak_low, 6),
        poison_at_k_mid=round(pak_mid, 6),
        poison_at_k_high=round(pak_high, 6),
        attack_recall_low=round(ar_low, 6),
        attack_recall_mid=round(ar_mid, 6),
        attack_recall_high=round(ar_high, 6),
        gtr_low=round(gtr_low, 6),
        gtr_mid=round(gtr_mid, 6),
        gtr_high=round(gtr_high, 6),
        mode=mode,
    )


def select_best_params(
    grid: List[Dict[str, Any]],
    weights: Optional[LossWeights] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pick best row by min loss (soft/pure) or security→feasible→min loss (hard)."""
    w = weights or LossWeights()
    ranked: List[Dict[str, Any]] = []

    for row in grid:
        enriched = {**row}
        if "loss_breakdown" not in row:
            stats = row.get("val_mixed") or row.get("val") or {}
            bd = compute_val_loss(
                stats,
                weights,
                clean_keep=row.get("clean_keep"),
                placement_breakdown=row.get("val_placement"),
            )
            enriched["loss_breakdown"] = bd.to_dict()
        ranked.append(enriched)

    if w.soft_constraint or w.pure_loss:
        sort_key = lambda r: (
            r["loss_breakdown"]["loss"],
            r["loss_breakdown"]["poison_at_k"],
            r["loss_breakdown"].get("poison_at_k_low", 0) + r["loss_breakdown"].get("poison_at_k_mid", 0),
            -r["loss_breakdown"]["gtr_utility"],
            -r["loss_breakdown"]["clean_keep"],
            r.get("threshold", 0.0),
        )
    else:
        sort_key = lambda r: (
            0 if r["loss_breakdown"]["security_feasible"] else 1,
            0 if r["loss_breakdown"]["feasible"] else 1,
            r["loss_breakdown"]["loss"],
            1.0 - r["loss_breakdown"]["attack_recall"],
            r["loss_breakdown"].get("poison_at_k_low", 0) + r["loss_breakdown"].get("poison_at_k_mid", 0),
            r["loss_breakdown"]["poison_at_k"],
            -r["loss_breakdown"]["gtr_utility"],
            -r["loss_breakdown"]["clean_keep"],
            r.get("threshold", 0.0),
        )

    ranked.sort(key=sort_key)
    best = ranked[0] if ranked else None
    return best, ranked


def loss_formula_markdown(weights: Optional[LossWeights] = None) -> str:
    w = weights or LossWeights()
    if w.soft_constraint and not w.pure_loss:
        return "\n".join(
            [
                "## Cloud VAL selection (soft constraints, no hard Stage 1/2)",
                "",
                "Loss on mixed VAL (lower is better):",
                "```",
                f"L = {w.alpha_poison_at_k}·Poison@k + {w.beta_gtr}·(1−GTR) + {w.zeta_clean}·(1−clean_keep)",
                f"  + {w.lambda_util}·[relu({w.min_gtr}−GTR) + relu({w.min_clean_keep}−clean_keep)]",
                "```",
                "",
                "- No hard reject / +100 gate",
                "- Joint grid (δ0, k, thr); freeze; report TEST Poison@k + True ASR",
            ]
        )
    if w.pure_loss:
        return "\n".join(
            [
                "## Cloud VAL selection (pure loss, no Stage 1/2 constraints)",
                "",
                "Loss on mixed VAL (lower is better):",
                "```",
                f"L = {w.alpha_attack_recall}·(1−AR) + {w.beta_gtr}·(1−GTR)",
                f"  + {w.zeta_clean}·(1−clean_keep)",
                "```",
                "",
                "- Joint grid over (δ0, k, thr); freeze; report TEST",
            ]
        )
    sec = "Poison@k_low=0 AND Poison@k_mid=0" if w.require_block_low_mid else "(no bucket Poison@k constraint)"
    return "\n".join(
        [
            "## Cloud VAL selection protocol (GCP-only, no mock)",
            "",
            "**Stage 1 — security** (single-bucket VAL):",
            f"  {sec}",
            "",
            "**Stage 2 — utility floor**:",
            f"  GTR >= {w.min_gtr}, clean_keep >= {w.min_clean_keep}",
            "",
            "**Stage 3 — pick best among feasible**:",
            "  maximize mixed AttackRecall -> GTR -> clean_keep -> lower thr",
            "",
            "Loss (lower is better):",
            "```",
            f"L = {w.alpha_attack_recall}·(1−AR) + {w.beta_gtr}·(1−GTR)",
            f"  + {w.zeta_clean}·(1−clean_keep)   if security + utility OK",
            f"L = {w.hard_penalty} + {w.alpha_attack_recall}·(1−AR) + P@k_low + P@k_mid",
            "  + utility_violation   otherwise",
            "```",
            "",
            "- Selection: joint grid over (δ0, k, thr) on cloud VAL; freeze; report TEST",
        ]
    )
