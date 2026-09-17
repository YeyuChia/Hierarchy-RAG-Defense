"""Why all 30 VAL (delta0, k) cells keep the same documents."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = FIG / "why_thirty_configs_same.png"

LDOC = {
    "public": 0.03125,
    "team": 0.40625,
    "dept": 0.65625,
    "restr": 0.8125,
    "priv": 0.90625,
}
LU = 0.90625
GAPS = {k: max(0.0, LU - v) for k, v in LDOC.items()}
D0S = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
KS = [0.08, 0.10, 0.12, 0.16, 0.20]


def sig(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# VAL tau = (L+R)/2 from cloud_ladder_val_nowriter.md
TAU = [
    0.045, 0.064, 0.080, 0.108, 0.129,
    0.076, 0.094, 0.110, 0.135, 0.154,
    0.119, 0.134, 0.146, 0.166, 0.181,
    0.173, 0.179, 0.187, 0.199, 0.209,
    0.229, 0.228, 0.230, 0.235, 0.239,
    0.282, 0.277, 0.274, 0.271, 0.269,
]

xs = list(range(30))
labels = [f"{d0:.2f}/{k:.2f}" for d0 in D0S for k in KS]
T = {name: [] for name in LDOC}
for d0 in D0S:
    for k in KS:
        for name, ld in LDOC.items():
            th = 1.0 - sig((GAPS[name] - d0) / k)
            T[name].append(ld * th)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
})

fig, axes = plt.subplots(
    2, 1, figsize=(11.2, 7.4), sharex=True,
    gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.12},
)
ax, ax2 = axes

ax.fill_between(
    xs, T["team"], T["dept"],
    color="#cfe8d4", alpha=0.85, zorder=0,
    label="empty gap: T_team < · < T_dept",
)
ax.plot(xs, T["public"], color="#9aa0a6", lw=1.4, label="T public (drop)")
ax.plot(xs, T["team"], color="#c0392b", lw=2.0, label="T team = L (drop)")
ax.plot(xs, TAU, color="#1a73e8", lw=2.2, ls="--", label="τ = (L+R)/2")
ax.plot(xs, T["dept"], color="#188038", lw=2.0, label="T dept = R (keep)")
ax.plot(xs, T["restr"], color="#5f6368", lw=1.3, alpha=0.85, label="T restricted (keep)")
ax.plot(xs, T["priv"], color="#202124", lw=1.3, label="T private (keep)")

ax.set_ylabel("trust score  T")
ax.set_ylim(-0.02, 1.0)
ax.set_xlim(-0.5, 29.5)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.grid(axis="y", color="#e8eaed", lw=0.8)
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.5)
ax.set_title(
    "All 30 VAL configs cut the same ladder rung\n"
    r"$T=T_{\mathrm{doc}}\times T_{\mathrm{hier}}$, no writer, user-31",
    loc="left", pad=10,
)
ax.annotate(
    "τ always sits in this empty band\n→ drop buckets 1–2, keep 3–5",
    xy=(25, 0.20), xytext=(16.2, 0.02),
    fontsize=8.5, color="#188038",
    arrowprops=dict(arrowstyle="->", color="#188038", lw=1.0),
)

ax2.axhline(0.368, color="#c0392b", lw=2.0, label="Poison@k = 0.368")
ax2.axhline(0.615, color="#188038", lw=2.0, label="clean_keep = 0.615")
ax2.set_ylim(0, 1)
ax2.set_ylabel("VAL metric")
ax2.set_xlabel(r"$(\delta_0,\,k)$")
ax2.set_xticks(xs)
ax2.set_xticklabels(labels, rotation=90, fontsize=7.2)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax2.grid(axis="y", color="#e8eaed", lw=0.8)
ax2.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=2)
ax2.set_xlim(-0.5, 29.5)

fig.text(
    0.01, 0.01,
    "Source: theoretical T per bucket + VAL τ from cloud_ladder_val_nowriter.md. "
    "30/30 cells have T_team < τ < T_dept.",
    fontsize=7.5, color="#5f6368",
)
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16)
fig.savefig(OUT, dpi=160)
print(OUT)
