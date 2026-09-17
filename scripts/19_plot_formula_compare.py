"""Compare Trust combine formulas (VAL + TEST)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = FIG / "formula_compare.png"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))

# --- left: VAL selected keep metrics ---
ax = axes[0]
labels = [
    "ACL",
    "product\nno writer",
    "logistic\n+ writer",
    "logistic-max\n(max-gap)",
    "logistic-max\n(other cut)",
]
pak = [0.611, 0.368, 0.368, 0.368, 0.304]
ck = [1.000, 0.615, 0.615, 0.615, 0.530]
x = np.arange(len(labels))
w = 0.38
b1 = ax.bar(x - w / 2, pak, w, color="#c0392b", label="Poison@k")
b2 = ax.bar(x + w / 2, ck, w, color="#188038", label="clean_keep")
ax.set_ylim(0, 1.15)
ax.set_ylabel("VAL metric")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title("VAL: swapping the formula (same 125+40 Qs)", loc="left")
ax.axhline(0.368, color="#c0392b", lw=0.8, ls=":", alpha=0.5)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", color="#e8eaed", lw=0.8)
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, h + 0.02,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
        )

# --- right: TEST product + appendix ---
ax = axes[1]
labels2 = [
    "ACL\n(open poison)",
    "product\n+ writer",
    "product\nno writer",
    "ACL\n(private poison)",
    "product+writer\n(private poison)",
]
pak2 = [0.603, 0.363, 0.363, 0.603, 0.603]
ck2 = [1.000, 0.605, 0.605, 1.000, 0.610]
asr = [0.63, 0.44, 0.44, 0.63, 0.70]
x = np.arange(len(labels2))
b1 = ax.bar(x - w / 2, pak2, w, color="#c0392b", label="Poison@k")
b2 = ax.bar(x + w / 2, ck2, w, color="#188038", label="clean_keep")
ax.plot(x, asr, color="#1a73e8", marker="o", lw=1.6, label="True ASR")
ax.set_ylim(0, 1.15)
ax.set_ylabel("TEST metric")
ax.set_xticks(x)
ax.set_xticklabels(labels2, fontsize=8.5)
ax.set_title("TEST: product formula, writer on vs off", loc="left")
ax.legend(frameon=False, loc="upper right", ncol=1)
ax.grid(axis="y", color="#e8eaed", lw=0.8)
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, h + 0.02,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
        )

fig.suptitle(
    "Changing T does not change the keep-set (except logistic-max's extra cut)",
    fontsize=12, y=1.02,
)
fig.text(
    0.01, -0.02,
    "Sources: cloud_ladder_val_nowriter.md, cloud_ladder_val_logistic.md, "
    "cloud_ladder_val_logistic_max.md, cloud_ladder_main_k5.md. "
    "product/logistic: 30/30 cells = 0.368 / 0.615. logistic-max mid: 8 cells = 0.368, 22 cells = 0.304.",
    fontsize=7.5, color="#5f6368",
)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(OUT)
