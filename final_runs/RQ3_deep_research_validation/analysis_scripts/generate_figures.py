#!/usr/bin/env python3
"""
Generate RQ3 Deep Research figures for thesis.

Updated 2025-12-18: Now uses full 285-edge dataset from all 3 CLDs
(Depressive, Social, Older Persons) instead of 229-edge subset.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

# Input/output roots (support rerouting outputs under a single run folder)
# Default INPUT_DIR to the RQ3_deep_research folder (parent of analysis_scripts)
INPUT_DIR = Path(os.environ.get("RQ3_INPUT_DIR", str(Path(__file__).parent.parent))).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("RQ3_OUTPUT_DIR", str(INPUT_DIR))).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the full 285-edge dataset (all 3 CLDs) to support true distribution plots.
DATA_DIR = INPUT_DIR / "Data"
DATA_FILES = [
    "deep_research_results_depressive_symptoms_20251013_032002_84edges.json",
    "deep_research_results_social_norms_20251012_213641_17edges.json",
    "deep_research_results_older_persons_ALL_EDGES_184edges.json",
]


def _load_all_results():
    all_rows = []
    for fn in DATA_FILES:
        fp = DATA_DIR / fn
        with open(fp, "r") as f:
            data = json.load(f)
        rows = data["results"] if isinstance(data, dict) and "results" in data else data
        all_rows.extend(rows)
    return all_rows


def _summarize(rows):
    by_cls = {"TP": [], "FP": [], "FN": []}
    by_cls_found = {"TP": 0, "FP": 0, "FN": 0}
    by_cls_total = {"TP": 0, "FP": 0, "FN": 0}

    for r in rows:
        cls = r.get("classification")
        if cls not in by_cls_total:
            continue
        by_cls_total[cls] += 1
        if r.get("direct_causal_found"):
            by_cls_found[cls] += 1
        # Use self-assessed confidence for *all edges* (matches Table 13).
        conf = r.get("confidence")
        if conf is not None:
            by_cls[cls].append(conf)

    summary = {}
    for cls in ["TP", "FP", "FN"]:
        total = by_cls_total[cls]
        found = by_cls_found[cls]
        rate = 100.0 * found / total if total else 0.0
        conf_vals = by_cls[cls]
        avg_conf = float(np.mean(conf_vals)) if conf_vals else 0.0
        summary[cls] = {
            "total": total,
            "found": found,
            "rate": rate,
            "conf": avg_conf,
            "conf_values": conf_vals,
            "label": {
                "TP": "TP\n(Expert-accepted)",
                "FP": "FP\n(LLM hallucinations)",
                "FN": "FN\n(Expert-rejected)",
            }[cls],
        }
    return summary


rows = _load_all_results()
data = _summarize(rows)

# Load aggregate pairwise stats (keeps figure annotations consistent with Table 13).
STATS_PATH = OUTPUT_DIR / "rq3_comprehensive_stats.json"
rq3_stats = None
if STATS_PATH.exists():
    with open(STATS_PATH, "r") as f:
        rq3_stats = json.load(f).get("statistical_tests", None)

output_dir = OUTPUT_DIR / 'figures'
output_dir.mkdir(exist_ok=True)

# Figure 1: Detection Rate Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

classifications = ['TP', 'FP', 'FN']
rates = [data[c]['rate'] for c in classifications]
labels = [data[c]['label'] for c in classifications]
colors = ['#2ecc71', '#e74c3c', '#f39c12']  # Green, Red, Orange

bars = ax.bar(labels, rates, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

# Add value labels on bars
for bar, rate, cls in zip(bars, rates, classifications):
    height = bar.get_height()
    ax.annotate(f'{rate:.1f}%\n({data[cls]["found"]}/{data[cls]["total"]})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

# Add horizontal line at 50% for reference
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Random baseline (50%)')

# Add TP-FP gap annotation
ax.annotate('', xy=(0.15, 70.5), xytext=(1.15, 62.4),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(0.65, 74, 'Gap: 8.1 pp\n(MODERATE)', ha='center', va='bottom', fontsize=11, fontweight='bold', color='orange')

ax.set_ylabel('Detection Rate (%)', fontweight='bold')
ax.set_xlabel('Edge Classification', fontweight='bold')
ax.set_title('RQ3: Deep Research Detection of Direct Causal Evidence\nby Edge Classification (n=285 edges)', fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='lower right')

# Add interpretation box
textstr = 'Key Finding: TP shows higher detection\n(70.5%) than FP (62.4%), gap = 8.1 pp\n→ DR shows moderate discrimination ability'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.25, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / 'rq3_detection_rate.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'rq3_detection_rate.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'rq3_detection_rate.png'}")

# Figure 2: Confidence Score Comparison
fig, ax = plt.subplots(figsize=(8, 5))

conf_values = [data[c]["conf_values"] for c in classifications]

# True distribution plot (violin + jittered points) to show overlap/spread.
parts = ax.violinplot(conf_values, showmeans=True, showmedians=True, showextrema=False)
for pc, color in zip(parts["bodies"], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor("black")
    pc.set_alpha(0.6)
parts["cmeans"].set_color("black")
parts["cmedians"].set_color("black")

# Jittered points (subsample FP for readability)
rng = np.random.default_rng(0)
for i, (cls, vals, color) in enumerate(zip(classifications, conf_values, colors), start=1):
    vals = np.array(vals, dtype=float)
    if cls == "FP" and len(vals) > 80:
        vals = rng.choice(vals, size=80, replace=False)
    x = rng.normal(loc=i, scale=0.06, size=len(vals))
    ax.scatter(x, vals, s=12, alpha=0.35, color=color, edgecolors="none")

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels)
ax.set_ylabel("Confidence Score (1–10)", fontweight="bold")
ax.set_xlabel('Edge Classification', fontweight='bold')
mean_confs = [data[c]["conf"] for c in classifications]
mean_range = max(mean_confs) - min(mean_confs)
ax.set_title(f"RQ3: Deep Research Self-Assessed Confidence (Distributions)\nMean range across classes = {mean_range:.2f}", fontweight="bold")
ax.set_ylim(0, 10)

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'rq3_confidence_scores.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'rq3_confidence_scores.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'rq3_confidence_scores.png'}")

# Figure 3: Combined 2-panel figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Detection Rate
ax = axes[0]
# Replace aggregate bar chart with a more informative view:
# detection rates by CLD (with Wilson 95% CI), showing domain dependence.
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * (n**2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def normalize_cld(name: str) -> str:
    s = (name or "").lower()
    if "depressive" in s:
        return "Depressive"
    if "social" in s:
        return "Social Norms"
    return "Older Persons"


# Aggregate counts by (CLD, class)
clds = ["Depressive", "Social Norms", "Older Persons"]
cls_order = ["TP", "FP", "FN"]
counts = {(cld, cls): 0 for cld in clds for cls in cls_order}
found = {(cld, cls): 0 for cld in clds for cls in cls_order}

for r in rows:
    cls = r.get("classification")
    if cls not in cls_order:
        continue
    cld = normalize_cld(r.get("CLD", ""))
    counts[(cld, cls)] += 1
    if r.get("direct_causal_found"):
        found[(cld, cls)] += 1

x = np.arange(len(clds), dtype=float)
offsets = {"TP": -0.22, "FP": 0.0, "FN": 0.22}
markers = {"TP": "o", "FP": "s", "FN": "D"}
color_map = {"TP": colors[0], "FP": colors[1], "FN": colors[2]}

for cls in cls_order:
    y = []
    yerr_lower = []
    yerr_upper = []
    n_labels = []
    for cld in clds:
        k = found[(cld, cls)]
        n = counts[(cld, cls)]
        lo, hi = wilson_ci(k, n)
        p = (k / n) if n else 0.0
        y.append(100 * p)
        yerr_lower.append(100 * (p - lo))
        yerr_upper.append(100 * (hi - p))
        n_labels.append(n)

    y = np.array(y)
    yerr = np.vstack([yerr_lower, yerr_upper])
    ax.errorbar(
        x + offsets[cls],
        y,
        yerr=yerr,
        fmt=markers[cls],
        markersize=7,
        color=color_map[cls],
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        label=f"{cls}",
    )

# Note: We intentionally avoid adding numeric gap/effect/p-value annotations here.
# Those values are reported in Tables 13–15 to reduce redundancy and keep figures clean.

ax.axhline(y=50, color="gray", linestyle="--", linewidth=1)
ax.set_ylabel("Detection Rate (%)", fontweight="bold")
ax.set_xlabel("CLD (with TP/FP/FN)", fontweight="bold")
ax.set_title("(A) Detection Rates by CLD (Wilson 95% CI)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(clds)
ax.set_ylim(0, 100)
ax.legend(title="Class", loc="lower left")

# Panel B: Confidence by CLD and classification (mirrors Panel A structure exactly)
ax = axes[1]

# Collect confidence values by (CLD, classification)
conf_by_cld_cls = {(cld, cls): [] for cld in clds for cls in cls_order}
for r in rows:
    cls = r.get("classification")
    if cls not in cls_order:
        continue
    cld = normalize_cld(r.get("CLD", ""))
    conf = r.get("confidence")
    if conf is not None:
        conf_by_cld_cls[(cld, cls)].append(conf)

x = np.arange(len(clds), dtype=float)

for cls in cls_order:
    y = []
    yerr = []
    for cld in clds:
        vals = conf_by_cld_cls[(cld, cls)]
        mean_val = np.mean(vals) if vals else 0.0
        std_val = np.std(vals) if len(vals) > 1 else 0.0
        y.append(mean_val)
        yerr.append(std_val)

    y = np.array(y)
    yerr = np.array(yerr)
    ax.errorbar(
        x + offsets[cls],
        y,
        yerr=yerr,
        fmt=markers[cls],
        markersize=7,
        color=color_map[cls],
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        label=f"{cls}",
    )

ax.set_ylabel("Confidence (1–10)", fontweight="bold")
ax.set_xlabel("CLD (with TP/FP/FN)", fontweight="bold")
ax.set_title("(B) Confidence Scores by CLD (Mean ± SD)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(clds)
ax.set_ylim(0, 10)
ax.legend(title="Class", loc="lower left")

plt.tight_layout()
plt.savefig(output_dir / 'rq3_combined_figure.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'rq3_combined_figure.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'rq3_combined_figure.png'}")

print("\nAll RQ3 figures generated successfully!")







