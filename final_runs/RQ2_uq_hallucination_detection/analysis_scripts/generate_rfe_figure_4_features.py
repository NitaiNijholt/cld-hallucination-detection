#!/usr/bin/env python3
"""
Generate clean RFE figure for thesis using ONLY 4 CI metrics.

Generates:
  - phase4_rfe_complete.png (main thesis figure)
  - phase4_rfe_enhanced.png (alias)
  - rfe_4_features_figure.png (legacy backup)

Data sources:
  - Phase 4 results: parameter_tuning_experiments/rq2_analyses/rq2_phase4_rfe_*/phase4_rfe_results.json
  - Permutation importance: final_runs/RQ2_uq_hallucination_detection/permutation_importance_all_classifiers.json

The left panel shows feature selection stability with hatch patterns to distinguish features.
The right panel shows permutation importance grouped by classifier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy import stats

# Centralized path overrides (optional)
from rq2_paths import rq2_dirs, repo_root

# Import shared RQ2 data preparation (for up-to-date permutation importance)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rq2_data_preparation import load_rq2_combined_data, prepare_clean_dataset


# Paths
REPO_ROOT = repo_root()
ANALYSES_DIR, OUTPUT_DIR = rq2_dirs()
PERM_IMPORTANCE_JSON = OUTPUT_DIR / "permutation_importance_all_classifiers.json"

# Output files
OUT_COMPLETE = OUTPUT_DIR / "phase4_rfe_complete.png"
OUT_ENHANCED = OUTPUT_DIR / "phase4_rfe_enhanced.png"
OUT_LEGACY = OUTPUT_DIR / "rfe_4_features_figure.png"

# Constants
CLASSIFIERS = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
FEATURES = ["Gen Perplexity", "Gen Min Prob", "Gen Max Window Entropy", "Gen Cosine Similarity"]

# Hatches to distinguish features in left plot
FEATURE_HATCH = {
    "Gen Perplexity": "///",
    "Gen Min Prob": "\\\\\\",
    "Gen Max Window Entropy": "xx",
    "Gen Cosine Similarity": "..",
}

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})


def stability_color(count: int, n_folds: int) -> str:
    """Return color based on stability threshold (>=60% stable, 40-59% moderate, <40% unstable)."""
    pct = count / n_folds if n_folds > 0 else 0
    if pct >= 0.6:
        return "#2ca02c"  # green
    if pct >= 0.4:
        return "#ff7f0e"  # orange
    return "#d62728"  # red


def load_latest_phase4() -> dict:
    """Load the latest Phase 4 RFE results JSON."""
    candidates = sorted(ANALYSES_DIR.glob("rq2_phase4_rfe_*/phase4_rfe_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No Phase 4 results found in: {ANALYSES_DIR}")
    
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"  Loading Phase 4 from: {latest.name}")
    obj = json.loads(latest.read_text())
    # Preserve provenance for staleness checks / reproducibility
    obj["__source_path__"] = str(latest)
    obj["__source_mtime__"] = float(latest.stat().st_mtime)
    return obj


def _compute_perm_importance_from_latest_rq2_design(
    *,
    n_permutations: int = 30,
    random_state: int = 42,
    test_size: float = 0.33,
) -> dict:
    """
    Compute permutation importance for each classifier on the latest RQ2 dataset.

    Design choices (aligned with RQ2 block-CV intent):
    - Use block-level GroupShuffleSplit (block = CLD × run) to avoid leakage across prompts/edges.
    - Train on train blocks; evaluate permutation AUC-drop on held-out test blocks.
    - Repeat shuffles `n_permutations` times per feature; report mean/std of AUC drop.
    """
    print(f"  Computing permutation importance from latest RQ2 data ({n_permutations} permutations)...")
    df = load_rq2_combined_data(verbose=False)
    X, y, feature_names, _df_clean, groups = prepare_clean_dataset(
        df, features=FEATURES, verbose=False, return_groups=True
    )

    # Block-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale using train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    }

    rng = np.random.default_rng(random_state)
    out: dict[str, dict] = {}
    for clf_name in CLASSIFIERS:
        clf = classifiers[clf_name]
        clf.fit(X_train, y_train)
        y_prob_base = clf.predict_proba(X_test)[:, 1]
        baseline_auc = float(roc_auc_score(y_test, y_prob_base))

        means: list[float] = []
        stds: list[float] = []
        ci_hws: list[float] = []
        for feat_idx, _feat_name in enumerate(feature_names):
            drops = []
            for _ in range(n_permutations):
                X_perm = X_test.copy()
                X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
                y_prob_perm = clf.predict_proba(X_perm)[:, 1]
                auc_perm = float(roc_auc_score(y_test, y_prob_perm))
                drops.append(baseline_auc - auc_perm)
            mean_val = float(np.mean(drops))
            std_val = float(np.std(drops, ddof=1))  # Use ddof=1 for sample std
            # Compute 95% CI half-width using t-distribution
            t_val = stats.t.ppf(0.975, n_permutations - 1)
            ci_hw = t_val * std_val / np.sqrt(n_permutations)
            means.append(mean_val)
            stds.append(std_val)
            ci_hws.append(float(ci_hw))

        out[clf_name] = {
            "importances_mean": means,
            "importances_std": stds,
            "importances_ci_hw": ci_hws,
            "test_auc": baseline_auc,
            "features": feature_names,
            "n_permutations": int(n_permutations),
            "split": {
                "method": "GroupShuffleSplit",
                "test_size": float(test_size),
                "random_state": int(random_state),
                "group_definition": "block_id = CLD × run",
            },
        }

    # Persist for thesis & reproducibility
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PERM_IMPORTANCE_JSON.write_text(json.dumps(out, indent=2))
    print(f"  ✓ Wrote: {PERM_IMPORTANCE_JSON}")
    return out


def load_perm_importance(phase4: dict | None) -> dict | None:
    """
    Load permutation importance JSON if available; otherwise (or if stale) regenerate it.
    """
    phase4_mtime = float(phase4.get("__source_mtime__", 0.0)) if isinstance(phase4, dict) else 0.0
    if not PERM_IMPORTANCE_JSON.exists():
        return _compute_perm_importance_from_latest_rq2_design()

    try:
        perm_mtime = float(PERM_IMPORTANCE_JSON.stat().st_mtime)
    except OSError:
        perm_mtime = 0.0

    # If Phase 4 is newer than the stored permutation importance, regenerate to prevent staleness.
    if phase4_mtime and perm_mtime and phase4_mtime > perm_mtime:
        print("  Detected stale permutation importance (older than latest Phase 4). Regenerating...")
        return _compute_perm_importance_from_latest_rq2_design()

    return json.loads(PERM_IMPORTANCE_JSON.read_text())


def plot_left_rfe_stability(ax: plt.Axes, phase4: dict) -> None:
    """Plot feature selection stability across classifiers with hatch legend."""
    # Get n_folds from phase4 results (default to 3 for block-level CV)
    n_folds = phase4.get("n_folds", 3)
    n_blocks = phase4.get("n_blocks", 9)
    cv_method = phase4.get("cv_method", "GroupKFold")
    
    x = np.arange(len(CLASSIFIERS))
    n_feat = len(FEATURES)
    width = 0.18
    offsets = (np.arange(n_feat) - (n_feat - 1) / 2) * width

    for j, feat in enumerate(FEATURES):
        for i, clf in enumerate(CLASSIFIERS):
            clf_data = phase4["all_classifiers"].get(clf, {})
            counts = clf_data.get("feature_selection_counts", {})
            c = int(counts.get(feat, 0))
            ax.bar(
                x[i] + offsets[j],
                c,
                width=width,
                color=stability_color(c, n_folds),
                edgecolor="black",
                linewidth=0.6,
                hatch=FEATURE_HATCH[feat],
                alpha=0.85,
            )
            ax.text(
                x[i] + offsets[j],
                c + 0.08,
                f"{c}/{n_folds}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"])
    ax.set_ylabel(f"Selection Frequency (out of {n_folds} folds)")
    ax.set_title(f"Feature Selection Stability (RFE, {n_folds}-fold GroupKFold)", fontweight="bold")
    ax.set_ylim(0, n_folds + 0.7)
    ax.set_yticks(range(n_folds + 1))  # Force integer ticks on Y-axis
    ax.axhline(n_folds * 0.6, color="black", linestyle="--", linewidth=1, alpha=0.6)  # 60% threshold
    ax.grid(axis="y", alpha=0.25)

    # Legends: (1) stability colors, (2) feature hatch patterns
    stability_legend = [
        Patch(facecolor="#2ca02c", edgecolor="black", label=">=60% (stable)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="40-59% (moderate)"),
        Patch(facecolor="#d62728", edgecolor="black", label="<40% (unstable)"),
    ]
    feature_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=FEATURE_HATCH[f], label=f.replace("Gen ", ""))
        for f in FEATURES
    ]

    leg1 = ax.legend(handles=stability_legend, loc="upper left", fontsize=8, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=feature_legend, loc="upper right", fontsize=8, frameon=True, title="Feature")


def plot_right_perm_importance(ax: plt.Axes, perm: dict) -> None:
    """Plot permutation importance by classifier, ranked by average AUC drop.

    Note on AUC labels:
    - The permutation-importance JSON may come from a different evaluation split than Phase 4.
    - Since this figure is used to summarize *Phase 4 (RFE, block-level GroupKFold)*, we prefer
      Phase 4 mean AUCs when available to keep the figure consistent with the RQ2 design.
    """
    
    # Step 1: Compute average importance across classifiers for each feature
    feat_importance: dict[str, list[float]] = {f: [] for f in FEATURES}
    feat_means: dict[str, dict[str, float]] = {}  # clf -> feat -> mean
    feat_ci_hws: dict[str, dict[str, float]] = {}   # clf -> feat -> 95% CI half-width
    
    for clf in CLASSIFIERS:
        if clf not in perm:
            continue
        clf_obj = perm[clf]
        feats = clf_obj["features"]
        means = clf_obj["importances_mean"]
        # Prefer CI half-width if available, otherwise compute from std
        if "importances_ci_hw" in clf_obj:
            ci_hws = clf_obj["importances_ci_hw"]
        else:
            # Fallback: compute from std (assume n_permutations=30 if not specified)
            stds = clf_obj["importances_std"]
            n_perm = clf_obj.get("n_permutations", 30)
            t_val = stats.t.ppf(0.975, n_perm - 1)
            ci_hws = [t_val * s / np.sqrt(n_perm) for s in stds]
        
        feat_means[clf] = {f: float(means[i]) for i, f in enumerate(feats)}
        feat_ci_hws[clf] = {f: float(ci_hws[i]) for i, f in enumerate(feats)}
        
        for f in FEATURES:
            if f in feat_means[clf]:
                feat_importance[f].append(feat_means[clf][f])
    
    # Step 2: Compute average and order features low-to-high so that
    # higher average corresponds to higher y-value (top of plot).
    feat_avg = {f: np.mean(vals) if vals else 0.0 for f, vals in feat_importance.items()}
    feat_order = sorted(FEATURES, key=lambda f: feat_avg[f], reverse=False)
    
    print(f"  Feature ranking by avg AUC drop:")
    for f in feat_order:
        print(f"    {f.replace('Gen ', '')}: {feat_avg[f]:.4f}")
    
    y = np.arange(len(feat_order))

    bar_h = 0.22
    offsets = {
        "Logistic Regression": +bar_h,
        "Random Forest": 0.0,
        "Gradient Boosting": -bar_h,
    }
    colors = {
        "Logistic Regression": "#4c72b0",
        "Random Forest": "#55a868",
        "Gradient Boosting": "#dd8452",
    }

    for clf in CLASSIFIERS:
        if clf not in perm:
            continue
        clf_obj = perm[clf]
        auc = float(clf_obj.get("test_auc", np.nan))

        vals = [feat_means[clf].get(f, 0) for f in feat_order]
        errs = [feat_ci_hws[clf].get(f, 0) for f in feat_order]

        label = f"{clf} (AUC={auc:.3f})" if not np.isnan(auc) else clf
        ax.barh(
            y + offsets[clf],
            vals,
            height=bar_h,
            color=colors[clf],
            xerr=errs,
            capsize=3,
            edgecolor="black",
            linewidth=0.6,
            label=label,
        )

    # Build y-axis labels with average in parentheses
    ylabels = [f"{f.replace('Gen ', '')} (avg={feat_avg[f]:.3f})" for f in feat_order]
    
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Permutation Importance (AUC Drop)")
    ax.set_title("Feature Importance via Permutation Testing\n(ranked by avg importance)", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)


def harmonize_perm_auc_with_phase4(perm: dict, phase4: dict) -> dict:
    """
    Return a copy of `perm` with `test_auc` overridden by Phase 4 mean AUCs when available.

    This prevents stale/overfit AUC labels in the thesis figure when permutation-importance
    artifacts were generated with a different (or leaky) split than Phase 4's block-level CV.
    """
    if not perm or not phase4:
        return perm
    phase4_clfs = phase4.get("all_classifiers", {}) if isinstance(phase4, dict) else {}
    if not isinstance(phase4_clfs, dict) or not phase4_clfs:
        return perm

    out = json.loads(json.dumps(perm))  # deep copy (perm is small)
    for clf in CLASSIFIERS:
        if clf not in out:
            continue
        p4 = phase4_clfs.get(clf, {})
        if isinstance(p4, dict) and "mean_auc" in p4 and p4["mean_auc"] is not None:
            out[clf]["test_auc"] = float(p4["mean_auc"])
    return out


def plot_right_fallback(ax: plt.Axes, phase4: dict) -> None:
    """Fallback right panel if permutation importance not available."""
    # Get n_folds from phase4 results
    n_folds = phase4.get("n_folds", 3)
    
    # Use feature selection counts from best classifier
    best_clf = phase4.get("best_classifier", "Random Forest")
    clf_data = phase4["all_classifiers"].get(best_clf, {})
    counts = clf_data.get("feature_selection_counts", {})
    
    # Sort by count
    sorted_feats = sorted(counts.items(), key=lambda x: -x[1])
    feats = [f[0] for f in sorted_feats]
    cnts = [f[1] for f in sorted_feats]
    
    colors = [stability_color(c, n_folds) for c in cnts]
    y = np.arange(len(feats))
    
    ax.barh(y, cnts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("Gen ", "") for f in feats])
    ax.set_xlabel(f'Selection Frequency (out of {n_folds} folds)')
    ax.set_title(f'Feature Selection: {best_clf}', fontweight='bold')
    ax.axvline(n_folds * 0.6, color='black', linestyle='--', alpha=0.5, linewidth=2)  # 60% threshold
    ax.set_xlim([0, n_folds + 0.5])
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, cnt in enumerate(cnts):
        pct = (cnt / n_folds * 100) if n_folds > 0 else 0
        ax.text(cnt + 0.1, i, f'{cnt}/{n_folds} ({pct:.0f}%)', va='center', ha='left', fontsize=10)


def main():
    print("\n=== Generating RFE Figure (4 CI metrics) ===\n")
    
    # Load data
    phase4 = load_latest_phase4()
    perm = load_perm_importance(phase4)
    perm = harmonize_perm_auc_with_phase4(perm, phase4) if perm else perm
    
    # Create figure
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Feature selection stability with feature legend
    plot_left_rfe_stability(ax_l, phase4)
    
    # Right: Permutation importance or fallback
    if perm:
        plot_right_perm_importance(ax_r, perm)
    else:
        plot_right_fallback(ax_r, phase4)
    
    plt.tight_layout()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COMPLETE, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_ENHANCED, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_LEGACY, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {OUT_COMPLETE}")
    print(f"  Saved: {OUT_ENHANCED}")
    print(f"  Saved: {OUT_LEGACY}")
    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
