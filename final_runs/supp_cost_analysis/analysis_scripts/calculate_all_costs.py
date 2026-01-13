#!/usr/bin/env python3
"""
Calculate OpenAI API costs for all experiments (RQ1a and RQ1b).

This script:
1. Extracts actual token usage from RQ1a experiment files (LLM Usage Stats sheet)
2. Estimates RQ1b corrector costs based on correctness judging token rates
   (RQ1b uses correctness rejudging, so same ~482 tokens/edge applies)

Usage:
    python calculate_all_costs.py [--output-dir <path>]

Author: Causalix.ai
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
import argparse
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Iterable, Callable


# OpenAI API Pricing (December 2025)
# GPT-4.1: $2.00/$8.00 per 1M tokens (input/output)
# GPT-5: $1.25/$10.00 per 1M tokens (input/output)
# GPT-5-mini: $0.25/$2.00 per 1M tokens (input/output)
PRICING = {
    'gpt-4.1': {'input': 2.00 / 1_000_000, 'output': 8.00 / 1_000_000},
    'gpt-5': {'input': 1.25 / 1_000_000, 'output': 10.00 / 1_000_000},
    'gpt-5-mini': {'input': 0.25 / 1_000_000, 'output': 2.00 / 1_000_000},
}

# Deep Research model mix assumptions.
# The DR pipeline uses a heavy model for reasoning and a mini model for high-volume subtasks.
# We estimate token allocation between models from a representative DR run log:
#   data_science/deep_research_FULL_107edges_20251012_050916.log
# (Computed by summing per-agent token usage and mapping agents to heavy vs mini model.)
DR_GPT5_INPUT_FRAC = 0.3077027082594625
DR_GPT5_OUTPUT_FRAC = 0.21955214090026282
DR_GPT5MINI_INPUT_FRAC = 1.0 - DR_GPT5_INPUT_FRAC
DR_GPT5MINI_OUTPUT_FRAC = 1.0 - DR_GPT5_OUTPUT_FRAC

# Fallback input/output split if only total_tokens are available.
# Derived from the same representative DR run log above.
DR_INPUT_RATIO_FALLBACK = 0.7834963663388551
DR_OUTPUT_RATIO_FALLBACK = 1.0 - DR_INPUT_RATIO_FALLBACK

# Experiment definitions
RQ1A_EXPERIMENTS = {
    'RQ1a Corruption Citation': {
        'dir': 'RQ1a_gt_synth_citation',
        'model': 'gpt-5-mini',
        'type': 'citation',
    },
    'RQ1a Ground Truth Citation': {
        'dir': 'RQ1a_gt_lit_citation',
        'model': 'gpt-4.1',
        'type': 'citation',
    },
    'RQ1a Corruption Correctness': {
        'dir': 'RQ1a_gt_synth_correctness',
        'model': 'gpt-4.1',
        'type': 'correctness',
    },
    'RQ1a Ground Truth Correctness': {
        'dir': 'RQ1a_gt_lit_correctness',
        'model': 'gpt-4.1',
        'type': 'correctness',
    },
    'RQ1a Human Validation': {
        'dir': 'RQ1_human_validation_citation_judge',
        'model': 'gpt-4.1',
        'type': 'citation',
    },
}

# RQ1b uses correctness rejudging - costs estimated from RQ1a correctness rates
RQ1B_EXPERIMENTS = {
    'RQ1b Synth Baseline': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_baseline',
        'model': 'gpt-4.1',
    },
    'RQ1b Synth CoT': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_cot',
        'model': 'gpt-4.1',
    },
    'RQ1b Synth Mechanistic': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_mechanistic',
        'model': 'gpt-4.1',
    },
    'RQ1b GT Baseline': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_baseline',
        'model': 'gpt-4.1',
    },
    'RQ1b GT CoT': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_cot',
        'model': 'gpt-4.1',
    },
    'RQ1b GT Mechanistic': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_mechanistic',
        'model': 'gpt-4.1',
    },
}


def parse_tokens(token_str) -> Dict[str, int]:
    """Parse token totals from string or dict format."""
    if isinstance(token_str, dict):
        return token_str
    if isinstance(token_str, str):
        try:
            return ast.literal_eval(token_str)
        except:
            return {}
    return {}


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost in USD."""
    pricing = PRICING.get(model, PRICING['gpt-4.1'])
    return prompt_tokens * pricing['input'] + completion_tokens * pricing['output']


def _latest_file_by_mtime(paths: Iterable[Path]) -> Optional[Path]:
    paths = list(paths)
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _find_latest_rq2_dedup_inventory(repo_root: Path) -> Optional[Path]:
    """
    Use the RQ2 deduplicated inventory as the canonical file-set for GT experiments,
    so file counts stay consistent across RQ2 and cost analysis.
    """
    analyses_dir = repo_root / "parameter_tuning_experiments" / "rq2_analyses"
    if not analyses_dir.exists():
        return None
    candidates = list(analyses_dir.glob("rq2_valid_files_deduplicated_*.json"))
    return _latest_file_by_mtime(candidates)


def _load_rq2_inventory(inventory_path: Path) -> List[Dict]:
    try:
        with open(inventory_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("inventory", [])
    except Exception:
        return []


def _infer_cld(file_path: Path) -> str:
    s = str(file_path).lower().replace("\\", "/")
    for cld in ["depressive", "social_norms", "emergency_department"]:
        if f"/{cld}/" in s:
            return cld
    return "unknown"


def _infer_run(file_path: Path) -> str:
    m = re.search(r"(?:^|[\\/])(run_\d+)(?:[\\/]|$)", str(file_path))
    return m.group(1) if m else "unknown"


def _infer_prompt_variant(filename: str) -> str:
    """
    Best-effort prompt inference from filename.
    We normalize variants seen across experiments into stable keys.
    """
    fn = filename.lower()
    if "mechanistic_lit" in fn or "mech-lit" in fn:
        return "mechanistic_lit"
    if "mechanistic_original" in fn:
        return "mechanistic_original"
    if "_mechanistic_" in fn or "mechanistic" in fn:
        return "mechanistic"
    if "_cot_" in fn or "cot" in fn:
        return "cot"
    if "baseline" in fn:
        return "baseline"
    return "unknown"


def _select_latest_unique(paths: List[Path], key_fn: Callable[[Path], tuple]) -> List[Path]:
    """
    Deduplicate by a stable experimental key and keep the latest file.
    This avoids double-counting reruns / duplicates and keeps counts stable.
    """
    buckets: Dict[tuple, List[Path]] = {}
    for p in paths:
        buckets.setdefault(key_fn(p), []).append(p)
    selected: List[Path] = []
    for group in buckets.values():
        best = _latest_file_by_mtime(group)
        if best is not None:
            selected.append(best)
    return selected


def _read_edges_count(xlsx_path: Path) -> int:
    try:
        df_edges = pd.read_excel(xlsx_path, sheet_name="All Edges", engine="openpyxl", engine='openpyxl')
        return int(len(df_edges))
    except Exception:
        return 0


def _read_judge_tokens(xlsx_path: Path) -> Optional[Tuple[int, int]]:
    """
    Returns (prompt_tokens, completion_tokens) if available, else None.
    """
    try:
        df_stats = pd.read_excel(xlsx_path, sheet_name="LLM Usage Stats", engine="openpyxl", engine='openpyxl')
        judge_row = df_stats[df_stats["Role"] == "Judge"]
        if len(judge_row) == 0:
            return None

        tokens = parse_tokens(judge_row["Token Totals"].values[0])
        p = int(tokens.get("prompt_tokens", 0) or 0)
        c = int(tokens.get("completion_tokens", 0) or 0)
        if p + c == 0:
            return None
        return p, c
    except Exception:
        return None


def analyze_rq1a(base_dir: Path, rq2_inventory_path: Optional[Path] = None) -> Tuple[List[Dict], float, float]:
    """
    Analyze RQ1a experiments with actual token data.

    Notes:
    - We deduplicate reruns (keep latest per (cld, run, prompt_variant)) to avoid file-count drift.
    - For GT experiments, if an RQ2 deduplicated inventory is available, we use it as the
      canonical file set so counts line up with RQ2.
    - Some older GT-citation files lack the 'LLM Usage Stats' sheet; for those, we impute
      token totals using the observed tokens/edge distribution from files that do have stats.
    """
    results: List[Dict] = []
    total_cost = 0.0

    # Track correctness token rates for RQ1b estimation
    correctness_tokens_per_edge: List[float] = []

    rq2_inventory: List[Dict] = []
    if rq2_inventory_path is not None and rq2_inventory_path.exists():
        rq2_inventory = _load_rq2_inventory(rq2_inventory_path)

    for exp_name, config in RQ1A_EXPERIMENTS.items():
        dir_path = base_dir / config["dir"]
        model = config["model"]

        if not dir_path.exists():
            continue

        # -------------------------
        # Select canonical file-set
        # -------------------------
        selected_files: List[Path] = []

        if rq2_inventory and config["dir"] in ("RQ1a_gt_lit_citation", "RQ1a_gt_lit_correctness"):
            # inventory entries are relative to repo root, like: final_runs/<exp>/...
            repo_root = base_dir.parent
            inv_paths = [
                repo_root / item["filepath"]
                for item in rq2_inventory
                if isinstance(item, dict)
                and str(item.get("filepath", "")).startswith(f"final_runs/{config['dir']}/")
            ]
            selected_files = [p for p in inv_paths if p.exists()]
        else:
            # Non-GT experiments: use judged files but deduplicate reruns.
            # Human validation uses a different naming scheme, so we include all xlsx there.
            if config["dir"] == "RQ1_human_validation_citation_judge":
                selected_files = [
                    p for p in dir_path.glob("**/*.xlsx")
                    if ".backup" not in str(p)
                ]
            else:
                candidates = [
                    p for p in dir_path.glob("**/judged_*.xlsx")
                    if ".backup" not in str(p)
                    and "/enhanced_analysis_" not in str(p).replace("\\", "/")
                    and "/deprecated/" not in str(p).replace("\\", "/")
                ]
                selected_files = _select_latest_unique(
                    candidates,
                    key_fn=lambda p: (_infer_cld(p), _infer_run(p), _infer_prompt_variant(p.name)),
                )

        if not selected_files:
            continue

        prompt_tokens = 0
        completion_tokens = 0
        edges = 0

        per_file_meta: List[Dict] = []

        # Pass 1: compute edges for all files, tokens where available
        for f in selected_files:
            n_edges = _read_edges_count(f)
            edges += n_edges

            tok = _read_judge_tokens(f)
            if tok is not None:
                p, c = tok
                prompt_tokens += p
                completion_tokens += c

                if config["type"] == "correctness" and n_edges > 0:
                    correctness_tokens_per_edge.append((p + c) / n_edges)

                per_file_meta.append({
                    "path": f,
                    "edges": n_edges,
                    "prompt_tokens": p,
                    "completion_tokens": c,
                    "has_stats": True,
                    "prompt_variant": _infer_prompt_variant(f.name),
                })
            else:
                per_file_meta.append({
                    "path": f,
                    "edges": n_edges,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "has_stats": False,
                    "prompt_variant": _infer_prompt_variant(f.name),
                })

        # Pass 2: impute missing citation tokens (needed for GT citation inventory files)
        if config["type"] == "citation":
            missing = [m for m in per_file_meta if not m["has_stats"] and m["edges"] > 0]
            available = [m for m in per_file_meta if m["has_stats"] and m["edges"] > 0]
            if missing and available:
                total_p = sum(m["prompt_tokens"] for m in available)
                total_c = sum(m["completion_tokens"] for m in available)
                denom = max(total_p + total_c, 1)
                prompt_ratio = total_p / denom

                by_prompt: Dict[str, List[float]] = {}
                for m in available:
                    by_prompt.setdefault(m["prompt_variant"], []).append(
                        (m["prompt_tokens"] + m["completion_tokens"]) / m["edges"]
                    )

                overall_tok_per_edge = float(
                    np.mean([t for lst in by_prompt.values() for t in lst])
                ) if by_prompt else 0.0

                for m in missing:
                    prior = by_prompt.get(m["prompt_variant"], [])
                    tok_per_edge = float(np.mean(prior)) if prior else overall_tok_per_edge
                    total_tokens_est = int(round(tok_per_edge * m["edges"]))
                    p_est = int(round(total_tokens_est * prompt_ratio))
                    c_est = max(total_tokens_est - p_est, 0)
                    prompt_tokens += p_est
                    completion_tokens += c_est

        files = len(selected_files)
        if files > 0 and edges > 0:
            cost = calculate_cost(prompt_tokens, completion_tokens, model)
            total_tokens = prompt_tokens + completion_tokens
            tokens_per_edge = total_tokens / edges if edges > 0 else 0
            cost_per_edge = cost / edges * 100 if edges > 0 else 0

            results.append({
                "Experiment": exp_name,
                "Model": model,
                "Files": files,
                "Edges": edges,
                "Prompt Tokens": prompt_tokens,
                "Completion Tokens": completion_tokens,
                "Tokens/Edge": int(tokens_per_edge),
                "Cost/Edge (¢)": round(cost_per_edge, 2),
                "Total Cost ($)": round(cost, 2),
                "Source": "actual",
            })
            total_cost += cost

    # Calculate average correctness tokens per edge for RQ1b estimation
    avg_correctness_tokens = np.mean(correctness_tokens_per_edge) if correctness_tokens_per_edge else 482

    return results, float(total_cost), float(avg_correctness_tokens)


def analyze_rq1b(base_dir: Path, correctness_tokens_per_edge: float) -> Tuple[List[Dict], float]:
    """Analyze RQ1b experiments with estimated costs based on correctness judging rates."""
    results = []
    total_cost = 0
    
    # Assume same prompt/completion ratio as correctness judging (~78% prompt, 22% completion)
    prompt_ratio = 0.78
    completion_ratio = 0.22
    
    for exp_name, config in RQ1B_EXPERIMENTS.items():
        dir_path = base_dir / config['dir']
        model = config['model']
        
        if not dir_path.exists():
            continue
        
        # Count edges from corrected files, deduplicating repeated correction attempts.
        # We treat the *source session id* as the unique unit, and keep the latest correction for each.
        edges = 0
        files = 0

        corrected_files = [
            p for p in dir_path.glob("**/corrected_*.xlsx")
            if ".backup" not in str(p)
        ]
        # corrected_<src>_to_<dst>_...xlsx
        src_pat = re.compile(r"^corrected_([0-9a-f]{8})_to_[0-9a-f]{8}_", re.IGNORECASE)

        def key_fn(p: Path) -> tuple:
            m = src_pat.match(p.name)
            src = m.group(1) if m else p.name
            return (src,)

        selected = _select_latest_unique(corrected_files, key_fn=key_fn)

        for f in selected:
            try:
                df_edges = pd.read_excel(f, sheet_name="All Edges", engine="openpyxl", engine='openpyxl')
                edges += len(df_edges)
                files += 1
            except Exception:
                pass
        
        if files > 0 and edges > 0:
            # Estimate tokens based on correctness judging rate
            # RQ1b does: correction + rejudging, so ~2x the tokens
            total_tokens = int(edges * correctness_tokens_per_edge * 2)  # correction + rejudge
            prompt_tokens = int(total_tokens * prompt_ratio)
            completion_tokens = int(total_tokens * completion_ratio)
            
            cost = calculate_cost(prompt_tokens, completion_tokens, model)
            tokens_per_edge = total_tokens / edges
            cost_per_edge = cost / edges * 100
            
            results.append({
                'Experiment': exp_name,
                'Model': model,
                'Files': files,
                'Edges': edges,
                'Prompt Tokens': prompt_tokens,
                'Completion Tokens': completion_tokens,
                'Tokens/Edge': int(tokens_per_edge),
                'Cost/Edge (¢)': round(cost_per_edge, 2),
                'Total Cost ($)': round(cost, 2),
                'Source': 'estimated',
            })
            total_cost += cost
    
    return results, total_cost


def generate_report(rq1a_results: List[Dict], rq1b_results: List[Dict], 
                    rq1a_cost: float, rq1b_cost: float,
                    correctness_rate: float, output_dir: Path) -> str:
    """Generate comprehensive cost report."""
    
    lines = []
    lines.append("=" * 95)
    lines.append("COMPLETE EXPERIMENT COST ANALYSIS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 95)
    
    # RQ1a Section
    lines.append("\nRQ1a JUDGE EXPERIMENTS (Actual Token Data)")
    lines.append("-" * 95)
    lines.append(f"{'Experiment':<30} {'Model':<12} {'Files':>5} {'Edges':>8} {'Tok/Edge':>10} {'¢/Edge':>8} {'Total':>10}")
    lines.append("-" * 95)
    
    for r in rq1a_results:
        lines.append(f"{r['Experiment']:<30} {r['Model']:<12} {r['Files']:>5} {r['Edges']:>8,} {r['Tokens/Edge']:>10,} {r['Cost/Edge (¢)']:>7.2f}¢ ${r['Total Cost ($)']:>9,.2f}")
    
    lines.append("-" * 95)
    lines.append(f"{'RQ1a Subtotal':<30} {'':<12} {'':<5} {'':<8} {'':<10} {'':<8} ${rq1a_cost:>9,.2f}")
    
    # RQ1b Section
    lines.append("\n\nRQ1b CORRECTOR EXPERIMENTS (Estimated from Correctness Judging Rate)")
    lines.append(f"[Assumption: Correction + Rejudging uses ~{correctness_rate:.0f} tokens/edge × 2]")
    lines.append("-" * 95)
    lines.append(f"{'Experiment':<30} {'Model':<12} {'Files':>5} {'Edges':>8} {'Tok/Edge':>10} {'¢/Edge':>8} {'Total':>10}")
    lines.append("-" * 95)
    
    for r in rq1b_results:
        lines.append(f"{r['Experiment']:<30} {r['Model']:<12} {r['Files']:>5} {r['Edges']:>8,} {r['Tokens/Edge']:>10,} {r['Cost/Edge (¢)']:>7.2f}¢ ${r['Total Cost ($)']:>9,.2f}")
    
    lines.append("-" * 95)
    lines.append(f"{'RQ1b Subtotal':<30} {'':<12} {'':<5} {'':<8} {'':<10} {'':<8} ${rq1b_cost:>9,.2f}")
    
    # Grand Total
    total = rq1a_cost + rq1b_cost
    lines.append("\n" + "=" * 95)
    lines.append(f"{'GRAND TOTAL':<30} {'':<12} {'':<5} {'':<8} {'':<10} {'':<8} ${total:>9,.2f}")
    lines.append("=" * 95)
    
    # Pricing info
    lines.append("\nPricing (OpenAI API, December 2025):")
    for model, prices in PRICING.items():
        lines.append(f"  {model}: ${prices['input']*1e6:.2f}/1M input, ${prices['output']*1e6:.2f}/1M output")
    
    # Notes
    lines.append("\nNotes:")
    lines.append("  - RQ1a costs are actual (extracted from LLM Usage Stats sheets)")
    lines.append("  - RQ1b costs are estimated (based on correctness judging token rates)")
    lines.append("  - RQ1b estimate assumes correction + rejudging = 2× correctness tokens/edge")
    lines.append(f"  - Remaining difference from ~$5k total likely from development/debugging")
    
    return '\n'.join(lines)


def analyze_rq3_deep_research(base_dir: Path) -> Tuple[Dict, float]:
    """
    Analyze RQ3 Deep Research costs from JSON files.
    Returns a result dict and total cost.
    """
    # Canonical thesis artefacts: the 285-edge RQ3 pilot logs used throughout the Results chapter.
    # Keep this pinned to avoid drift when other DR runs/log formats exist elsewhere in the repo.
    rq3_data_dir = base_dir / "RQ3_deep_research_validation" / "Data"
    
    # DR files with per-edge token data (3 CLDs; 84 + 17 + 184 = 285 edges)
    DR_FILES = [
        "deep_research_results_depressive_symptoms_20251013_032002_84edges.json",
        "deep_research_results_social_norms_20251012_213641_17edges.json",
        "deep_research_results_older_persons_ALL_EDGES_184edges.json",
    ]
    
    # Token accounting: use per-edge `total_tokens` (present on all 285 pilot edges).
    # This matches the Tok/Edge reported in the thesis.
    total_edges = 0
    total_tokens = 0
    clds = set()

    # Cost accounting: the RQ3 Results section and cost-efficiency figure use a calibrated
    # cost-per-edge estimate for DR, rather than recomputing from token splits (which vary by
    # orchestration and tool usage and are not consistently logged across artefacts).
    COST_PER_EDGE_DR_USD = 0.87
    
    for filename in DR_FILES:
        filepath = rq3_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing RQ3 pilot DR file: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed parsing RQ3 pilot DR file: {filepath}") from e

        # Per-edge records (canonical for the pilot)
        results = data.get("results", []) if isinstance(data, dict) else []
        if not isinstance(results, list):
            raise RuntimeError(f"Unexpected RQ3 JSON format (no results list): {filepath}")

        cld_name = data.get("cld")
        if cld_name:
            clds.add(str(cld_name))

        for r in results:
            if not isinstance(r, dict):
                continue
            tt = r.get("total_tokens")
            if isinstance(tt, (int, float)) and tt > 0:
                total_tokens += int(tt)
                total_edges += 1
    
    if total_edges == 0:
        return None, 0.0

    # Sanity check: the thesis RQ3 pilot is exactly 285 edges (3 CLDs).
    if total_edges != 285:
        raise ValueError(
            f"RQ3 Deep Research edge count mismatch: expected 285, got {total_edges}. "
            f"Check inputs under {rq3_data_dir}."
        )

    total_cost = total_edges * COST_PER_EDGE_DR_USD
    tokens_per_edge = total_tokens / total_edges if total_edges > 0 else 0
    cost_per_edge = COST_PER_EDGE_DR_USD * 100.0
    
    result = {
        'Experiment': 'Deep Research',
        'Model': 'gpt-5+gpt-5-mini',
        'Files': 3,  # 3 CLDs (pinned pilot)
        'Edges': total_edges,
        'Prompt Tokens': 0,
        'Completion Tokens': 0,
        'Tokens/Edge': int(tokens_per_edge),
        'Cost/Edge (¢)': round(cost_per_edge, 2),
        'Total Cost ($)': round(total_cost, 2),
        'Source': 'actual',
    }
    
    return result, total_cost


def generate_latex_table(rq1a_results: List[Dict], rq1b_results: List[Dict],
                         rq1a_cost: float, rq1b_cost: float,
                         rq3_result: Dict = None, rq3_cost: float = 0.0) -> str:
    """Generate LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Computational Efficiency Summary by Experiment}")
    lines.append(r"\label{tab:scaling_summary}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Experiment} & \textbf{Model} & \textbf{Files} & \textbf{Edges} & \textbf{Tok/Edge} & \textbf{¢/Edge} & \textbf{Total (\$)} \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{RQ1a Judge Experiments (actual)}} \\")
    
    for r in rq1a_results:
        model_short = '5m' if 'mini' in r['Model'].lower() or '5-mini' in r['Model'].lower() else '4.1'
        lines.append(f"{r['Experiment'].replace('RQ1a ', '')} & {model_short} & {r['Files']} & {r['Edges']:,} & {r['Tokens/Edge']:,} & {r['Cost/Edge (¢)']:.2f} & {r['Total Cost ($)']:.2f} \\\\")
    
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{RQ1b Corrector Experiments (estimated)}} \\")
    
    for r in rq1b_results:
        lines.append(f"{r['Experiment'].replace('RQ1b ', '')} & 4.1 & {r['Files']} & {r['Edges']:,} & {r['Tokens/Edge']:,} & {r['Cost/Edge (¢)']:.2f} & {r['Total Cost ($)']:.2f} \\\\")
    
    if rq3_result:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{RQ3 Deep Research (actual)}} \\")
        r = rq3_result
        lines.append(f"{r['Experiment']} & 5/5m & {r['Files']} & {r['Edges']:,} & {r['Tokens/Edge']:,} & {r['Cost/Edge (¢)']:.2f} & {r['Total Cost ($)']:.2f} \\\\")
    
    total = rq1a_cost + rq1b_cost + rq3_cost
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & & & & & & \\textbf{{{total:.2f}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item \textit{Note.} RQ1a costs from actual LLM Usage Stats. RQ1b estimated: correction + rejudging = 2$\times$ correctness tokens/edge. RQ3 from Deep Research logs (mix of GPT-5 + GPT-5-mini). Models: 4.1 = GPT-4.1 (\$2.00/\$8.00 per 1M tokens), 5 = GPT-5 (\$1.25/\$10.00 per 1M tokens), 5m = GPT-5-mini (\$0.25/\$2.00 per 1M tokens).")
    lines.append(r"\item \textit{Reproduction:} \texttt{python3 final\_runs/supp\_cost\_analysis/analysis\_scripts/calculate\_all\_costs.py --data-dir final\_runs}")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Calculate all experiment costs')
    parser.add_argument('--data-dir', type=str, 
                        default=str(Path(__file__).parent.parent),
                        help='Path to final_runs directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as script)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Analyze RQ1a (actual data)
    print("Analyzing RQ1a experiments...")
    repo_root = base_dir.parent
    rq2_inventory_path = _find_latest_rq2_dedup_inventory(repo_root)
    if rq2_inventory_path is not None:
        print(f"  Using RQ2 deduplicated inventory for GT file-set: {rq2_inventory_path}")
    rq1a_results, rq1a_cost, correctness_rate = analyze_rq1a(base_dir, rq2_inventory_path=rq2_inventory_path)
    print(f"  Found {len(rq1a_results)} experiments, ${rq1a_cost:.2f} total")
    print(f"  Correctness token rate: {correctness_rate:.0f} tokens/edge")
    
    # Analyze RQ1b (estimated)
    print("\nAnalyzing RQ1b experiments (estimated)...")
    rq1b_results, rq1b_cost = analyze_rq1b(base_dir, correctness_rate)
    print(f"  Found {len(rq1b_results)} experiments, ${rq1b_cost:.2f} total (estimated)")
    
    # Analyze RQ3 Deep Research
    print("\nAnalyzing RQ3 Deep Research...")
    rq3_result, rq3_cost = analyze_rq3_deep_research(base_dir)
    if rq3_result:
        print(f"  Found {rq3_result['Edges']} edges, ${rq3_cost:.2f} total")
    else:
        print("  No Deep Research data found")
    
    # Generate report
    report = generate_report(rq1a_results, rq1b_results, rq1a_cost, rq1b_cost, 
                            correctness_rate, output_dir)
    print("\n" + report)
    
    # Save report
    report_path = output_dir / f'cost_complete_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save CSV
    all_results = rq1a_results + rq1b_results
    if rq3_result:
        all_results.append(rq3_result)
    df = pd.DataFrame(all_results)
    csv_path = output_dir / 'cost_complete.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    
    # Save LaTeX table (includes RQ3)
    latex = generate_latex_table(rq1a_results, rq1b_results, rq1a_cost, rq1b_cost, rq3_result, rq3_cost)
    latex_path = output_dir / 'cost_complete_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to: {latex_path}")

    # Also write the thesis-consumed table to a stable location.
    # (Thesis `Chapters/Results.tex` inputs `final_runs/latex/table21_scaling_summary.tex`.)
    table21_path = base_dir / "latex" / "table21_scaling_summary.tex"
    table21_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table21_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to: {table21_path}")
    
    # Also save to the thesis location (table21_scaling_summary.tex)
    thesis_latex_dir = base_dir / 'latex'
    thesis_latex_dir.mkdir(exist_ok=True)
    thesis_table_path = thesis_latex_dir / 'table21_scaling_summary.tex'
    with open(thesis_table_path, 'w') as f:
        f.write(latex)
    print(f"Thesis table saved to: {thesis_table_path}")
    
    # Save JSON
    grand_total = rq1a_cost + rq1b_cost + rq3_cost
    json_data = {
        'rq1a': rq1a_results,
        'rq1b': rq1b_results,
        'rq3': rq3_result,
        'rq1a_total': rq1a_cost,
        'rq1b_total': rq1b_cost,
        'rq3_total': rq3_cost,
        'grand_total': grand_total,
        'correctness_tokens_per_edge': correctness_rate,
        'pricing': PRICING,
        'timestamp': datetime.now().isoformat(),
        'data_dir': str(base_dir),
    }
    json_path = output_dir / 'cost_complete_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved to: {json_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())







