#!/usr/bin/env python3
"""
Extract LLM Generator F1 Variability from Final Runs

This script extracts edge F1 scores from the RQ1a ground truth correctness
experiments which have 3 runs per CLD (seeds 10, 20, 30). It computes
mean and 95% confidence intervals for reporting LLM generator performance
with variability measures.

Usage:
    python extract_llm_f1_variability.py [--output results.json]
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import argparse
from datetime import datetime


# Final runs path
FINAL_RUNS_DIR = Path(__file__).parent.parent / "final_runs" / "RQ1a_gt_lit_correctness"

CLDS = {
    "depressive": "depressive",
    "social_norms": "social_norms", 
    "emergency_dept": "emergency_department"
}

CLD_DISPLAY_NAMES = {
    "depressive": "Depressive Symptoms",
    "social_norms": "Social Norms",
    "emergency_dept": "Emergency Department"
}


def extract_f1_from_excel(xlsx_path: Path) -> dict:
    """
    Extract TP, FP, FN from All Edges Summary sheet and compute F1.
    """
    df = pd.read_excel(xlsx_path, sheet_name='All Edges Summary')
    
    tp = df[df['Metric'] == 'True Positives (TP)']['Value'].values[0]
    fp = df[df['Metric'] == 'False Positives (FP)']['Value'].values[0]
    fn = df[df['Metric'] == 'False Negatives (FN)']['Value'].values[0]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_stats(values: list) -> dict:
    """
    Compute mean, std, and range statistics.
    
    For small n (e.g., n=3), min-max range is more appropriate than t-distribution CI
    because:
    1. t-critical value is very large (4.303 for 95% CI with df=2)
    2. Normality assumption cannot be verified with only 3 observations
    3. Min-max shows actual observed range without parametric assumptions
    """
    n = len(values)
    mean = np.mean(values)
    
    if n > 1:
        std = np.std(values, ddof=1)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Also compute t-distribution CI for comparison (but recommend min-max for small n)
        sem = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_lower_t = mean - t_crit * sem
        ci_upper_t = mean + t_crit * sem
    else:
        std = 0
        min_val = max_val = mean
        ci_lower_t = ci_upper_t = mean
    
    return {
        "n": n,
        "values": values,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        # Keep t-dist CI for reference but use min-max for reporting
        "ci_lower_t": ci_lower_t,
        "ci_upper_t": ci_upper_t,
        # Use min-max as primary range for small n
        "range_lower": min_val,
        "range_upper": max_val
    }


def extract_all_f1_scores(prompt_filter: str = "mechanistic") -> dict:
    """
    Extract F1 scores from all CLDs and runs.
    
    Args:
        prompt_filter: Which prompt type to use ('baseline', 'cot', 'mechanistic')
                      or 'all' for average across prompts
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "source": str(FINAL_RUNS_DIR),
        "prompt_filter": prompt_filter,
        "clds": {}
    }
    
    for cld_key, cld_folder in CLDS.items():
        cld_path = FINAL_RUNS_DIR / cld_folder
        
        if not cld_path.exists():
            print(f"Warning: CLD path not found: {cld_path}")
            continue
        
        all_scores = []
        run_details = []
        
        for run_num in [1, 2, 3]:
            run_path = cld_path / f"run_{run_num}"
            if not run_path.exists():
                continue
            
            # Find non-backup xlsx files
            xlsx_files = [f for f in run_path.glob("judged_*.xlsx") 
                         if not f.name.endswith(".backup.xlsx")]
            
            run_scores = []
            for xlsx_file in xlsx_files:
                # Determine prompt type from filename
                if "baseline" in xlsx_file.name:
                    prompt = "baseline"
                elif "cot" in xlsx_file.name:
                    prompt = "cot"
                elif "mechanistic" in xlsx_file.name:
                    prompt = "mechanistic"
                else:
                    continue
                
                # Apply filter
                if prompt_filter != "all" and prompt != prompt_filter:
                    continue
                
                try:
                    metrics = extract_f1_from_excel(xlsx_file)
                    run_scores.append(metrics['f1'])
                    run_details.append({
                        "run": run_num,
                        "prompt": prompt,
                        "file": xlsx_file.name,
                        **metrics
                    })
                except Exception as e:
                    print(f"Error reading {xlsx_file}: {e}")
            
            # For this run, use average if multiple prompts, otherwise single value
            if run_scores:
                if prompt_filter == "all":
                    all_scores.append(np.mean(run_scores))
                else:
                    all_scores.extend(run_scores)
        
        if all_scores:
            stats = compute_stats(all_scores)
            results["clds"][cld_key] = {
                "name": CLD_DISPLAY_NAMES[cld_key],
                "statistics": stats,
                "run_details": run_details
            }
    
    return results


def print_summary(results: dict):
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print("LLM GENERATOR F1 VARIABILITY - FINAL RUNS")
    print("=" * 80)
    print(f"Source: {results['source']}")
    print(f"Prompt: {results['prompt_filter']}")
    print()
    
    print(f"{'CLD':<25} {'Mean F1':>10} {'Min-Max Range':>20} {'t-dist CI':>20} {'N':>5}")
    print("-" * 80)
    
    for cld_key in ["depressive", "social_norms", "emergency_dept"]:
        if cld_key in results["clds"]:
            data = results["clds"][cld_key]
            s = data["statistics"]
            range_str = f"[{s['min']:.3f}, {s['max']:.3f}]"
            ci_str = f"[{s['ci_lower_t']:.3f}, {s['ci_upper_t']:.3f}]"
            print(f"{data['name']:<25} {s['mean']:>10.3f} {range_str:>20} {ci_str:>20} {s['n']:>5}")
    
    # Average
    all_means = [results["clds"][k]["statistics"]["mean"] for k in results["clds"]]
    all_mins = [results["clds"][k]["statistics"]["min"] for k in results["clds"]]
    all_maxs = [results["clds"][k]["statistics"]["max"] for k in results["clds"]]
    print("-" * 80)
    avg_range_str = f"[{np.mean(all_mins):.3f}, {np.mean(all_maxs):.3f}]"
    print(f"{'Average':<25} {np.mean(all_means):>10.3f} {avg_range_str:>20}")
    print()
    print("Note: For n=3, min-max range is preferred over t-distribution CI")
    print("      (t-crit for 95% CI with df=2 is 4.303, making CI unreliable)")
    print()


def generate_latex_table_row(results: dict) -> str:
    """Generate LaTeX table rows for thesis integration using min-max range."""
    lines = []
    for cld_key in ["depressive", "social_norms", "emergency_dept"]:
        if cld_key in results["clds"]:
            data = results["clds"][cld_key]
            s = data["statistics"]
            # Format: mean [min, max] - using min-max range for n=3
            lines.append(
                f"{data['name']}: {s['mean']:.3f} [{s['min']:.2f}, {s['max']:.2f}]"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM generator F1 variability from final runs"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--prompt", "-p", type=str, default="mechanistic",
        choices=["baseline", "cot", "mechanistic", "all"],
        help="Prompt type to analyze (default: mechanistic)"
    )
    
    args = parser.parse_args()
    
    # Extract F1 scores
    results = extract_all_f1_scores(prompt_filter=args.prompt)
    
    # Print summary
    print_summary(results)
    
    # Print LaTeX format
    print("LaTeX format:")
    print(generate_latex_table_row(results))
    
    # Save to file
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()







