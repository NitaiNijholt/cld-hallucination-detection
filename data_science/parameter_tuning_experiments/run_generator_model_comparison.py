#!/usr/bin/env python3
"""
Generator Model Comparison Experiment

Compares CLD generation quality across 3 LLMs:
- GPT-4.1 (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Sonar-Pro (Perplexity)

Evaluates on 3 ground truth CLDs with F1/Precision/Recall metrics.

Usage:
    cd /home/nitai/code/causalix.ai
    source .venv/bin/activate
    nohup python data_science/parameter_tuning_experiments/run_generator_model_comparison.py \
        > logs/generator_model_comparison_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

import os
import sys
import datetime
import uuid
import json
import logging
import argparse
from typing import Any, Dict, List, Optional
import pandas as pd

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SCIENCE_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(DATA_SCIENCE_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env.dev"), override=True)
except Exception:
    pass

# Force local Neo4j
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_HOST"] = "localhost"
# Force credentials for the local Neo4j instance used by experiments (see .neo4j_local setup).
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "devpassword"
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("ES_HOST", "localhost")
os.environ.setdefault("POSTGRES_HOST", "localhost")

from data_science.modules import run_discovery_experiment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("generator_model_comparison")

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

GENERATOR_CONFIGS = [
    {"provider": "openai", "model": "gpt-4.1", "temperature": 0.7},
    {"provider": "perplexity", "model": "sonar-pro", "temperature": 0.7},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "temperature": 0.7},
]

# Fixed judge config (not used since judge_edges=False, but required by function)
JUDGE_CONFIG = {"provider": "openai", "model": "gpt-4.1", "temperature": 0.3}

EXCEL_FILES = [
    "Social_norms_and_obesity_prevalence.xlsx",
    "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
    "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx",
]

CLD_SHORTNAMES = {
    "Social_norms_and_obesity_prevalence.xlsx": "social_norms",
    "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx": "depressive",
    "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx": "emergency_dept",
}

YAML_PATH = os.path.join(BASE_DIR, "alternative_prompts", "prompts_Nitai_C.yaml")
GROUND_TRUTH_DIR = os.path.join(DATA_SCIENCE_DIR, "ground_truth_CLDs")

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def _default_output_base_dir() -> str:
    """
    Default to the analysis data folder so we can append runs without overwriting.
    """
    return os.path.join(PROJECT_ROOT, "final_runs", "generator_model_comparison_analysis", "data")


def _model_output_dir_name(provider: str, model: str) -> str:
    """
    Canonicalize model folder names so we reuse existing analysis folders.
    """
    if provider == "openai" and model == "gpt-4.1":
        return "gpt-4.1"
    if provider == "perplexity" and model == "sonar-pro":
        return "sonar-pro"
    if provider == "anthropic" and model.startswith("claude-sonnet-4"):
        # Existing folder name in final_runs/generator_model_comparison_analysis/data/
        return "claude-sonnet-4"
    # Fallback (safe, deterministic)
    return model.replace("/", "_").replace(":", "_")


def _timestamp_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_single_experiment(
    generator_config: Dict[str, Any],
    excel_file: str,
    output_base_dir: str,
    run_tag: str,
    *,
    result_excel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single generation experiment and write results to a unique workbook."""
    
    model_name = generator_config["model"]
    provider = generator_config.get("provider", "")
    cld_name = CLD_SHORTNAMES.get(excel_file, excel_file.replace(".xlsx", ""))
    
    logger.info(f"=" * 60)
    logger.info(f"Running: {model_name} on {cld_name}")
    logger.info(f"=" * 60)
    
    excel_path = os.path.join(GROUND_TRUTH_DIR, excel_file)
    
    # Canonical model dir so we append runs into existing folders.
    model_dir = _model_output_dir_name(provider, model_name)
    run_output_dir = os.path.join(output_base_dir, model_dir, cld_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    if result_excel_path is None:
        # Use a timestamped filename so we never overwrite previous runs.
        model_prefix = model_name.replace("/", "_").replace(":", "_")
        result_excel_path = os.path.join(
            run_output_dir,
            f"{model_prefix}_{cld_name}_result_{run_tag}.xlsx",
        )
    
    start_time = datetime.datetime.now()
    
    try:
        info = run_discovery_experiment(
            excel_path=excel_path,
            export_json=True,
            plot_graph=True,
            dev_mode=False,
            load_env=False,
            corruption_rate=0.0,
            judge_edges=False,  # No judging needed - compare directly to ground truth
            yaml_path=YAML_PATH,
            num_judges=1,
            generator_config=generator_config,
            corruptor_config={"provider": "openai", "model": "gpt-4.1"},  # Not used (corruption_rate=0)
            judge_config=JUDGE_CONFIG,
            generation_parallel=True,
            generation_max_workers=3,
            plot_session_graph=True,
            plot_validation_graph=True,
            only_cited_edges=False,
            only_consistent_edges=False,
            filter_validation_nodes_to_session_nodes=True,
            experiment_description=f"gen_model_comparison_{model_dir}_{cld_name}_{run_tag}",
            output_json_prefix=f"gen_model_comparison_{model_dir}_{cld_name}_{run_tag}",
            compare_variable_overlap=False,
            result_excel_path=result_excel_path,
            overide_target_variable="",
            citation_search_provider=None,  # No citation search needed without judging
            output_dir=run_output_dir,
            node_comparison_enable=False,
            embedding_enable=False,  # Disable expensive embeddings for speed
        )
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Extract metrics from info
        result = {
            "model": model_name,
            "provider": generator_config["provider"],
            "cld": cld_name,
            "excel_file": excel_file,
            "status": "success",
            "duration_seconds": duration,
            "output_dir": run_output_dir,
            "result_excel": result_excel_path,
        }
        
        # Try to extract F1/Precision/Recall from info
        if info:
            result["session_id"] = info.get("session_id", "")
            result["n_edges_generated"] = info.get("n_edges", 0)
            result["n_nodes_generated"] = info.get("n_nodes", 0)
            
            # Extract validation metrics if available
            validation = info.get("validation_metrics", {})
            result["precision"] = validation.get("precision", None)
            result["recall"] = validation.get("recall", None)
            result["f1"] = validation.get("f1", None)
            result["tp"] = validation.get("tp", 0)
            result["fp"] = validation.get("fp", 0)
            result["fn"] = validation.get("fn", 0)
            result["tn"] = validation.get("tn", 0)
        
        logger.info(f"Completed {model_name} on {cld_name} in {duration:.1f}s")
        return result
        
    except Exception as e:
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.error(f"FAILED: {model_name} on {cld_name}: {e}")
        return {
            "model": model_name,
            "provider": generator_config["provider"],
            "cld": cld_name,
            "excel_file": excel_file,
            "status": "failed",
            "error": str(e),
            "duration_seconds": duration,
            "output_dir": run_output_dir,
            "result_excel": result_excel_path,
        }


def generate_latex_table(results_df: pd.DataFrame, output_path: str) -> None:
    """Generate a LaTeX table for the thesis."""
    
    # Check if we have F1 data
    if "f1" not in results_df.columns or results_df["f1"].isna().all():
        logger.warning("No F1 scores available - skipping LaTeX table generation")
        with open(output_path, "w") as f:
            f.write("% No F1 scores available - experiments may have failed\n")
        return
    
    # Filter to successful results only
    success_df = results_df[results_df["status"] == "success"]
    if len(success_df) == 0:
        logger.warning("No successful experiments - skipping LaTeX table generation")
        with open(output_path, "w") as f:
            f.write("% No successful experiments\n")
        return
    
    # Pivot to get models as rows, CLDs as columns
    pivot_f1 = success_df.pivot(index="model", columns="cld", values="f1")
    pivot_precision = success_df.pivot(index="model", columns="cld", values="precision")
    pivot_recall = success_df.pivot(index="model", columns="cld", values="recall")
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{CLD Generation Quality by Model (F1 Score)}
\label{tab:generator_model_comparison}
\begin{tabular}{l""" + "r" * len(pivot_f1.columns) + r"""}
\toprule
\textbf{Model} & """ + " & ".join([f"\\textbf{{{c}}}" for c in pivot_f1.columns]) + r""" \\
\midrule
"""
    
    for model in pivot_f1.index:
        row_values = []
        for cld in pivot_f1.columns:
            f1_val = pivot_f1.loc[model, cld]
            if pd.notna(f1_val):
                row_values.append(f"{f1_val:.3f}")
            else:
                row_values.append("--")
        latex += f"{model} & " + " & ".join(row_values) + r" \\" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, "w") as f:
        f.write(latex)
    
    logger.info(f"LaTeX table saved to: {output_path}")


def main() -> None:
    """Append generator-model comparison runs without overwriting existing results."""

    parser = argparse.ArgumentParser(description="Append generator model comparison runs (non-overwriting).")
    parser.add_argument("--output-base-dir", default=_default_output_base_dir(), help="Base directory to store results under model/cld subfolders.")
    parser.add_argument("--target-runs-per-model-cld", type=int, default=3, help="Ensure each model×CLD has at least this many result workbooks.")
    parser.add_argument("--max-new-runs", type=int, default=2, help="Cap the number of NEW runs to add per model×CLD (default: add 2 to reach 3 total).")
    args = parser.parse_args()

    output_base_dir = os.path.abspath(args.output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    
    logger.info(f"=" * 80)
    logger.info(f"GENERATOR MODEL COMPARISON EXPERIMENT")
    logger.info(f"Output base directory: {output_base_dir}")
    logger.info(f"Models: {[c['model'] for c in GENERATOR_CONFIGS]}")
    logger.info(f"CLDs: {list(CLD_SHORTNAMES.values())}")
    logger.info(f"Target runs per model×CLD: {args.target_runs_per_model_cld}")
    logger.info(f"Max new runs per model×CLD this invocation: {args.max_new_runs}")
    logger.info(f"=" * 80)
    
    all_results: List[Dict[str, Any]] = []
    
    # Append runs until each model×CLD reaches target count (bounded by max-new-runs).
    for generator_config in GENERATOR_CONFIGS:
        provider = generator_config["provider"]
        model = generator_config["model"]
        model_dir = _model_output_dir_name(provider, model)
        for excel_file in EXCEL_FILES:
            cld_name = CLD_SHORTNAMES.get(excel_file, excel_file.replace(".xlsx", ""))
            cld_dir = os.path.join(output_base_dir, model_dir, cld_name)
            os.makedirs(cld_dir, exist_ok=True)

            existing = sorted([p for p in os.listdir(cld_dir) if p.endswith(".xlsx") and "_result_" in p])
            existing_count = len(existing)
            needed_total = max(args.target_runs_per_model_cld - existing_count, 0)
            to_add = min(needed_total, args.max_new_runs)

            if to_add <= 0:
                logger.info(f"Skip (already have {existing_count}): {model_dir}/{cld_name}")
                continue

            logger.info(f"Need {to_add} more (have {existing_count}, target {args.target_runs_per_model_cld}): {model_dir}/{cld_name}")
            for _ in range(to_add):
                run_tag = _timestamp_tag()
                result = run_single_experiment(generator_config, excel_file, output_base_dir, run_tag)
                all_results.append(result)
    
    # Save a summary of *new* runs in a timestamped file (does not overwrite).
    if all_results:
        results_df = pd.DataFrame(all_results)
        stamp = _timestamp_tag()
        summary_path = os.path.join(PROJECT_ROOT, "final_runs", f"generator_model_comparison_append_{stamp}.xlsx")
        results_df.to_excel(summary_path, index=False)
        logger.info(f"New-run summary saved to: {summary_path}")
        json_path = os.path.join(PROJECT_ROOT, "final_runs", f"generator_model_comparison_append_{stamp}.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"New-run JSON saved to: {json_path}")
    else:
        logger.info("No new runs executed (everything already at target count).")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    
    successful = results_df[results_df["status"] == "success"]
    failed = results_df[results_df["status"] == "failed"]
    
    logger.info(f"Successful: {len(successful)}/{len(all_results)}")
    logger.info(f"Failed: {len(failed)}/{len(all_results)}")
    
    if len(successful) > 0:
        logger.info("\nF1 Scores by Model and CLD:")
        for _, row in successful.iterrows():
            f1_str = f"{row['f1']:.3f}" if pd.notna(row.get('f1')) else "N/A"
            logger.info(f"  {row['model']:40s} | {row['cld']:15s} | F1={f1_str}")
    
    logger.info(f"\nOutput base directory: {output_base_dir}")


if __name__ == "__main__":
    main()






