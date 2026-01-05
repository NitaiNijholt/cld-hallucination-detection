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
from typing import Any, Dict, List
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
    # {"provider": "openai", "model": "gpt-4.1", "temperature": 0.7},  # Already have these results
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "temperature": 0.7},  # Claude Sonnet 4
    # {"provider": "perplexity", "model": "sonar-pro", "temperature": 0.7},  # Already completed
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

def run_single_experiment(
    generator_config: Dict[str, Any],
    excel_file: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Run a single generation + judging experiment."""
    
    model_name = generator_config["model"]
    cld_name = CLD_SHORTNAMES.get(excel_file, excel_file.replace(".xlsx", ""))
    
    logger.info(f"=" * 60)
    logger.info(f"Running: {model_name} on {cld_name}")
    logger.info(f"=" * 60)
    
    excel_path = os.path.join(GROUND_TRUTH_DIR, excel_file)
    
    # Sanitize model name for file paths
    model_dir = model_name.replace("/", "_").replace(":", "_")
    run_output_dir = os.path.join(output_dir, model_dir, cld_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    result_excel_path = os.path.join(
        run_output_dir,
        f"{model_dir}_{cld_name}_result.xlsx"
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
            experiment_description=f"gen_model_comparison_{model_dir}_{cld_name}",
            output_json_prefix=f"gen_model_comparison_{model_dir}_{cld_name}",
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
    """Run the full generator model comparison experiment."""
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "final_runs", f"generator_model_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"=" * 80)
    logger.info(f"GENERATOR MODEL COMPARISON EXPERIMENT")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models: {[c['model'] for c in GENERATOR_CONFIGS]}")
    logger.info(f"CLDs: {list(CLD_SHORTNAMES.values())}")
    logger.info(f"Total experiments: {len(GENERATOR_CONFIGS) * len(EXCEL_FILES)}")
    logger.info(f"=" * 80)
    
    all_results: List[Dict[str, Any]] = []
    
    # Run all experiments
    for generator_config in GENERATOR_CONFIGS:
        for excel_file in EXCEL_FILES:
            result = run_single_experiment(generator_config, excel_file, output_dir)
            all_results.append(result)
            
            # Save intermediate results
            results_df = pd.DataFrame(all_results)
            results_df.to_excel(
                os.path.join(output_dir, "comparison_summary_partial.xlsx"),
                index=False
            )
    
    # Generate final summary
    results_df = pd.DataFrame(all_results)
    
    # Save results
    summary_path = os.path.join(output_dir, "comparison_summary.xlsx")
    results_df.to_excel(summary_path, index=False)
    logger.info(f"Results saved to: {summary_path}")
    
    # Save JSON for programmatic access
    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"JSON results saved to: {json_path}")
    
    # Generate LaTeX table
    latex_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)
    generate_latex_table(results_df, os.path.join(latex_dir, "table_model_comparison.tex"))
    
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
    
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"Summary Excel: {summary_path}")


if __name__ == "__main__":
    main()
