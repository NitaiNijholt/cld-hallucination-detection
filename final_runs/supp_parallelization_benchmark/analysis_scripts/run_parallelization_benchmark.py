#!/usr/bin/env python3
"""
Parallelization Benchmark: Wall-Clock Time Measurement

Measures wall-clock time for judging with different parallelization settings
and judge modes (correctness, citation, combined).

Usage:
    python run_supp_parallelization_benchmark.py --mode correctness
    python run_supp_parallelization_benchmark.py --mode citation
    python run_supp_parallelization_benchmark.py --mode combined

Output:
    - data/benchmark_results_MODE_YYYYMMDD_HHMMSS.xlsx
    - logs/benchmark_MODE_YYYYMMDD_HHMMSS.log
"""

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "data_science"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env.dev")

import openai

# Configuration
BENCHMARK_CONFIG = {
    "cld_name": "Depressive symptoms",
    "n_edges": 34,  # Depressive CLD edge count
    "worker_configs": [1, 3, 5, 10],
    "runs_per_config": 3,
    "judge_model": "gpt-4.1",
    "judge_temperature": 0.3,
}

# Test prompts for different judge types
CORRECTNESS_PROMPT = """Evaluate whether the following causal relationship is scientifically plausible:

Source: Stress
Target: Cortisol levels  
Relationship: POSITIVE
Motivation: Psychological stress activates the hypothalamic-pituitary-adrenal axis, leading to increased cortisol secretion.

Rate this causal claim on scientific accuracy. Respond with:
- CORRECT: The relationship is scientifically well-established
- PARTIALLY_CORRECT: Some aspects are supported but with caveats
- INCORRECT: The relationship contradicts scientific evidence

Provide your verdict followed by a brief justification (2-3 sentences)."""

CITATION_PROMPT = """You are evaluating whether a causal relationship is supported by scientific literature.

EDGE TO EVALUATE:
Source: Stress
Target: Cortisol levels  
Relationship: POSITIVE
Motivation: Psychological stress activates the hypothalamic-pituitary-adrenal axis, leading to increased cortisol secretion.

RETRIEVED CITATIONS:
[1] Smith et al. (2020). "The Stress-Cortisol Connection: A Meta-Analysis" - Journal of Endocrinology
Abstract: This meta-analysis of 45 studies confirms that acute psychological stressors reliably increase salivary cortisol levels within 20-30 minutes of stress onset. Effect sizes ranged from d=0.5 to d=1.2 depending on stressor type.

[2] Johnson & Lee (2019). "HPA Axis Activation in Response to Social Stressors" - Psychoneuroendocrinology  
Abstract: We examined cortisol responses to the Trier Social Stress Test in 200 healthy adults. Results showed significant cortisol elevations (mean increase 150% from baseline) peaking 20 minutes post-stressor.

[3] Williams (2018). "Chronic Stress and Cortisol Dysregulation" - Frontiers in Neuroscience
Abstract: While acute stress reliably elevates cortisol, chronic stress can lead to HPA axis dysregulation and blunted cortisol responses. This review discusses the mechanisms underlying stress-induced cortisol changes.

Based on these citations, evaluate whether the causal relationship is supported:
- FULLY_SUPPORTED (1.0): Strong evidence directly supports this relationship
- PARTIALLY_SUPPORTED (0.5): Some evidence but with limitations  
- NOT_SUPPORTED (0.0): No evidence or contradicting evidence

Provide your verdict, a confidence score (0-1), and cite specific references."""


def setup_logging(output_dir: Path, mode: str) -> logging.Logger:
    """Setup logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"benchmark_{mode}_{timestamp}.log"
    
    # Clear existing handlers
    logging.root.handlers = []
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def single_api_call(idx: int, prompt: str, mode: str) -> tuple:
    """Make a single API call and return timing."""
    start = time.time()
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=BENCHMARK_CONFIG["judge_model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500 if mode == "citation" else 200,
            temperature=BENCHMARK_CONFIG["judge_temperature"],
        )
        elapsed = time.time() - start
        tokens = response.usage.total_tokens if response.usage else 0
        return (idx, elapsed, "success", None, tokens)
    except Exception as e:
        elapsed = time.time() - start
        return (idx, elapsed, "error", str(e), 0)


def run_benchmark_mode(mode: str, output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Run benchmark for a specific mode (correctness, citation, or combined).
    """
    results = []
    n_edges = BENCHMARK_CONFIG["n_edges"]
    
    # Select prompt based on mode
    if mode == "correctness":
        prompt = CORRECTNESS_PROMPT
        modes_to_run = ["correctness"]
    elif mode == "citation":
        prompt = CITATION_PROMPT
        modes_to_run = ["citation"]
    elif mode == "combined":
        modes_to_run = ["correctness", "citation"]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    for current_mode in modes_to_run:
        current_prompt = CORRECTNESS_PROMPT if current_mode == "correctness" else CITATION_PROMPT
        
        for worker_count in BENCHMARK_CONFIG["worker_configs"]:
            for run_idx in range(1, BENCHMARK_CONFIG["runs_per_config"] + 1):
                logger.info(f"Benchmark: mode={current_mode}, workers={worker_count}, run={run_idx}/{BENCHMARK_CONFIG['runs_per_config']}")
                
                # Measure wall-clock time for n_edges calls with worker_count parallelism
                wall_start = time.time()
                call_times = []
                total_tokens = 0
                
                if worker_count == 1:
                    # Sequential
                    for i in range(n_edges):
                        _, elapsed, status, _, tokens = single_api_call(i, current_prompt, current_mode)
                        if status == "success":
                            call_times.append(elapsed)
                            total_tokens += tokens
                else:
                    # Parallel
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        futures = [executor.submit(single_api_call, i, current_prompt, current_mode) for i in range(n_edges)]
                        for future in as_completed(futures):
                            _, elapsed, status, _, tokens = future.result()
                            if status == "success":
                                call_times.append(elapsed)
                                total_tokens += tokens
                
                wall_end = time.time()
                wall_clock = wall_end - wall_start
                cumulative = sum(call_times)
                
                results.append({
                    "mode": current_mode,
                    "worker_count": worker_count,
                    "run_idx": run_idx,
                    "n_edges": n_edges,
                    "wall_clock_s": wall_clock,
                    "cumulative_api_s": cumulative,
                    "avg_call_time_s": cumulative / len(call_times) if call_times else 0,
                    "total_tokens": total_tokens,
                    "tokens_per_edge": total_tokens / len(call_times) if call_times else 0,
                    "successful_calls": len(call_times),
                    "timestamp": datetime.now().isoformat(),
                })
                
                logger.info(f"  Wall-clock: {wall_clock:.1f}s, Cumulative: {cumulative:.1f}s, Speedup: {cumulative/wall_clock:.2f}x")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Run parallelization benchmark')
    parser.add_argument('--mode', type=str, choices=['correctness', 'citation', 'combined'],
                       default='combined', help='Judge mode to benchmark')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    logger = setup_logging(output_dir, args.mode)
    logger.info("=" * 80)
    logger.info(f"PARALLELIZATION BENCHMARK - Mode: {args.mode.upper()}")
    logger.info("=" * 80)
    logger.info(f"Config: {json.dumps(BENCHMARK_CONFIG, indent=2)}")
    
    # Run benchmark
    logger.info(f"\nRunning {args.mode} benchmark...")
    df = run_benchmark_mode(args.mode, output_dir, logger)
    
    # Save results
    output_file = data_dir / f"benchmark_results_{args.mode}_{timestamp}.xlsx"
    df.to_excel(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    for mode in df["mode"].unique():
        mode_df = df[df["mode"] == mode]
        baseline = mode_df[mode_df["worker_count"] == 1]["wall_clock_s"].mean()
        
        logger.info(f"\n{mode.upper()} Judge:")
        for workers in BENCHMARK_CONFIG["worker_configs"]:
            worker_df = mode_df[mode_df["worker_count"] == workers]
            if len(worker_df) == 0:
                continue
            parallel_time = worker_df["wall_clock_s"].mean()
            speedup = baseline / parallel_time
            efficiency = speedup / workers * 100
            logger.info(f"  Workers={workers}: {parallel_time:.1f}s, Speedup={speedup:.2f}x, Efficiency={efficiency:.0f}%")
    
    # Save metadata
    metadata_file = output_dir / "BENCHMARK_METADATA.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Parallelization Benchmark\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Config:\n{json.dumps(BENCHMARK_CONFIG, indent=2)}\n\n")
        f.write(f"Results file: {output_file}\n")
    
    logger.info(f"\nMetadata saved to: {metadata_file}")
    logger.info("\nBenchmark complete!")
    
    return df


if __name__ == "__main__":
    main()







