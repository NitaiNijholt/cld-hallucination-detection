#!/usr/bin/env python3
"""Parity test: compare vLLM serving API output against evaluation predictions.

Loads a subset of test prompts, sends them to a running vLLM server, and
compares the verdicts against the stored evaluation predictions XLSX.

Usage:
    # First, start the vLLM server (locally or on cloud)
    # Then run this test pointing at the server:

    python -m finetuning.src.serving.parity_test \
        --server_url http://localhost:8000 \
        --model_name cld_judge \
        --predictions_xlsx finetuning/results/canonical_shared/62a7584-shared-edge-v1/qwen3_14b_shared_lit_verdict_only/predictions_*.xlsx \
        --test_data finetuning/data/canonical_shared/62a7584-shared-edge-v1/verdict_only/shared_test_lit.xlsx \
        --n_samples 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a causal diagram expert evaluating the quality of a causal explanation."
)

USER_TEMPLATE_VERDICT_ONLY = (
    "SOURCE: {source}\n"
    "TARGET: {target}\n"
    "RELATIONSHIP: {relationship}\n"
    "EXPLANATION: {motivation}\n\n"
    "Assess whether the causal reasoning is logically sound based on:\n"
    "- Plausibility: is there a clear causal mechanism?\n"
    "- Temporality: does the cause precede the effect?\n"
    "- Strength: is the relationship substantial?\n"
    "- Coherence: is the reasoning internally consistent?\n\n"
    "Respond with VERDICT only."
)


def parse_verdict(text: str) -> str | None:
    """Extract CORRECT/INCORRECT/PARTIALLY_CORRECT from model output."""
    text_upper = text.strip().upper()
    for verdict in ("PARTIALLY_CORRECT", "INCORRECT", "CORRECT"):
        if verdict in text_upper:
            return verdict
    return None


def build_prompt(row: pd.Series) -> str:
    return USER_TEMPLATE_VERDICT_ONLY.format(
        source=row.get("source", row.get("Source", "")),
        target=row.get("target", row.get("Target", "")),
        relationship=row.get("relationship", row.get("Relationship", "")),
        motivation=row.get("motivation", row.get("Motivation", row.get("explanation", ""))),
    )


def query_server(
    server_url: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def run_parity_test(
    server_url: str,
    model_name: str,
    test_data_path: str,
    predictions_path: str | None,
    n_samples: int,
) -> dict:
    test_df = pd.read_excel(test_data_path)
    sample = test_df.head(n_samples)

    ref_verdicts: dict[int, str] = {}
    if predictions_path:
        pred_path = Path(predictions_path)
        if pred_path.exists():
            pred_df = pd.read_excel(pred_path)
            verdict_col = None
            for col in pred_df.columns:
                if "prediction" in col.lower() or "verdict" in col.lower():
                    verdict_col = col
                    break
            if verdict_col:
                for idx, val in zip(pred_df.index[:n_samples], pred_df[verdict_col][:n_samples]):
                    v = parse_verdict(str(val))
                    if v:
                        ref_verdicts[idx] = v

    results = []
    match_count = 0
    total_with_ref = 0

    for idx, row in sample.iterrows():
        prompt = build_prompt(row)
        try:
            raw_output = query_server(server_url, model_name, prompt)
        except Exception as e:
            logger.error("Request failed for row %d: %s", idx, e)
            results.append({"idx": idx, "error": str(e)})
            continue

        vllm_verdict = parse_verdict(raw_output)
        ref = ref_verdicts.get(idx)

        matched = vllm_verdict == ref if (vllm_verdict and ref) else None
        if ref and vllm_verdict:
            total_with_ref += 1
            if matched:
                match_count += 1

        results.append({
            "idx": int(idx),
            "vllm_raw": raw_output[:200],
            "vllm_verdict": vllm_verdict,
            "ref_verdict": ref,
            "match": matched,
        })
        status = "MATCH" if matched else ("MISMATCH" if matched is False else "no-ref")
        logger.info("Row %3d: vllm=%s  ref=%s  [%s]", idx, vllm_verdict, ref, status)

    parity_rate = match_count / total_with_ref if total_with_ref > 0 else None
    summary = {
        "server_url": server_url,
        "model_name": model_name,
        "n_tested": len(results),
        "n_with_reference": total_with_ref,
        "n_matched": match_count,
        "parity_rate": parity_rate,
        "results": results,
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Parity test: vLLM serving vs eval predictions")
    p.add_argument("--server_url", required=True, help="vLLM server URL, e.g. http://localhost:8000")
    p.add_argument("--model_name", default="cld_judge", help="Model name registered in vLLM")
    p.add_argument("--test_data", required=True, help="Test data XLSX (same used in evaluation)")
    p.add_argument("--predictions_xlsx", default=None, help="Reference predictions XLSX from eval")
    p.add_argument("--n_samples", type=int, default=20, help="Number of samples to test")
    p.add_argument("--output_json", default=None, help="Save results to JSON")
    args = p.parse_args()

    logger.info("Parity test: %s (model=%s, n=%d)", args.server_url, args.model_name, args.n_samples)

    # Health check
    try:
        health = requests.get(f"{args.server_url}/health", timeout=10)
        health.raise_for_status()
        logger.info("Server health: OK")
    except Exception as e:
        logger.error("Server health check failed: %s", e)
        sys.exit(1)

    summary = run_parity_test(
        server_url=args.server_url,
        model_name=args.model_name,
        test_data_path=args.test_data,
        predictions_path=args.predictions_xlsx,
        n_samples=args.n_samples,
    )

    rate_str = f"{summary['parity_rate']:.1%}" if summary["parity_rate"] is not None else "N/A"
    logger.info(
        "Parity: %d/%d matched (%s) out of %d tested",
        summary["n_matched"], summary["n_with_reference"], rate_str, summary["n_tested"],
    )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Results saved to %s", args.output_json)

    if summary["parity_rate"] is not None and summary["parity_rate"] < 0.8:
        logger.warning("LOW PARITY (< 80%%) — investigate serving vs eval differences")
        sys.exit(2)


if __name__ == "__main__":
    main()
