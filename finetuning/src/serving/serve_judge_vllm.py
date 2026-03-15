#!/usr/bin/env python3
"""
Serve CLD judge model via vLLM OpenAI-compatible API.
Supports LoRA adapter or merged checkpoint.

Usage:
    python -m finetuning.src.serving.serve_judge_vllm \\
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \\
        --lora_path finetuning/runs/mistral7b_judge_gtsynth/lora_adapter \\
        --host 0.0.0.0 --port 8000
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve CLD judge via vLLM")
    p.add_argument("--base_model", required=True, help="Base model id or path")
    p.add_argument("--lora_path", default=None, help="LoRA adapter path (omit for merged model)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lora_path = Path(args.lora_path).resolve() if (args.lora_path and args.lora_path.strip()) else None

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.base_model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    if lora_path and lora_path.exists():
        cmd.extend(["--enable-lora", "--lora-modules", f"cld_judge={lora_path}"])
        logger.info("Serving with LoRA: %s", lora_path)
    else:
        logger.info("Serving base model only")

    logger.info("Starting vLLM server: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
