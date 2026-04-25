#!/usr/bin/env python3
"""Serve CLD judge model via vLLM OpenAI-compatible API.

Supports LoRA adapter hot-loading or pre-merged checkpoints.
Configurable via CLI args or environment variables (CLD_*).

Usage (CLI):
    python -m finetuning.src.serving.serve_judge_vllm \
        --base_model Qwen/Qwen3-14B \
        --lora_path finetuning/runs/.../lora_adapter

Usage (env vars):
    CLD_BASE_MODEL=Qwen/Qwen3-14B \
    CLD_LORA_PATH=/adapters/qwen3_14b_lit_vo \
    python -m finetuning.src.serving.serve_judge_vllm
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PRESETS: dict[str, dict[str, str | int]] = {
    "qwen3-14b": {
        "base_model": "Qwen/Qwen3-14B",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
    },
    "qwen3-32b": {
        "base_model": "Qwen/Qwen3-32B",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
    },
    "qwen3-235b": {
        "base_model": "Qwen/Qwen3-235B-A22B",
        "tensor_parallel_size": 2,
        "max_model_len": 2048,
    },
}


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"CLD_{name}") or default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve CLD judge via vLLM")
    p.add_argument(
        "--preset",
        choices=list(MODEL_PRESETS),
        default=_env("PRESET"),
        help="Model preset (applies base_model, TP, max_model_len defaults)",
    )
    p.add_argument("--base_model", default=_env("BASE_MODEL"), help="HF model id or local path")
    p.add_argument("--lora_path", default=_env("LORA_PATH"), help="LoRA adapter path (omit for merged model)")
    p.add_argument("--lora_name", default=_env("LORA_NAME", "cld_judge"), help="Registered adapter name")
    p.add_argument("--host", default=_env("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env("PORT", "8000")))
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--quantization", default=_env("QUANTIZATION"), help="e.g. awq, gptq, None")
    p.add_argument("--dtype", default=_env("DTYPE", "auto"), help="Model dtype (auto, float16, bfloat16)")
    p.add_argument("--gpu_memory_utilization", type=float, default=float(_env("GPU_MEM_UTIL", "0.90")))
    return p.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply preset defaults, then env-var overrides."""
    preset = MODEL_PRESETS.get(args.preset or "", {})
    if not args.base_model:
        args.base_model = preset.get("base_model")
    if args.tensor_parallel_size is None:
        tp_env = _env("TENSOR_PARALLEL_SIZE")
        args.tensor_parallel_size = int(tp_env) if tp_env else int(preset.get("tensor_parallel_size", 1))
    if args.max_model_len is None:
        ml_env = _env("MAX_MODEL_LEN")
        args.max_model_len = int(ml_env) if ml_env else int(preset.get("max_model_len", 2048))
    if not args.base_model:
        raise SystemExit("ERROR: --base_model or --preset is required (or set CLD_BASE_MODEL)")
    return args


def build_vllm_cmd(args: argparse.Namespace) -> list[str]:
    lora_path = Path(args.lora_path).resolve() if args.lora_path and args.lora_path.strip() else None

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.base_model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    if lora_path and lora_path.exists():
        cmd.extend([
            "--enable-lora",
            "--lora-modules", f"{args.lora_name}={lora_path}",
            "--max-loras", "1",
            "--max-lora-rank", "32",
        ])
        logger.info("LoRA adapter: %s (registered as '%s')", lora_path, args.lora_name)
    elif lora_path:
        logger.warning("LoRA path does not exist, serving base model only: %s", lora_path)
    else:
        logger.info("No LoRA adapter specified, serving base model only")

    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
        logger.info("Quantization: %s", args.quantization)

    return cmd


def main() -> None:
    args = resolve_args(parse_args())

    logger.info("Model  : %s", args.base_model)
    logger.info("TP     : %d", args.tensor_parallel_size)
    logger.info("MaxLen : %d", args.max_model_len)
    logger.info("Port   : %d", args.port)

    cmd = build_vllm_cmd(args)
    logger.info("Command: %s", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
