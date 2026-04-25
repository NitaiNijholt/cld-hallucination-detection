#!/usr/bin/env python3
"""Quantize a merged FP16 model to AWQ 4-bit for efficient vLLM serving.

Must be run on a machine with enough GPU VRAM to load the FP16 model:
- 14B: ~28 GB  (1x A100-40GB)
- 32B: ~64 GB  (1x A100-80GB)
- 235B: ~470GB (needs multi-GPU or offloading — better to quantize the
         base model from HF directly, then apply LoRA at serve time)

Usage:
    pip install autoawq

    # Quantize merged 14B model:
    python -m finetuning.src.serving.quantize_awq \
        --model_path finetuning/runs/.../merged_fp16 \
        --output_path finetuning/runs/.../awq_4bit \
        --calib_samples 128

    # Quantize HF base model directly (for 235B LoRA serving):
    python -m finetuning.src.serving.quantize_awq \
        --model_path Qwen/Qwen3-235B-A22B \
        --output_path /workspace/qwen3_235b_awq \
        --calib_samples 128
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Quantize model to AWQ 4-bit")
    p.add_argument("--model_path", required=True, help="FP16 model path or HF model id")
    p.add_argument("--output_path", required=True, help="Output directory for AWQ model")
    p.add_argument("--w_bit", type=int, default=4, help="Weight bit width")
    p.add_argument("--q_group_size", type=int, default=128, help="Quantization group size")
    p.add_argument("--calib_samples", type=int, default=128, help="Calibration dataset samples")
    p.add_argument("--calib_seq_len", type=int, default=512, help="Calibration sequence length")
    args = p.parse_args()

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("autoawq is required: pip install autoawq")
        sys.exit(1)

    logger.info("Loading model: %s", args.model_path)
    model = AutoAWQForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }
    logger.info("Quantizing with config: %s", quant_config)
    logger.info("Calibration: %d samples, seq_len=%d", args.calib_samples, args.calib_seq_len)

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data="pileval",
        n_samples=args.calib_samples,
        seqlen=args.calib_seq_len,
    )

    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output))
    tokenizer.save_pretrained(str(output))
    logger.info("AWQ model saved to: %s", output)
    logger.info("Serve with: vllm --model %s --quantization awq", output)


if __name__ == "__main__":
    main()
