"""vLLM-based batch inference for CLD judge evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_inference_vllm(
    prompts: list[str],
    base_model: str,
    lora_path: str | None,
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> list[str]:
    """
    Run batch inference via vLLM with optional LoRA adapter.
    Uses greedy decoding (temperature=0).
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as e:
        raise ImportError("vLLM required for --inference_backend vllm. pip install vllm") from e

    lora_path_resolved = str(Path(lora_path).resolve()) if lora_path else None
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    llm = LLM(
        model=base_model,
        enable_lora=bool(lora_path_resolved),
        max_loras=1 if lora_path_resolved else 0,
        max_model_len=2048,
    )

    lora_request = None
    if lora_path_resolved:
        lora_request = LoRARequest("cld_judge", 1, lora_path_resolved)

    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="  Inference (vLLM)"):
        batch = prompts[i : i + batch_size]
        gen = llm.generate(batch, sampling_params=sampling, lora_request=lora_request)
        for o in gen:
            text = o.outputs[0].text if o.outputs else ""
            outputs.append(text)
    return outputs
