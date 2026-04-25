# Serving Finetuned LLMs — Key Concepts

This document explains the concepts behind serving the CLD judge models, from what a LoRA adapter is to how it ends up behind an API endpoint on a cloud GPU.

---

## 1. What We Trained vs. What We Serve

During finetuning, we used **QLoRA** (Quantized Low-Rank Adaptation). This involves two things that matter for serving:

- **Base model**: The full pretrained model from HuggingFace (e.g. `Qwen/Qwen3-14B`). This is ~28 GB in FP16. We never modified these weights.
- **LoRA adapter**: A small set of low-rank matrices (~500 MB for 14B) that are added on top of the base model's attention and MLP layers. This is what we actually trained. It lives in the `lora_adapter/` directories.

At training time, the base model was loaded in 4-bit (NF4 quantization via bitsandbytes) to save GPU memory. But that 4-bit format is a **training-only trick** — it's not how we serve the model.

For serving, there are two paths:

### Path A: Base Model + LoRA at Runtime

Load the base model at full precision (FP16) and attach the LoRA adapter at runtime. The serving framework (vLLM) handles the merge transparently.

```
  HuggingFace Hub              Local disk
  ┌─────────────┐         ┌──────────────┐
  │ Qwen/Qwen3  │         │ lora_adapter/ │
  │   -14B      │         │  (506 MB)     │
  │  (~28 GB)   │         └──────┬────────┘
  └──────┬──────┘                │
         │     vLLM loads both   │
         ▼                       ▼
  ┌──────────────────────────────────┐
  │  vLLM server                     │
  │  base weights + LoRA merged      │
  │  on the GPU at runtime           │
  └──────────────────────────────────┘
```

**Pros**: Small adapter files, easy to swap between adapters, base model is downloaded once and cached.
**Cons**: Need enough VRAM for the full FP16 base model.

### Path B: Pre-merged FP16 Checkpoint

Merge the base model + adapter into a single set of weights ahead of time (we have these as `merged_fp16/` directories for some models). Serve it as a plain model, no LoRA needed at runtime.

**Pros**: Simpler serving config, no adapter management.
**Cons**: Large files (28-64 GB), can't swap adapters, need to re-merge if you retrain.

**Our approach**: We use **Path A** (base + LoRA at runtime) because it's more flexible and the adapter files are tiny.

---

## 2. Why Not Serve in 4-bit NF4 Like Training?

The 4-bit NF4 format (bitsandbytes) that we used during training is designed for memory-efficient gradient computation, not fast inference. It:

- Dequantizes weights on-the-fly per layer, which is slow for token generation
- Doesn't support the batching and paged attention optimizations that inference servers use
- Isn't supported by vLLM or other production serving frameworks

For inference, the standard quantization formats are **AWQ** and **GPTQ**, which are designed to be fast at serving time.

---

## 3. vLLM: What It Is and Why We Use It

[vLLM](https://docs.vllm.ai/) is an inference server for LLMs. It takes a model and serves it as an HTTP API that's compatible with the OpenAI chat completions format.

What vLLM does for us:

- **PagedAttention**: Manages GPU memory efficiently so the model + KV cache fit without OOM. Think of it like virtual memory for attention states.
- **Continuous batching**: Groups incoming requests together dynamically, so the GPU stays busy even if requests arrive at different times.
- **Tensor parallelism (TP)**: Splits a model across multiple GPUs. For the 235B model, we set `--tensor-parallel-size 2` to split it across 2 GPUs.
- **LoRA hot-loading**: Can load a base model once and attach/detach LoRA adapters per request. When we pass `--enable-lora --lora-modules cld_judge=/path/to/adapter`, vLLM registers the adapter under the name `cld_judge`.
- **OpenAI-compatible API**: Exposes `/v1/chat/completions`, `/v1/completions`, and `/health` endpoints. Any OpenAI client library works out of the box.

### How a Request Flows

```
Client                          vLLM Server                    GPU
  │                                │                            │
  │  POST /v1/chat/completions     │                            │
  │  model: "cld_judge"            │                            │
  │  messages: [...]               │                            │
  │ ──────────────────────────────►│                            │
  │                                │  Tokenize + apply chat     │
  │                                │  template                  │
  │                                │                            │
  │                                │  Look up "cld_judge" LoRA  │
  │                                │  adapter weights            │
  │                                │                            │
  │                                │  Schedule into next batch   │
  │                                │ ──────────────────────────►│
  │                                │                            │  Forward pass
  │                                │                            │  (base + LoRA)
  │                                │  ◄──────────────────────── │
  │                                │  Detokenize                │
  │  ◄──────────────────────────── │                            │
  │  {"choices": [{"message":      │                            │
  │    {"content": "VERDICT: ..."}}│                            │
```

### The `model` Field in API Requests

When you send `"model": "cld_judge"` in a request, that's the **adapter name**, not the HuggingFace model ID. vLLM maps it to the adapter that was registered at startup via `--lora-modules cld_judge=/path/to/adapter`. If you don't use LoRA (serving a plain or merged model), the model name is the HuggingFace ID or path that was passed to `--model`.

---

## 4. GPU Memory — What Needs to Fit

A GPU has a fixed amount of VRAM. When serving, three things compete for it:

1. **Model weights**: The dominant cost. Qwen3-14B in FP16 is ~28 GB. Qwen3-235B (MoE, all experts loaded) is ~470 GB in FP16.
2. **KV cache**: Stores the attention state for in-flight requests. With `max_model_len=2048`, this is typically 2-4 GB. vLLM manages this automatically via PagedAttention.
3. **LoRA adapter weights**: Small. Our rank-32 adapters are 500 MB to 1.1 GB. Loaded once and shared across requests.

The `--gpu-memory-utilization 0.90` flag tells vLLM to use up to 90% of VRAM, reserving the rest for the CUDA runtime and overhead.

### VRAM Budget per Model

| Model | FP16 Weights | KV Cache (2048 ctx) | LoRA | Total | Min GPU |
|---|---|---|---|---|---|
| Qwen3-14B | ~28 GB | ~2-3 GB | ~0.5 GB | ~31 GB | 1x A100-80GB |
| Qwen3-32B | ~64 GB | ~3-4 GB | ~1.1 GB | ~68 GB | 1x A100-80GB |
| Qwen3-235B (FP16) | ~470 GB | ~4 GB | ~25 GB | ~499 GB | Not feasible |
| Qwen3-235B (AWQ 4-bit) | ~120 GB | ~4 GB | ~25 GB | ~149 GB | 2x H100-80GB |

---

## 5. AWQ Quantization — Making 235B Servable

The 235B model doesn't fit on any reasonable GPU config at FP16. AWQ (Activation-aware Weight Quantization) compresses the weights to 4-bit while preserving quality for inference:

- Analyzes which weight channels matter most (using a small calibration dataset)
- Quantizes less important channels more aggressively
- Produces a model that vLLM can load natively with `--quantization awq`
- ~4x VRAM reduction: 470 GB -> ~120 GB, which fits on 2x H100-80GB

This is different from the NF4 quantization used during training. AWQ is an **inference-optimized** format — the weights stay quantized during the forward pass and specialized CUDA kernels do the math directly in low precision.

```
  Original FP16 Model       AWQ Quantization         Quantized Model
  ┌─────────────────┐      ┌──────────────┐      ┌──────────────────┐
  │  470 GB          │ ──►  │ Calibrate on │ ──►  │  120 GB           │
  │  (235B params)   │      │ pileval data │      │  4-bit weights    │
  └─────────────────┘      └──────────────┘      │  vLLM-compatible  │
                                                  └──────────────────┘
```

You run quantization once (on a GPU big enough to load the FP16 model, or using offloading), save the result, then serve from the quantized checkpoint forever.

---

## 6. Tensor Parallelism (TP) — Splitting Across GPUs

When a model doesn't fit on a single GPU, tensor parallelism shards the weight matrices across multiple GPUs. Each GPU holds a slice of every layer, and they communicate intermediate results via fast NVLink interconnects.

```
  TP=2 (two GPUs)
  ┌──────────────┐    ┌──────────────┐
  │   GPU 0       │    │   GPU 1       │
  │ Left half of  │◄──►│ Right half of │
  │ each layer    │    │ each layer    │
  └──────────────┘    └──────────────┘
       NVLink / PCIe communication
```

- **TP=1**: Entire model on one GPU. Works for 14B and 32B on an A100-80GB.
- **TP=2**: Split across 2 GPUs. Required for 235B AWQ on 2x H100.
- **TP=4**: Split across 4 GPUs. Used for very large models or higher throughput.

TP must be set at server startup (`--tensor-parallel-size N`). All N GPUs must be on the same machine.

---

## 7. Batching and Scheduling — How vLLM Handles Multiple Requests

LLM inference is heavily GPU-bound. Batching multiple requests together is the single biggest lever for throughput. vLLM handles this automatically, but understanding how it works helps you tune it.

### Static Batching vs. Continuous Batching

Traditional inference (like our `vllm_inference.py` batch script) uses **static batching**: collect N prompts, run them all through the model, wait for all to finish. The problem is that short outputs wait for long outputs to complete, wasting GPU cycles.

vLLM uses **continuous batching** (also called iteration-level scheduling): it decodes one token at a time across all active requests, and as soon as a request finishes (hits EOS or `max_tokens`), a waiting request takes its slot immediately.

```
  Static batching                    Continuous batching (vLLM)
  ┌────────────────────────┐         ┌────────────────────────┐
  │ Req A: ████████████     │         │ Req A: ████████████     │
  │ Req B: █████ (idle..)   │         │ Req B: █████             │
  │ Req C: ██████████       │         │ Req C: ██████████        │
  │                         │         │ Req D:      ████████     │
  │ All wait for A to end   │         │ D starts when B finishes │
  └────────────────────────┘         └────────────────────────┘
```

For our use case (short verdict outputs, typically 1-5 tokens), continuous batching means the GPU almost never sits idle between requests.

### Prefill vs. Decode Phases

Every request goes through two phases:

1. **Prefill**: Process the entire prompt in one forward pass (parallelizable, fast). For our prompts (~100-200 tokens), this takes milliseconds.
2. **Decode**: Generate output tokens one at a time (sequential, slower). For "VERDICT: CORRECT" that's about 3-5 tokens.

vLLM can interleave prefill and decode across different requests — while it's decoding tokens for request A, it can prefill the prompt for request B in the same batch step.

### Key Serving Parameters

These are the vLLM parameters that matter most for our workload, what they control, and how to think about tuning them:

#### `--max-model-len` (we use: 2048)

The maximum combined length of prompt + output in tokens. Determines how much KV cache vLLM pre-allocates. Our prompts are ~100-200 tokens and outputs are ~3-500 tokens, so 2048 is plenty. Setting this lower than the model's native context (e.g. 32k for Qwen3) saves significant VRAM that goes to serving more concurrent requests instead.

#### `--gpu-memory-utilization` (we use: 0.90)

Fraction of VRAM that vLLM is allowed to use. After loading model weights + LoRA, the remaining memory goes to KV cache. Higher values = more concurrent requests but less margin for CUDA overhead spikes. 0.90 is a safe default. Drop to 0.85 if you see OOM errors.

#### `--max-num-seqs` (default: 256)

Maximum number of requests that can be batched together in a single forward pass. For our short-output workload this rarely matters — we hit memory limits before sequence limits.

#### `--max-num-batched-tokens` (default: auto)

Maximum number of tokens across all sequences in a single scheduler step. vLLM auto-calculates this based on available KV cache memory. Only override this if you need to limit GPU utilization for latency reasons.

#### `--enforce-eager` (default: off)

Disables CUDA graph compilation. CUDA graphs speed up the decode phase by replaying a captured kernel sequence instead of re-launching kernels per step. Trades ~10-30% decode throughput for faster startup and less memory. Useful for debugging but leave it off for production.

#### `--enable-chunked-prefill` (default: off)

Splits long prefills into chunks that can interleave with decode steps. Useful when some prompts are very long and would otherwise block decoding for all other requests. Not critical for our workload (short prompts) but worth enabling for mixed-length traffic.

#### `--max-loras` and `--max-lora-rank`

`--max-loras 1` means vLLM pre-allocates space for 1 active adapter. `--max-lora-rank 32` matches our adapter rank. If you load multiple adapters for A/B testing, increase `--max-loras` (costs VRAM proportional to adapter count).

### Temperature and Sampling Parameters (Per-Request)

These are set per-request, not at server startup:

| Parameter | Our setting | What it does |
|---|---|---|
| `temperature` | 0.0 | Greedy decoding — always pick the highest-probability token. Deterministic output. |
| `max_tokens` | 512 | Maximum output length. Our verdicts are 1-5 tokens, but 512 allows for reason+verdict variants. |
| `top_p` | 1.0 (default) | Nucleus sampling threshold. Irrelevant at temperature=0. |
| `top_k` | -1 (default) | Top-k sampling. Irrelevant at temperature=0. |
| `frequency_penalty` | 0.0 (default) | Penalizes token repetition. Not needed for our short outputs. |
| `stop` | not set | Stop sequences. Could set `["\n"]` to stop after the verdict line, but `max_tokens` handles this. |

For CLD judge inference, **greedy decoding (temperature=0)** is critical for reproducibility. Any temperature > 0 introduces randomness that makes parity testing and result comparison unreliable.

### Throughput vs. Latency Tradeoff

Higher batch sizes = higher throughput (tokens/second across all requests) but higher per-request latency (each request waits longer for its turn in the batch). For our batch evaluation use case, throughput matters more. For a live API, latency matters more.

```
  Throughput-optimized              Latency-optimized
  (batch eval)                      (live API)
  ┌─────────────────────┐          ┌─────────────────────┐
  │ Large batches         │          │ Small batches         │
  │ gpu-mem-util=0.95    │          │ gpu-mem-util=0.85    │
  │ max-num-seqs=256     │          │ max-num-seqs=32      │
  │ Higher total tok/s   │          │ Lower per-req latency │
  └─────────────────────┘          └─────────────────────┘
```

vLLM's defaults are biased toward throughput, which is right for our evaluation workload.

---

## 8. Docker — Why and What the Dockerfile Does

The Dockerfile packages our serving script on top of the official `vllm/vllm-openai` image (which has vLLM, PyTorch, CUDA, and all dependencies pre-installed). This gives us:

- **Reproducibility**: Same environment everywhere (cloud GPU, Snellius, local).
- **Portability**: Cloud providers (RunPod, etc.) natively run Docker images.
- **Isolation**: No dependency conflicts with the training environment.

### What's in the Container vs. What's Mounted

```
  Container (baked in)                  Mounted at runtime
  ┌──────────────────────┐         ┌───────────────────────┐
  │ vLLM + CUDA + PyTorch │         │ /adapters/             │
  │ serve_judge_vllm.py   │         │   └── lora_adapter/    │
  │ CLD_* env var defaults │         │                       │
  └──────────────────────┘         │ /root/.cache/hf/       │
                                    │   └── Qwen3-14B/       │
                                    │       (downloaded once) │
                                    └───────────────────────┘
```

The **base model weights** are not in the Docker image (they're 28+ GB). They're downloaded from HuggingFace on first run and cached in a Docker volume so subsequent starts are fast.

The **LoRA adapter** is mounted as a read-only volume, so you can swap adapters without rebuilding the image.

---

## 9. The OpenAI-Compatible API

vLLM exposes the same API shape as the OpenAI API. This means any code that talks to OpenAI can talk to our server by changing the base URL.

### Key Endpoints

| Endpoint | Method | What It Does |
|---|---|---|
| `/health` | GET | Returns 200 if the model is loaded and ready |
| `/v1/chat/completions` | POST | Chat-style inference (system + user messages) |
| `/v1/completions` | POST | Raw text completion |
| `/v1/models` | GET | Lists available models/adapters |

### Example Request

```bash
curl http://<server>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cld_judge",
    "messages": [
      {"role": "system", "content": "You are a causal diagram expert..."},
      {"role": "user", "content": "SOURCE: Depression\nTARGET: ..."}
    ],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

### Example Response

```json
{
  "id": "cmpl-abc123",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "VERDICT: CORRECT"
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 142, "completion_tokens": 3, "total_tokens": 145}
}
```

### Using It from Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://<server>:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="cld_judge",
    messages=[
        {"role": "system", "content": "You are a causal diagram expert..."},
        {"role": "user", "content": "SOURCE: ...\nTARGET: ..."},
    ],
    temperature=0.0,
    max_tokens=512,
)
print(response.choices[0].message.content)  # "VERDICT: CORRECT"
```

---

## 10. Cloud GPU Deployment — What Actually Happens

Here's the concrete sequence when you deploy to a cloud provider like RunPod:

1. **Provision a GPU instance**: You pick a GPU type (A100-80GB for 14B/32B, 2x H100 for 235B) and a Docker image (`vllm/vllm-openai:latest`). The provider spins up a virtual machine with that GPU attached.

2. **Upload the LoRA adapter**: rsync or SCP your ~500 MB adapter from local to the instance. The base model weights (28+ GB) are downloaded from HuggingFace — you don't upload those.

3. **Start vLLM**: The serve command loads the base model onto the GPU, attaches the LoRA adapter, and starts listening on port 8000.

4. **First-run model download**: On the first run, vLLM downloads the base model from HuggingFace to `~/.cache/huggingface/`. This takes a few minutes. On subsequent runs (if the cache is on a persistent volume), it starts instantly.

5. **Ready to serve**: Once the `/health` endpoint returns 200, the model is loaded and accepting requests. Typical cold start: 2-5 minutes for 14B, 5-10 minutes for 32B.

### Cost Structure

You pay for GPU time by the hour (or by the second on serverless platforms). The model is only in VRAM while the instance is running — shut it down and you stop paying.

| | On-demand (always on) | Serverless (pay per inference) |
|---|---|---|
| Startup | Instant (already running) | Cold start: 30-120s |
| Cost model | $/hour regardless of usage | $/second of compute |
| Good for | Steady request volume | Sporadic / batch usage |

---

## 11. Parity Testing — Why It Matters

The evaluation pipeline (`evaluate_judge_multigpu.py`) uses HuggingFace Transformers with bitsandbytes 4-bit quantization. The serving pipeline uses vLLM with FP16 weights. These are different code paths with different numerical behavior.

Parity testing sends the same prompts to the vLLM server and compares the verdicts against what the evaluation pipeline produced. We expect high (>90%) but not necessarily perfect agreement because:

- FP16 serving vs. NF4 training-time quantization produce slightly different logits
- vLLM's sampling implementation may differ from HuggingFace's at the margins
- Both use greedy decoding (`temperature=0`), so determinism is mostly preserved

The `parity_test.py` script automates this: it reads the test data, sends prompts to the server, and reports the match rate.

---

## Glossary

| Term | Meaning |
|---|---|
| **LoRA** | Low-Rank Adaptation — parameter-efficient finetuning that trains small matrices added to frozen base weights |
| **QLoRA** | LoRA but with the base model quantized to 4-bit during training to save VRAM |
| **AWQ** | Activation-aware Weight Quantization — an inference-optimized 4-bit format |
| **FP16** | 16-bit floating point — the standard precision for serving LLMs |
| **NF4** | NormalFloat 4-bit — bitsandbytes training quantization format (not for serving) |
| **Tensor Parallelism (TP)** | Splitting model layers across multiple GPUs |
| **KV Cache** | Cached key/value attention states for in-flight requests |
| **PagedAttention** | vLLM's memory management for KV cache, avoiding fragmentation |
| **MoE** | Mixture of Experts — architecture where only a subset of parameters activate per token (Qwen3-235B has 235B total but ~22B active) |
| **vLLM** | High-performance LLM inference server |
| **HuggingFace Hub** | Model repository where base models are hosted and downloaded from |
| **Adapter hot-loading** | Loading/swapping LoRA adapters at runtime without restarting the server |
| **Continuous batching** | Scheduling that adds/removes requests from GPU batches at each decode step, rather than waiting for the whole batch to finish |
| **Prefill** | The phase where the full prompt is processed in one parallel forward pass |
| **Decode** | The phase where output tokens are generated one at a time (auto-regressively) |
| **CUDA graphs** | Pre-captured GPU kernel sequences that eliminate per-step launch overhead during decode |
| **Greedy decoding** | Always picking the highest-probability token (temperature=0), producing deterministic output |
| **max_model_len** | Maximum sequence length (prompt + output). Controls KV cache pre-allocation. |
| **gpu_memory_utilization** | Fraction of VRAM vLLM may use. Leftover after weights goes to KV cache for concurrent requests. |
