# Embedding Model Benchmark

Comparison of embedding models for cosine similarity computation in hallucination detection.

## Purpose
Justify the choice of `all-mpnet-base-v2` over larger models like Qwen3-Embedding-8B for our batch workload.

## Data
- **n=30** real causal narratives from CLD experiments
- Source: `corrected_1c6dbdd4_to_8b63590a_20251215_191428_20251215_191500.xlsx`
- Average document length: 451 characters

## Models Tested
1. **all-mpnet-base-v2** (110M params, 768-dim embeddings) - via sentence-transformers
2. **Qwen3-Embedding-8B** (8B params Q4_K_M, 4096-dim embeddings) - via Ollama

## Key Results
| Model | ms/doc | docs/s | 16K corpus |
|-------|--------|--------|------------|
| MPNet | 3.0 | 336 | 49s |
| Qwen3-Embedding-8B | 67.3 | 14.9 | 18.5m |

**MPNet is 23x faster** while Qwen3-Embedding-8B tops MTEB leaderboard (score 70.58).

## Files
- `embedding_benchmark_real_cld.py` - Benchmark script
- `benchmark_results.txt` - Full output with LaTeX table
- `README.md` - This file

## Run
```bash
cd /home/nitai/code/causalix.ai
.venv/bin/python final_runs/embedding_model_benchmark/embedding_benchmark_real_cld.py
```

## Date
December 16, 2025










