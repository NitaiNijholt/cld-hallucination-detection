#!/usr/bin/env python3
"""
Embedding Model Benchmark: MPNet vs Qwen3-Embedding
Uses real causal narratives from actual CLD experiments.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Load real CLD motivations
def load_cld_motivations(n: int = 30) -> List[str]:
    """Load real causal narratives from CLD experiment data."""
    df = pd.read_excel('/home/nitai/code/causalix.ai/corrected_1c6dbdd4_to_8b63590a_20251215_191428_20251215_191500.xlsx')
    motivations = df['Motivation'].dropna().tolist()[:n]
    return [str(m).strip() for m in motivations]

def benchmark_model(model_name: str, documents: List[str], n_runs: int = 3) -> Tuple[float, float, int]:
    """Benchmark a model and return mean time, std, and embedding dim."""
    times = []
    embed_dim = 0
    
    if model_name == "all-mpnet-base-v2":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Warmup
        _ = model.encode(documents[:5])
        
        for _ in range(n_runs):
            start = time.perf_counter()
            embeddings = model.encode(documents)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            embed_dim = embeddings.shape[1]
            
    elif model_name == "Alibaba-NLP/gte-Qwen2-7B-instruct":
        # Use sentence-transformers with trust_remote_code for Qwen embedding model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        
        # Warmup
        _ = model.encode(documents[:5])
        
        for _ in range(n_runs):
            start = time.perf_counter()
            embeddings = model.encode(documents)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            embed_dim = embeddings.shape[1]
            
    elif model_name.startswith("ollama:"):
        import requests
        ollama_model = model_name.replace("ollama:", "")
        
        def get_ollama_embedding(text: str) -> List[float]:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={'model': ollama_model, 'prompt': text}
            )
            return response.json()['embedding']
        
        # Warmup
        _ = get_ollama_embedding(documents[0])
        
        for _ in range(n_runs):
            start = time.perf_counter()
            embeddings = [get_ollama_embedding(doc) for doc in documents]
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            embed_dim = len(embeddings[0])
    
    return np.mean(times), np.std(times), embed_dim

def main():
    print("=" * 60)
    print("EMBEDDING MODEL BENCHMARK: MPNet vs Qwen")
    print("Using REAL causal narratives from CLD experiments")
    print("=" * 60)
    
    # Load real CLD motivations
    documents = load_cld_motivations(30)
    n_docs = len(documents)
    
    print(f"\nLoaded {n_docs} real causal narratives from CLD data")
    print(f"Sample: '{documents[0][:80]}...'")
    print(f"Avg length: {np.mean([len(d) for d in documents]):.0f} chars")
    
    results = {}
    
    # Benchmark MPNet (what we use)
    print(f"\n{'=' * 60}")
    print("MODEL: all-mpnet-base-v2 (110M params)")
    print("=" * 60)
    
    try:
        mean_time, std_time, embed_dim = benchmark_model("all-mpnet-base-v2", documents)
        ms_per_doc = (mean_time / n_docs) * 1000
        docs_per_sec = n_docs / mean_time
        
        results["mpnet"] = {
            "params": "110M",
            "embed_dim": embed_dim,
            "mean_time": mean_time,
            "std_time": std_time,
            "ms_per_doc": ms_per_doc,
            "docs_per_sec": docs_per_sec
        }
        
        for i in range(3):
            print(f"  Run {i+1}: {mean_time:.4f}s ({n_docs} docs)")
        print(f"  Mean: {mean_time:.4f}s ± {std_time:.4f}s")
        print(f"  Per document: {ms_per_doc:.2f}ms")
        print(f"  Throughput: {docs_per_sec:.1f} docs/sec")
        print(f"  Embedding dim: {embed_dim}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Benchmark Qwen3-Embedding-8B via Ollama (MTEB leader)
    print(f"\n{'=' * 60}")
    print("MODEL: Qwen3-Embedding-8B via Ollama (8B params, Q4_K_M)")
    print("=" * 60)
    
    try:
        mean_time, std_time, embed_dim = benchmark_model("ollama:dengcao/Qwen3-Embedding-8B:Q4_K_M", documents)
        ms_per_doc = (mean_time / n_docs) * 1000
        docs_per_sec = n_docs / mean_time
        
        results["qwen_ollama"] = {
            "params": "8B (Q4)",
            "embed_dim": embed_dim,
            "mean_time": mean_time,
            "std_time": std_time,
            "ms_per_doc": ms_per_doc,
            "docs_per_sec": docs_per_sec
        }
        
        for i in range(3):
            print(f"  Run {i+1}: {mean_time:.4f}s ({n_docs} docs)")
        print(f"  Mean: {mean_time:.4f}s ± {std_time:.4f}s")
        print(f"  Per document: {ms_per_doc:.2f}ms")
        print(f"  Throughput: {docs_per_sec:.1f} docs/sec")
        print(f"  Embedding dim: {embed_dim}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Print comparison
    if len(results) == 2:
        print(f"\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        mpnet = results["mpnet"]
        qwen = results["qwen_ollama"]
        
        speedup = qwen["ms_per_doc"] / mpnet["ms_per_doc"]
        
        print(f"\n{'Metric':<26}{'MPNet (110M)':<21}{'Qwen3-Emb-8B (Q4)':<20}")
        print("-" * 65)
        print(f"{'Parameters':<26}{mpnet['params']:<21}{qwen['params']:<20}")
        print(f"{'Embedding dim':<26}{mpnet['embed_dim']:<21}{qwen['embed_dim']:<20}")
        print(f"{'ms per document':<26}{mpnet['ms_per_doc']:<21.2f}{qwen['ms_per_doc']:<20.2f}")
        print(f"{'docs/second':<26}{mpnet['docs_per_sec']:<21.1f}{qwen['docs_per_sec']:<20.1f}")
        print(f"{'Speedup factor':<26}{'1.0x (baseline)':<21}{f'{speedup:.0f}x slower':<20}")
        
        # Extrapolate to full corpus
        corpus_size = 16507
        mpnet_corpus_time = corpus_size / mpnet["docs_per_sec"]
        qwen_corpus_time = corpus_size / qwen["docs_per_sec"]
        
        print(f"\nFull corpus (n={corpus_size:,})")
        print(f"MPNet:                    {mpnet_corpus_time:.1f}s ({mpnet_corpus_time/60:.1f} min)")
        print(f"Qwen3-Embed-8B:           {qwen_corpus_time:.1f}s ({qwen_corpus_time/60:.1f} min)")
        
        # Generate LaTeX
        print(f"\n{'=' * 60}")
        print("LATEX FOR THESIS")
        print("=" * 60)
        print(f"""
\\begin{{table}}[H]
\\centering
\\caption{{Embedding Model Benchmark ($n=30$ causal narratives from CLD)}}
\\label{{tab:embed_benchmark}}
\\footnotesize
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
\\textbf{{Model}} & \\textbf{{Params}} & \\textbf{{ms/doc}} & \\textbf{{docs/s}} & \\textbf{{16K corpus}} \\\\
\\midrule
all-mpnet-base-v2 & 110M & {mpnet['ms_per_doc']:.1f} & {mpnet['docs_per_sec']:.0f} & {mpnet_corpus_time:.0f}s \\\\
Qwen3-Embedding-8B (Q4) & 8B & {qwen['ms_per_doc']:.1f} & {qwen['docs_per_sec']:.1f} & {qwen_corpus_time/60:.1f}m \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

if __name__ == "__main__":
    main()

