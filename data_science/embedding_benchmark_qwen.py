#!/usr/bin/env python3
"""
Benchmark: all-mpnet-base-v2 vs Qwen2.5-7B embeddings via Ollama.
"""
import time
import requests
import numpy as np
from typing import List, Tuple

# 30 sample documents (causal narratives)
SAMPLE_DOCS = [
    "Therapy compliance causes chronic addiction to decrease through improved behavioral patterns and reduced relapse rates.",
    "Social network size leads to healthier lifestyle choices through peer influence and social support mechanisms.",
    "Depression severity is positively associated with social isolation, creating a feedback loop that exacerbates symptoms.",
    "Regular physical exercise reduces cardiovascular disease risk by improving blood pressure and cholesterol levels.",
    "Childhood trauma exposure increases vulnerability to anxiety disorders in adulthood through altered stress response systems.",
    "Socioeconomic status affects health outcomes through differential access to healthcare, nutrition, and safe environments.",
    "Chronic stress elevates cortisol levels, which in turn suppresses immune function and increases disease susceptibility.",
    "Educational attainment is positively correlated with life expectancy due to health literacy and economic opportunities.",
    "Sleep deprivation impairs cognitive function by disrupting memory consolidation and executive control processes.",
    "Smoking cessation leads to improved lung function within weeks, demonstrating the reversibility of tobacco-related damage.",
    "Peer pressure in adolescence contributes to substance abuse initiation through social conformity mechanisms.",
    "Urban green spaces reduce stress levels by providing opportunities for physical activity and nature exposure.",
    "Genetic predisposition interacts with environmental factors to determine obesity risk through epigenetic modifications.",
    "Loneliness increases inflammation markers, suggesting a biological pathway linking social isolation to physical disease.",
    "Mindfulness meditation reduces anxiety symptoms by modulating activity in the amygdala and prefrontal cortex.",
    "Air pollution exposure is linked to respiratory diseases through oxidative stress and inflammatory responses.",
    "Parental involvement in education improves academic outcomes by fostering motivation and providing support.",
    "Unemployment is associated with increased depression rates due to financial stress and loss of purpose.",
    "Vaccination coverage affects herd immunity thresholds, protecting vulnerable populations from infectious disease outbreaks.",
    "Alcohol consumption increases liver disease risk through direct hepatotoxic effects and metabolic disruption.",
    "Social media use correlates with anxiety in adolescents, potentially through social comparison and FOMO mechanisms.",
    "Diet quality influences gut microbiome composition, which affects mental health through the gut-brain axis.",
    "Climate change exacerbates health inequalities by disproportionately affecting low-income communities.",
    "Early intervention in autism spectrum disorder improves developmental outcomes through neuroplasticity mechanisms.",
    "Sedentary behavior increases type 2 diabetes risk independent of physical activity levels.",
    "Cultural factors influence health-seeking behavior, affecting diagnosis and treatment timing.",
    "Noise pollution contributes to cardiovascular disease through chronic stress activation.",
    "Prenatal nutrition affects fetal development with long-term consequences for metabolic health.",
    "Community cohesion is associated with lower crime rates and better mental health outcomes.",
    "Digital health interventions show promise for reaching underserved populations with limited healthcare access.",
]

def benchmark_mpnet(texts: List[str], n_runs: int = 3) -> dict:
    """Benchmark sentence-transformers all-mpnet-base-v2."""
    from sentence_transformers import SentenceTransformer
    
    print("\n" + "="*60)
    print("MODEL: all-mpnet-base-v2 (110M params)")
    print("="*60)
    
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cuda')
    dim = model.get_sentence_embedding_dimension()
    
    # Warmup
    _ = model.encode(texts[:5], show_progress_bar=False)
    
    times = []
    for run in range(n_runs):
        start = time.time()
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.4f}s ({len(texts)} docs)")
    
    mean_total = np.mean(times)
    std_total = np.std(times)
    mean_per_doc = (mean_total / len(texts)) * 1000
    
    print(f"  Mean: {mean_total:.4f}s ± {std_total:.4f}s")
    print(f"  Per document: {mean_per_doc:.2f}ms")
    print(f"  Throughput: {len(texts)/mean_total:.1f} docs/sec")
    
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "model": "all-mpnet-base-v2",
        "params": "110M",
        "dim": dim,
        "ms_per_doc": mean_per_doc,
        "docs_per_sec": len(texts) / mean_total,
        "total_s": mean_total
    }


def benchmark_qwen_ollama(texts: List[str], n_runs: int = 3) -> dict:
    """Benchmark Qwen2.5-7B embeddings via Ollama API."""
    
    print("\n" + "="*60)
    print("MODEL: qwen2.5:7b-instruct-q4_K_M (7.6B params, Q4)")
    print("="*60)
    
    url = "http://localhost:11434/api/embeddings"
    model_name = "qwen2.5:7b-instruct-q4_K_M"
    
    # Warmup
    for t in texts[:3]:
        requests.post(url, json={"model": model_name, "prompt": t})
    
    times = []
    for run in range(n_runs):
        start = time.time()
        embeddings = []
        for t in texts:
            resp = requests.post(url, json={"model": model_name, "prompt": t})
            if resp.status_code == 200:
                embeddings.append(resp.json().get("embedding", []))
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.4f}s ({len(texts)} docs)")
    
    mean_total = np.mean(times)
    std_total = np.std(times)
    mean_per_doc = (mean_total / len(texts)) * 1000
    
    # Get embedding dim
    dim = len(embeddings[0]) if embeddings else 3584
    
    print(f"  Mean: {mean_total:.4f}s ± {std_total:.4f}s")
    print(f"  Per document: {mean_per_doc:.2f}ms")
    print(f"  Throughput: {len(texts)/mean_total:.1f} docs/sec")
    
    return {
        "model": "qwen2.5-7b-q4",
        "params": "7.6B",
        "dim": dim,
        "ms_per_doc": mean_per_doc,
        "docs_per_sec": len(texts) / mean_total,
        "total_s": mean_total
    }


def main():
    print("="*60)
    print("EMBEDDING MODEL BENCHMARK: MPNet vs Qwen")
    print(f"Sample size: {len(SAMPLE_DOCS)} documents")
    print("="*60)
    
    results = {}
    
    # Benchmark MPNet
    results["mpnet"] = benchmark_mpnet(SAMPLE_DOCS)
    
    # Benchmark Qwen
    results["qwen"] = benchmark_qwen_ollama(SAMPLE_DOCS)
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    mpnet = results["mpnet"]
    qwen = results["qwen"]
    
    speedup = qwen["ms_per_doc"] / mpnet["ms_per_doc"]
    
    print(f"\n{'Metric':<25} {'MPNet (110M)':<20} {'Qwen-7B (Q4)':<20}")
    print("-"*65)
    print(f"{'Parameters':<25} {mpnet['params']:<20} {qwen['params']:<20}")
    print(f"{'Embedding dim':<25} {mpnet['dim']:<20} {qwen['dim']:<20}")
    print(f"{'ms per document':<25} {mpnet['ms_per_doc']:<20.2f} {qwen['ms_per_doc']:<20.2f}")
    print(f"{'docs/second':<25} {mpnet['docs_per_sec']:<20.1f} {qwen['docs_per_sec']:<20.1f}")
    print(f"{'Speedup factor':<25} {'1.0x (baseline)':<20} {f'{speedup:.0f}x slower':<20}")
    
    # Extrapolate to full corpus
    corpus_size = 16507
    mpnet_time = corpus_size * mpnet["ms_per_doc"] / 1000
    qwen_time = corpus_size * qwen["ms_per_doc"] / 1000
    
    print(f"\n{'Full corpus (n=16,507)':<25}")
    print(f"{'MPNet':<25} {mpnet_time:.1f}s ({mpnet_time/60:.1f} min)")
    print(f"{'Qwen-7B':<25} {qwen_time:.1f}s ({qwen_time/60:.1f} min)")
    
    # LaTeX for thesis
    print("\n" + "="*60)
    print("LATEX FOR THESIS")
    print("="*60)
    
    print(r"""
\begin{table}[H]
\centering
\caption{Embedding Model Benchmark ($n=30$ causal narratives, RTX GPU)}
\label{tab:embed_benchmark}
\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{ms/doc} & \textbf{docs/s} & \textbf{16K corpus} \\
\midrule""")
    print(f"all-mpnet-base-v2 & {mpnet['params']} & {mpnet['ms_per_doc']:.1f} & {mpnet['docs_per_sec']:.0f} & {mpnet_time:.0f}s \\\\")
    print(f"qwen2.5-7b-q4 (Ollama) & {qwen['params']} & {qwen['ms_per_doc']:.1f} & {qwen['docs_per_sec']:.1f} & {qwen_time/60:.1f}m \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
    
    print(f"\n** MPNet is {speedup:.0f}x faster than Qwen-7B for embeddings **")
    
    return results


if __name__ == "__main__":
    main()










