#!/usr/bin/env python3
"""
Benchmark embedding models for thesis model selection rationale.
Compares all-mpnet-base-v2 vs larger alternatives on 30 sample documents.
"""
import time
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Sample documents: mix of causal narratives and citation-like texts
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

def benchmark_model(model_name: str, texts: List[str], n_runs: int = 3) -> Tuple[float, float, int]:
    """Benchmark a sentence-transformers model.
    
    Returns: (mean_time_per_doc_ms, total_time_s, embedding_dim)
    """
    from sentence_transformers import SentenceTransformer
    
    print(f"\n{'='*60}")
    print(f"Loading: {model_name}")
    
    # Load model
    load_start = time.time()
    try:
        model = SentenceTransformer(model_name, device='cuda')
    except Exception as e:
        print(f"  GPU failed ({e}), falling back to CPU")
        model = SentenceTransformer(model_name, device='cpu')
    load_time = time.time() - load_start
    
    dim = model.get_sentence_embedding_dimension()
    print(f"  Loaded in {load_time:.2f}s, embedding dim: {dim}")
    
    # Warmup
    _ = model.encode(texts[:5], show_progress_bar=False)
    
    # Benchmark runs
    times = []
    for run in range(n_runs):
        start = time.time()
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.4f}s ({len(texts)} docs)")
    
    mean_total = np.mean(times)
    std_total = np.std(times)
    mean_per_doc = (mean_total / len(texts)) * 1000  # ms
    
    print(f"  Mean: {mean_total:.4f}s ± {std_total:.4f}s")
    print(f"  Per document: {mean_per_doc:.2f}ms")
    print(f"  Throughput: {len(texts)/mean_total:.1f} docs/sec")
    
    # Cleanup GPU memory
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    return mean_per_doc, mean_total, dim


def main():
    print("="*60)
    print("EMBEDDING MODEL BENCHMARK")
    print(f"Sample size: {len(SAMPLE_DOCS)} documents")
    print("="*60)
    
    # Models to benchmark
    models = [
        ("sentence-transformers/all-mpnet-base-v2", "110M"),  # Our current model
        ("sentence-transformers/all-MiniLM-L6-v2", "22M"),    # Smaller alternative
        ("BAAI/bge-small-en-v1.5", "33M"),                    # BGE small
        ("BAAI/bge-large-en-v1.5", "335M"),                   # BGE large
    ]
    
    # Try to add Qwen if available
    try:
        from sentence_transformers import SentenceTransformer
        # Test if larger model can be loaded
        models.append(("BAAI/bge-m3", "568M"))
    except:
        pass
    
    results = {}
    for model_name, params in models:
        try:
            ms_per_doc, total_s, dim = benchmark_model(model_name, SAMPLE_DOCS)
            results[model_name] = {
                "params": params,
                "ms_per_doc": ms_per_doc,
                "total_s": total_s,
                "dim": dim,
                "docs_per_sec": len(SAMPLE_DOCS) / total_s
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            results[model_name] = {"error": str(e)}
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Model':<45} {'Params':<8} {'ms/doc':<10} {'docs/s':<10} {'Dim':<6}")
    print("-"*80)
    
    for model_name, data in results.items():
        if "error" in data:
            print(f"{model_name[:44]:<45} FAILED")
        else:
            short_name = model_name.split('/')[-1]
            print(f"{short_name:<45} {data['params']:<8} {data['ms_per_doc']:<10.2f} {data['docs_per_sec']:<10.1f} {data['dim']:<6}")
    
    # Extrapolate to full corpus
    print("\n" + "="*60)
    print("EXTRAPOLATION TO FULL CORPUS (n=16,507 edges)")
    print("="*60)
    
    for model_name, data in results.items():
        if "error" not in data:
            short_name = model_name.split('/')[-1]
            full_time_s = 16507 * data['ms_per_doc'] / 1000
            full_time_min = full_time_s / 60
            print(f"{short_name:<45} {full_time_s:>8.1f}s ({full_time_min:>5.1f} min)")
    
    # LaTeX output
    print("\n" + "="*60)
    print("LATEX TABLE FOR THESIS")
    print("="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Embedding Model Benchmark (30 samples, GPU)}")
    print(r"\footnotesize")
    print(r"\begin{tabular}{@{}lccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Params} & \textbf{ms/doc} & \textbf{docs/s} & \textbf{Dim} & \textbf{16K corpus} \\")
    print(r"\midrule")
    
    for model_name, data in results.items():
        if "error" not in data:
            short_name = model_name.split('/')[-1].replace('_', r'\_')
            full_time = 16507 * data['ms_per_doc'] / 1000
            if full_time < 60:
                time_str = f"{full_time:.0f}s"
            else:
                time_str = f"{full_time/60:.1f}m"
            print(f"{short_name} & {data['params']} & {data['ms_per_doc']:.1f} & {data['docs_per_sec']:.0f} & {data['dim']} & {time_str} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    return results


if __name__ == "__main__":
    main()










