"""
Multi-Model Experiment Runner: TruthfulQA Spectral Analysis

This script runs the spectral hallucination detection experiment
across multiple LLM types to test generalization of findings.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')
import traceback


def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies = {
        'transformer_lens': 'TransformerLens (pip install transformer-lens)',
        'transformers': 'HuggingFace Transformers',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'torch': 'PyTorch'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install transformer-lens transformers scipy scikit-learn pandas numpy torch")
        return False
    
    return True


def run_phase_1(sample_limit: int = 120):
    """Phase 1: Data Preparation."""
    print("\n" + "="*80)
    print("PHASE 1: DATA PREPARATION")
    print("="*80)
    
    import sys
    import importlib.util
    BASE_DIR = Path(__file__).parent.resolve()
    spec = importlib.util.spec_from_file_location("data_prep", str(BASE_DIR / "1_data_preparation.py"))
    data_prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_prep)
    TruthfulQADataLoader = data_prep.TruthfulQADataLoader
    
    loader = TruthfulQADataLoader()
    
    # Prefer the canonical TruthfulQA CSV in the repo
    csv_path_abs = (BASE_DIR.parent.parent / "datasets" / "TruthfulQA" / "data" / "v1" / "TruthfulQA.csv").resolve()
    qa_pairs = loader.load_from_csv(str(csv_path_abs), sample_limit=None)
    
    # Fallback to JSON export if CSV missing
    if not qa_pairs:
        json_path = "../qwen3_truthfulqa_evaluation_20250806_222711.json"
        qa_pairs = loader.load_from_json(json_path)
    
    # Supplement with synthetic to reach sample_limit
    if len(qa_pairs) < sample_limit:
        additional = sample_limit - len(qa_pairs)
        print(f"Supplementing with {additional} synthetic samples to reach {sample_limit}")
        qa_pairs.extend(loader.load_synthetic_data(sample_limit=additional))
    else:
        # Trim to sample_limit
        qa_pairs = qa_pairs[:sample_limit]
    
    # Prepare prompts
    prompts = loader.prepare_prompts(qa_pairs)
    
    # Split data
    train_prompts, test_prompts = loader.split_train_test(prompts, test_ratio=0.2)
    
    # Get statistics
    stats = loader.get_statistics(prompts)
    
    print(f"\n✓ Prepared {len(prompts)} prompts")
    print(f"  Train: {len(train_prompts)}, Test: {len(test_prompts)}")
    print(f"  Categories: {list(stats['categories'].keys())}")
    
    return train_prompts, test_prompts


def run_phase_2(prompts, model_name, limit=None):
    """Phase 2: Inference & Extraction for a specific model."""
    print("\n" + "="*80)
    print(f"PHASE 2: INFERENCE & EXTRACTION - {model_name}")
    print("="*80)
    
    import importlib.util
    BASE_DIR = Path(__file__).parent.resolve()
    spec = importlib.util.spec_from_file_location("inference_ext", str(BASE_DIR / "2_inference_extraction.py"))
    inference_ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference_ext)
    ModelInternalsExtractor = inference_ext.ModelInternalsExtractor
    
    # Limit prompts if specified
    if limit:
        prompts = prompts[:limit]
        print(f"Limited to {limit} prompts for testing")
    
    # Initialize extractor
    print(f"\nInitializing model: {model_name}...")
    extractor = ModelInternalsExtractor(
        model_name=model_name,
        use_transformerlens=True
    )
    
    # Extract internals
    print(f"Extracting internals for {len(prompts)} prompts...")
    
    # Create model-specific directory
    save_dir = str(BASE_DIR / f"extracted_internals_{model_name.replace('/', '_')}")
    internals = extractor.extract_batch(
        prompts,
        save_dir=save_dir
    )
    
    print(f"✓ Extracted internals for {len(internals)} samples")
    
    return internals, save_dir


def run_phase_3(model_name, internals_dir):
    """Phase 3: Spectral Analysis for a specific model."""
    print("\n" + "="*80)
    print(f"PHASE 3: SPECTRAL ANALYSIS - {model_name}")
    print("="*80)
    
    import importlib.util
    BASE_DIR = Path(__file__).parent.resolve()
    spec = importlib.util.spec_from_file_location("spectral_analysis", str(BASE_DIR / "3_spectral_analysis.py"))
    spectral_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spectral_analysis)
    process_saved_internals = spectral_analysis.process_saved_internals
    
    output_dir = str(BASE_DIR / f"spectral_metrics_{model_name.replace('/', '_')}")
    results = process_saved_internals(
        internals_dir=internals_dir,
        output_dir=output_dir
    )
    
    print(f"✓ Computed spectral metrics for {len(results)} samples")
    
    # Show sample metrics
    if results:
        sample = results[0]
        if 'spectral_analysis' in sample:
            agg = sample['spectral_analysis'].get('aggregate_metrics', {})
            print(f"\nSample metrics:")
            print(f"  Mean α: {agg.get('mean_alpha', 0):.3f}")
            print(f"  Risk score: {agg.get('mean_risk_score', 0):.3f}")
            print(f"  Total spikes: {agg.get('total_spikes', 0)}")
    
    return results, output_dir


def run_phase_4(model_name, spectral_dir):
    """Phase 4: Statistical Analysis for a specific model."""
    print("\n" + "="*80)
    print(f"PHASE 4: STATISTICAL ANALYSIS - {model_name}")
    print("="*80)
    
    import importlib.util
    import shutil
    BASE_DIR = Path(__file__).parent.resolve()
    spec = importlib.util.spec_from_file_location("statistical_analysis", str(BASE_DIR / "4_statistical_analysis.py"))
    statistical_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(statistical_analysis)
    
    # Use the correct class name
    HallucinationCorrelator = statistical_analysis.HallucinationCorrelator
    
    # First, copy the spectral metrics to the expected location
    expected_dir = BASE_DIR / "spectral_metrics"
    if expected_dir.exists():
        shutil.rmtree(expected_dir)
    shutil.copytree(spectral_dir, expected_dir)
    
    # Initialize correlator
    correlator = HallucinationCorrelator()
    
    # Load data
    try:
        data = correlator.load_data()
        print(f"Loaded {len(data)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Compute correlations
    correlations = correlator.compute_correlations()
    
    # Group comparison
    comparisons = correlator.group_comparison()
    
    # Train classifier
    classifier_results = correlator.train_classifier()
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    # Format results
    correlations_list = []
    if not correlations.empty:
        for _, row in correlations.iterrows():
            correlations_list.append({
                'feature': row['feature'],
                'pearson_r': float(row['pearson_r']),
                'pearson_p': float(row['pearson_p']),
                'spearman_r': float(row['spearman_r']),
                'spearman_p': float(row['spearman_p']),
                'pointbiserial_r': float(row['pointbiserial_r']),
                'pointbiserial_p': float(row['pointbiserial_p']),
                'significant': bool(row['significant'])
            })
    
    # Save results
    output_dir = BASE_DIR / f"analysis_results_{model_name.replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_name': model_name,
        'correlations': correlations_list,
        'group_comparisons': convert_numpy_types(comparisons),
        'classification': convert_numpy_types({k: v for k, v in classifier_results.items() if k not in ['model', 'scaler']})
    }
    
    with open(output_dir / "statistical_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up temporary directory
    shutil.rmtree(expected_dir)
    
    return results


def generate_comparison_report(all_model_results):
    """Generate comparison report across all models."""
    print("\n" + "="*80)
    print("MULTI-MODEL COMPARISON REPORT")
    print("="*80)
    
    BASE_DIR = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = (BASE_DIR / "reports" / f"multi_model_report_{timestamp}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics across models
    print("\n📊 MODEL COMPARISON")
    print("-" * 60)
    print(f"{'Model':<30} {'Spike Corr':<12} {'Alpha Corr':<12} {'ROC AUC':<10}")
    print("-" * 60)
    
    comparison_data = []
    
    for model_name, results in all_model_results.items():
        if results and 'statistical_results' in results:
            stats = results['statistical_results']
            
            # Find spike correlation
            spike_corr = None
            alpha_corr = None
            for corr in stats.get('correlations', []):
                if corr['feature'] == 'total_spikes':
                    spike_corr = corr['pearson_r']
                elif corr['feature'] == 'mean_alpha':
                    alpha_corr = corr['pearson_r']
            
            # Get classification performance
            roc_auc = stats.get('classification', {}).get('roc_auc', 0)
            
            print(f"{model_name:<30} {spike_corr or 0:<12.3f} {alpha_corr or 0:<12.3f} {roc_auc:<10.3f}")
            
            comparison_data.append({
                'model': model_name,
                'spike_correlation': spike_corr,
                'alpha_correlation': alpha_corr,
                'roc_auc': roc_auc,
                'accuracy': stats.get('classification', {}).get('accuracy', 0)
            })
    
    # Save full report
    report = {
        'timestamp': timestamp,
        'experiment': 'Multi-Model TruthfulQA Spectral Analysis',
        'models_tested': list(all_model_results.keys()),
        'comparison': comparison_data,
        'detailed_results': all_model_results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Full report saved to: {report_path}")
    
    # Summary insights
    print("\n🔍 KEY INSIGHTS:")
    if len(comparison_data) > 1:
        spike_corrs = [d['spike_correlation'] for d in comparison_data if d['spike_correlation']]
        if spike_corrs:
            avg_spike_corr = sum(spike_corrs) / len(spike_corrs)
            print(f"  • Average spike correlation across models: {avg_spike_corr:.3f}")
            print(f"  • Correlation range: {min(spike_corrs):.3f} to {max(spike_corrs):.3f}")
    
    return report


def run_experiment_for_model(model_name, prompts, sample_limit=30):
    """Run complete experiment for a single model."""
    results = {'model_name': model_name}
    
    try:
        # Phase 2: Inference & Extraction
        internals, internals_dir = run_phase_2(prompts, model_name, limit=sample_limit)
        results['internals_count'] = len(internals)
        
        # Phase 3: Spectral Analysis
        spectral_results, spectral_dir = run_phase_3(model_name, internals_dir)
        results['spectral_count'] = len(spectral_results)
        
        # Phase 4: Statistical Analysis
        statistical_results = run_phase_4(model_name, spectral_dir)
        results['statistical_results'] = statistical_results
        
        print(f"\n✅ Completed analysis for {model_name}")
        
    except Exception as e:
        print(f"\n❌ Error with {model_name}: {e}")
        traceback.print_exc()
        results['error'] = str(e)
    
    return results


def main():
    """Main function to run multi-model experiment."""
    
    print("\n" + "🔬"*40)
    print("   MULTI-MODEL TRUTHFULQA SPECTRAL HALLUCINATION DETECTION")
    print("🔬"*40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    print("\n✅ All dependencies satisfied")
    
    # Models to test - current state-of-the-art open source models (2025)
    MODELS_TO_TEST = [
        # Small models for quick testing

        # ~7B class models (closest available sizes)
        "Qwen/Qwen3-8B",            # Qwen 3 8B (no official 7B; 8B is the closest)
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # DeepSeek R1 distilled on Qwen 7B
        "meta-llama/Llama-3.2-3B",  # Llama 3.2 3B - smaller variant

        # Mid-size models (14B-20B)
        "Qwen/Qwen3-14B",           # Qwen 3 14B
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # DeepSeek R1 distilled on Llama (8B)

        # Larger models (32B class)
        "Qwen/Qwen3-32B",           # Qwen 3 32B
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # DeepSeek R1 distilled 32B

        # MoE models under 30B active params
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Mixtral 8x7B (~13B active)

    ]
    
    # Configuration
    config = {
        'sample_limit_per_model': 20,  # Further reduced due to many models
        'total_sample_pool': 120,      # Total samples to prepare
    }
    
    print("\n📋 Configuration:")
    print(f"  • Models to test: {len(MODELS_TO_TEST)}")
    print(f"  • Samples per model: {config['sample_limit_per_model']}")
    for model in MODELS_TO_TEST:
        print(f"    - {model}")
    
    # Run experiment
    all_model_results = {}
    
    try:
        # Phase 1: Data Preparation (shared across models)
        train_prompts, test_prompts = run_phase_1(sample_limit=config['total_sample_pool'])
        combined_prompts = train_prompts + test_prompts
        
        # Test each model
        for model_name in MODELS_TO_TEST:
            print(f"\n{'='*80}")
            print(f"TESTING MODEL: {model_name}")
            print(f"{'='*80}")
            
            results = run_experiment_for_model(
                model_name, 
                combined_prompts,
                sample_limit=config['sample_limit_per_model']
            )
            
            all_model_results[model_name] = results
            
            # Clear GPU cache between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Generate comparison report
        report = generate_comparison_report(all_model_results)
        
        print("\n" + "="*80)
        print("✅ MULTI-MODEL EXPERIMENT COMPLETE!")
        print("="*80)
        
        print("\n📊 Next Steps:")
        print("1. Review the multi-model comparison report")
        print("2. Check if spectral signatures generalize across model architectures")
        print("3. Identify which models show strongest correlations")
        print("4. Consider testing on larger models if patterns hold")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    return report


if __name__ == "__main__":
    # Check if we have CUDA
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU (will be slower)")
    
    # Run experiment
    report = main()
