"""
Main Experiment Runner: TruthfulQA Spectral Analysis

This script orchestrates the complete experiment pipeline:
1. Data preparation
2. Model inference & internal state extraction  
3. Spectral analysis
4. Statistical correlation
5. Visualization and reporting
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')


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
    # Dataset path relative to repo root: data_science/datasets/TruthfulQA/data/v1/TruthfulQA.csv
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


def run_phase_2(prompts, limit=None):
    """Phase 2: Inference & Extraction."""
    print("\n" + "="*80)
    print("PHASE 2: INFERENCE & INTERNAL STATE EXTRACTION")
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
    print("\nInitializing model...")
    extractor = ModelInternalsExtractor(
        model_name="gpt2",  # Use small model for demo
        use_transformerlens=True
    )
    
    # Extract internals
    print(f"Extracting internals for {len(prompts)} prompts...")
    internals = extractor.extract_batch(
        prompts,
        save_dir=str(BASE_DIR / "extracted_internals")
    )
    
    print(f"✓ Extracted internals for {len(internals)} samples")
    
    return internals


def run_phase_3():
    """Phase 3: Spectral Analysis."""
    print("\n" + "="*80)
    print("PHASE 3: SPECTRAL ANALYSIS")
    print("="*80)
    
    import importlib.util
    BASE_DIR = Path(__file__).parent.resolve()
    spec = importlib.util.spec_from_file_location("spectral_analysis", str(BASE_DIR / "3_spectral_analysis.py"))
    spectral_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spectral_analysis)
    process_saved_internals = spectral_analysis.process_saved_internals
    
    results = process_saved_internals(
        internals_dir=str(BASE_DIR / "extracted_internals"),
        output_dir=str(BASE_DIR / "spectral_metrics")
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
    
    return results


def run_phase_4():
    """Phase 4: Statistical Analysis."""
    print("\n" + "="*80)
    print("PHASE 4: STATISTICAL ANALYSIS & CORRELATION")
    print("="*80)
    
    import importlib.util
    BASE_DIR = Path(__file__).parent.resolve()
    spec = importlib.util.spec_from_file_location("statistical_analysis", str(BASE_DIR / "4_statistical_analysis.py"))
    statistical_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(statistical_analysis)
    analyze = statistical_analysis.main
    
    results = analyze()
    
    return results


def generate_report(all_results):
    """Generate final experiment report."""
    print("\n" + "="*80)
    print("FINAL EXPERIMENT REPORT")
    print("="*80)
    
    BASE_DIR = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = (BASE_DIR / "reports" / f"experiment_report_{timestamp}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load statistical results
    stats_file = BASE_DIR / "analysis_results" / "statistical_results.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    # Summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("-" * 40)
    
    # Key findings
    if 'group_comparisons' in stats:
        print("\n✅ Key Findings:")
        for metric, comparison in stats['group_comparisons'].items():
            if comparison.get('significant'):
                print(f"  • {metric}: Significant difference (p={comparison['t_pvalue']:.3f})")
                print(f"    - Truthful: {comparison['truthful_mean']:.3f}")
                print(f"    - Hallucinated: {comparison['hallucinated_mean']:.3f}")
    
    # Classification performance
    if 'classification' in stats:
        clf = stats['classification']
        print(f"\n🎯 Classification Performance:")
        print(f"  • Accuracy: {clf.get('accuracy', 0):.1%}")
        print(f"  • ROC AUC: {clf.get('roc_auc', 0):.3f}")
    
    # Save report
    report = {
        'timestamp': timestamp,
        'experiment': 'TruthfulQA Spectral Analysis',
        'hypothesis': 'Spectral signatures of attention matrices correlate with hallucinations',
        'results': stats,
        'conclusion': 'See statistical analysis for detailed results'
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Report saved to: {report_path}")
    
    return report


def main():
    """Main experiment runner."""
    
    print("\n" + "🔬"*40)
    print("   TRUTHFULQA SPECTRAL HALLUCINATION DETECTION EXPERIMENT")
    print("🔬"*40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    print("\n✅ All dependencies satisfied")
    
    # Configuration
    config = {
        'use_synthetic_data': False,  # Use real TruthfulQA data
        'sample_limit': 120,  # Large sample size for strong statistical power
        'model_name': 'gpt2',  # Model to analyze
    }
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # Run experiment phases
    try:
        # Phase 1: Data Preparation
        train_prompts, test_prompts = run_phase_1(sample_limit=config['sample_limit'])
        
        # Phase 2: Inference & Extraction
        # Use combined train+test to reach full sample_limit
        combined = train_prompts + test_prompts
        if config['sample_limit']:
            test_subset = combined[:config['sample_limit']]
        else:
            test_subset = combined
        internals = run_phase_2(test_subset)
        
        # Phase 3: Spectral Analysis
        spectral_results = run_phase_3()
        
        # Phase 4: Statistical Analysis
        statistical_results = run_phase_4()
        
        # Generate report
        report = generate_report({
            'config': config,
            'num_samples': len(test_subset),
            'spectral_results': len(spectral_results),
            'statistical_results': statistical_results
        })
        
        print("\n" + "="*80)
        print("✅ EXPERIMENT COMPLETE!")
        print("="*80)
        
        print("\n📊 Next Steps:")
        print("1. Review the statistical results in experiment/analysis_results/")
        print("2. Examine individual spectral metrics in experiment/spectral_metrics/")
        print("3. Scale up with more samples for stronger statistical power")
        print("4. Test with larger models (e.g., Llama, Phi-2) for better results")
        print("5. Implement real-time detection using the trained classifier")
        
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