"""Test script for RQ2 Data Loader"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.rq2_data_loader import RQ2DataLoader

def main():
    print("="*80)
    print("Testing RQ2 Data Loader")
    print("="*80)
    
    # Initialize loader
    print("\n[1] Initializing loader...")
    loader = RQ2DataLoader()
    print(f"    Results dir: {loader.results_base_dir}")
    print(f"    Exists: {loader.results_base_dir.exists()}")
    
    # Find experiments
    print("\n[2] Finding experiments...")
    experiments = loader.find_all_experiments()
    print(f"    Found: {len(experiments)} experiments")
    if experiments:
        print(f"    Most recent 5:")
        for exp in experiments[:5]:
            print(f"      - {exp}")
    
    # Test loading one experiment
    if experiments:
        exp_id = experiments[0]
        print(f"\n[3] Testing load of experiment: {exp_id}")
        
        # Load metadata first
        print("    Loading metadata...")
        metadata = loader.load_experiment_metadata(exp_id)
        print(f"    Metadata shape: {metadata.shape}")
        print(f"    Metadata columns: {metadata.columns.tolist()}")
        
        # Load experiment (limited to 2 files for testing)
        print("    Loading experiment data (max 2 files)...")
        df = loader.load_experiment(exp_id, max_files=2)
        print(f"    Loaded: {len(df)} edges")
        print(f"    Columns: {df.columns.tolist()}")
        
        # Test validation and cleaning
        print("\n[4] Testing validation and cleaning...")
        df_clean = loader.validate_and_clean(df)
        print(f"    After cleaning: {len(df_clean)} edges")
        
        # Show summary
        print("\n[5] Data Summary:")
        summary = loader.get_data_summary(df_clean)
        for key, value in sorted(summary.items()):
            print(f"    {key}: {value}")
        
        # Show sample data
        print("\n[6] Sample Data (first 3 rows):")
        cols_to_show = ['source', 'target', 'classification', 
                       'perplexity', 'aggregate_score', 'is_hallucination']
        available_cols = [c for c in cols_to_show if c in df_clean.columns]
        print(df_clean[available_cols].head(3).to_string())
        
        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80)
        
        return 0
    else:
        print("\n✗ No experiments found to test with")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)