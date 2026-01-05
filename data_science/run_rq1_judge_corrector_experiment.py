#!/usr/bin/env python3
"""
RQ1 Judge/Corrector Experiment Runner

Research Questions:
- RQ1: Can LLM-as-a-judge and LLM-as-a-corrector method be used to increase final CLD accuracy?
- RQ1a: Can LLM-as-a-judge be used to detect hallucinations?  
- RQ1b: Can LLM-as-a-corrector be used to correct hallucinations?

Experiment Design:
1. Load ground truth CLDs
2. Generate relationships with spurious motivations (30% corruption rate)
3. Use LLM-as-a-judge to detect problematic relationships
4. Use LLM-as-a-corrector to fix or remove problematic relationships
5. Compare final accuracy with and without judge/corrector
6. Use Brave Search for real citations to support judgments
"""

import os
import sys
import datetime
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from data_science.modules import multirun_parameter_experiments

def main():
    """Run the RQ1 Judge/Corrector experiment using multirun framework."""
    
    print("="*80)
    print("🧪 RQ1 JUDGE/CORRECTOR EXPERIMENT")
    print("="*80)
    print("Research Question: Can LLM-as-a-judge and LLM-as-a-corrector increase final CLD accuracy?")
    print()
    print("Experiment Design:")
    print("  • Generate relationships from ground truth CLDs")
    print("  • Corrupt 30% with spurious motivations (hallucinations)")
    print("  • Use 3-judge voting system to detect problems")
    print("  • Use LLM corrector to revise/remove problematic edges")
    print("  • Compare accuracy: Baseline vs Judge-only vs Judge+Corrector")
    print("  • Use Brave Search API for real citations")
    print()
    
    # Configuration paths
    PROMPTS_FOLDER = "parameter_tuning_experiments/alternative_prompts"
    CONFIG_FILE = "parameter_tuning_experiments/configs/experiment_judge_corrector_rq1.yaml"
    CLD_FOLDER = "parameter_tuning_experiments/ground_truth_clds_for_experiments"
    
    # Create results folder with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    RESULTS_FOLDER = f"parameter_tuning_experiments/results/rq1_judge_corrector_{timestamp}"
    
    print(f"📁 Results will be saved to: {RESULTS_FOLDER}")
    print(f"📋 Using config: {CONFIG_FILE}")
    print(f"📊 Using CLDs from: {CLD_FOLDER}")
    print()
    
    # Verify required files exist
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ ERROR: Config file not found: {CONFIG_FILE}")
        return 1
        
    if not os.path.exists(CLD_FOLDER):
        print(f"❌ ERROR: CLD folder not found: {CLD_FOLDER}")
        return 1
        
    # Check for Brave Search API key
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not brave_key:
        print("⚠️  WARNING: BRAVE_SEARCH_API_KEY not found in environment")
        print("   Citation search will be limited without Brave Search API")
    else:
        print(f"✅ Brave Search API key found: {brave_key[:8]}...{brave_key[-4:]}")
    
    print()
    print("🚀 Starting multirun experiment...")
    print("⏱️  This may take several hours depending on the number of runs")
    print()
    
    try:
        # Run the multirun experiment
        multirun_parameter_experiments(
            PROMPTS_FOLDER=PROMPTS_FOLDER,
            CONFIG_FILE=CONFIG_FILE,
            CLD_FOLDER=CLD_FOLDER,
            RESULTS_FOLDER=RESULTS_FOLDER
        )
        
        print()
        print("="*80)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Results saved to: {RESULTS_FOLDER}")
        print()
        print("📊 Next steps:")
        print("   1. Check the Excel files for detailed edge-by-edge analysis")
        print("   2. Look for parameter combo mapping CSV for aggregation")
        print("   3. Compare accuracy metrics across conditions:")
        print("      • Baseline (no judge/corrector)")
        print("      • Judge only (detection)")  
        print("      • Judge + Corrector (detection + correction)")
        print()
        print("🔍 Key metrics to analyze:")
        print("   • Precision/Recall/F1 for edge detection")
        print("   • True Positive rate (correctly identified real relationships)")
        print("   • False Positive rate (incorrectly flagged real relationships)")
        print("   • Correction success rate (spurious edges fixed)")
        print("   • Cost analysis (tokens/time per component)")
        
    except KeyboardInterrupt:
        print()
        print("⏹️  Experiment interrupted by user")
        return 1
    except Exception as e:
        print()
        print(f"❌ ERROR: Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)