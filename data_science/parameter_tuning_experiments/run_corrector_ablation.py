#!/usr/bin/env python3
"""
Corrector Prompt Ablation Study Runner

Runs corrector experiments with all prompt variants to test:
- Baseline (examples only)
- CoT (examples + reasoning)
- Mechanistic (examples + reasoning + Bradford Hill)

Usage:
    python run_corrector_ablation.py --clds social_norms --runs 3 --yes
"""

import subprocess
import sys
from pathlib import Path

PROMPT_VARIANTS = ['baseline', 'cot', 'mechanistic']

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run corrector ablation study')
    parser.add_argument('--rq1a-folder', type=str, 
                       default='RQ1a_gt_lit_correctness_test',
                       help='RQ1a folder to use for judged sessions')
    parser.add_argument('--clds', nargs='+', 
                       default=['social_norms', 'depressive', 'emergency_department'],
                       help='CLDs to test')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per CLD')
    parser.add_argument('--judge-variant', type=str, default='baseline',
                       help='Judge variant used in RQ1a')
    parser.add_argument('--yes', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CORRECTOR PROMPT ABLATION STUDY")
    print("="*80)
    print(f"RQ1a Folder: {args.rq1a_folder}")
    print(f"CLDs: {', '.join(args.clds)}")
    print(f"Runs per CLD: {args.runs}")
    print(f"Prompt Variants: {', '.join(PROMPT_VARIANTS)}")
    print(f"\nTotal experiments: {len(args.clds)} CLDs × {args.runs} runs × {len(PROMPT_VARIANTS)} prompts = {len(args.clds) * args.runs * len(PROMPT_VARIANTS)}")
    print("="*80)
    
    if not args.yes:
        response = input("\nProceed with ablation study? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return 1
    
    # Run each prompt variant
    failed_variants = []
    for variant in PROMPT_VARIANTS:
        print(f"\n{'='*80}")
        print(f"RUNNING VARIANT: {variant.upper()}")
        print(f"{'='*80}\n")
        
        cmd = [
            'python3',
            'data_science/parameter_tuning_experiments/run_rq1b_all_clds.py',
            '--rq1a-folder', args.rq1a_folder,
            '--clds', *args.clds,
            '--runs', str(args.runs),
            '--judge-variant', args.judge_variant,
            '--corrector-prompt', variant,
            '--yes'
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n❌ FAILED: {variant}")
            failed_variants.append(variant)
        else:
            print(f"\n✅ COMPLETED: {variant}")
    
    print(f"\n{'='*80}")
    if failed_variants:
        print(f"⚠️  ABLATION STUDY COMPLETED WITH ERRORS")
        print(f"{'='*80}")
        print(f"\nFailed variants: {', '.join(failed_variants)}")
        return 1
    else:
        print("✅ ABLATION STUDY COMPLETE")
        print(f"{'='*80}")
        print("\nResults saved to:")
        for variant in PROMPT_VARIANTS:
            folder = f"final_runs/RQ1b_corrector_experiment_{args.rq1a_folder.replace('RQ1a_', '')}_{variant}_{args.judge_variant}_judge/"
            print(f"  - {folder}rq1b_all_results.xlsx")
        
        print("\nNext steps:")
        print("  1. Run analysis: python data_science/parameter_tuning_experiments/analyze_corrector_ablation.py")
        print("  2. Compare results across variants")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

