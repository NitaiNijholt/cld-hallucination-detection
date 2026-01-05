#!/usr/bin/env python3
"""
Run specific prompts only for targeted judging.

Usage:
    python run_specific_prompts.py <session_id> <excel_file> <output_dir> --prompts baseline cot --judge-type citation
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env.dev")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_rq1a_ground_truth_judge import discover_prompts, judge_with_prompt, JUDGE_CONFIG
from neo4j_session_cloner import SessionCloner


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run specific prompts only')
    parser.add_argument('session_id', help='Session ID to judge')
    parser.add_argument('excel_file', help='Path to Excel file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--prompts', nargs='+', required=True,
                        help='List of prompts to run (e.g., baseline cot mechanistic)')
    parser.add_argument('--judge-type', type=str, default='correctness', choices=['correctness', 'citation'],
                        help='Type of judging: correctness or citation (default: correctness)')
    parser.add_argument('--embedding-device', type=str, default='cuda',
                        help='Device for embeddings: cpu, cuda (default), cuda:0, etc.')
    args = parser.parse_args()
    
    # Auto-discover ALL available prompts
    all_prompts = discover_prompts(prompt_type=args.judge_type)
    
    # Filter to only requested prompts
    selected_prompts = {}
    for prompt_name in args.prompts:
        if prompt_name in all_prompts:
            selected_prompts[prompt_name] = all_prompts[prompt_name]
        else:
            logging.error(f"Prompt '{prompt_name}' not found. Available: {list(all_prompts.keys())}")
            sys.exit(1)
    
    # Map judge_type to judge_approach
    judge_approach_map = {
        'correctness': 'correctness',
        'citation': 'per_citation_aggregate'
    }
    judge_approach = judge_approach_map[args.judge_type]
    
    logging.info("=" * 80)
    logging.info("RUN SPECIFIC PROMPTS")
    logging.info("=" * 80)
    logging.info(f"Session ID: {args.session_id}")
    logging.info(f"Excel file: {args.excel_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Judge type: {args.judge_type}")
    logging.info(f"Prompts to run: {list(selected_prompts.keys())}")
    logging.info(f"Judge model: {JUDGE_CONFIG['model']}")
    logging.info(f"Web search: {'ENABLED' if args.judge_type == 'citation' else 'DISABLED'}")
    logging.info(f"Embedding device: {args.embedding_device}")
    logging.info("")
    
    # Initialize Neo4j session cloner
    logging.info("Initializing Neo4j session cloner...")
    cloner = SessionCloner()
    logging.info("✅ Session cloner ready\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Judge with selected prompts
    results = []
    for prompt_name, prompt_info in selected_prompts.items():
        logging.info(f"\n{'='*80}")
        logging.info(f"PROMPT {len(results)+1}/{len(selected_prompts)}: {prompt_info['description'].upper()}")
        logging.info(f"{'='*80}\n")
        
        result = judge_with_prompt(
            session_id=args.session_id,
            prompt_name=prompt_name,
            prompt_info=prompt_info,
            excel_file=args.excel_file,
            output_dir=output_dir,
            cloner=cloner,
            judge_approach=judge_approach,
            judge_enable_web_search=(args.judge_type == 'citation'),
            embedding_device=args.embedding_device
        )
        results.append(result)
    
    logging.info("\n" + "=" * 80)
    logging.info(f"ALL {len(selected_prompts)} PROMPTS COMPLETE")
    logging.info("=" * 80)
    
    for i, result in enumerate(results, 1):
        logging.info(f"{i}. {result['prompt_description']}")
        logging.info(f"   File: {Path(result['result_file']).name}")
    
    logging.info("\n✅ DONE!")


if __name__ == "__main__":
    main()


