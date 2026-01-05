#!/usr/bin/env python3
"""
Convert TruthfulQA dataset to judge evaluation format.

Creates balanced dataset of truthful and hallucinated answers by:
- Using correct_answers as truthful responses
- Using incorrect_answers as hallucinated responses
"""

import json
import random
from pathlib import Path
from datetime import datetime
import argparse

def convert_truthqa_dataset(
    input_path: str,
    output_path: str,
    max_samples_per_type: int = None,
    seed: int = 42
):
    """
    Convert TruthfulQA to judge evaluation format.
    
    Args:
        input_path: Path to TruthfulQA dataset
        output_path: Path to save converted dataset
        max_samples_per_type: Max samples per type (truthful/hallucinated)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"Loading TruthfulQA from: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"Loaded {len(samples)} questions")
    
    # Create Q&A pairs
    generation_results = []
    
    for idx, sample in enumerate(samples):
        question = sample['question']
        correct_answers = sample.get('correct_answers', [])
        incorrect_answers = sample.get('incorrect_answers', [])
        
        # Add truthful example (use first correct answer)
        if correct_answers:
            generation_results.append({
                'question_num': len(generation_results) + 1,
                'question': question,
                'response': correct_answers[0],  # Use first correct answer
                'correct_answers': correct_answers,
                'truthfulness': 1,
                'is_truthful': True,
                'timestamp': datetime.now().isoformat()
            })
        
        # Add hallucinated example (use first incorrect answer)
        if incorrect_answers:
            generation_results.append({
                'question_num': len(generation_results) + 1,
                'question': question,
                'response': incorrect_answers[0],  # Use first incorrect answer
                'correct_answers': correct_answers,
                'truthfulness': 0,
                'is_truthful': False,
                'timestamp': datetime.now().isoformat()
            })
    
    # Shuffle to mix truthful and hallucinated
    random.shuffle(generation_results)
    
    # Limit if requested
    if max_samples_per_type:
        truthful = [r for r in generation_results if r['is_truthful']]
        hallucinated = [r for r in generation_results if not r['is_truthful']]
        
        truthful = truthful[:max_samples_per_type]
        hallucinated = hallucinated[:max_samples_per_type]
        
        generation_results = truthful + hallucinated
        random.shuffle(generation_results)
        
        print(f"Limited to {max_samples_per_type} samples per type")
    
    # Re-number
    for idx, result in enumerate(generation_results, 1):
        result['question_num'] = idx
    
    # Count distribution
    truthful_count = sum(1 for r in generation_results if r['is_truthful'])
    hallucinated_count = len(generation_results) - truthful_count
    
    # Create output
    output = {
        'model': 'truthfulqa_reference_answers',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'total_questions': len(generation_results),
            'truthful': truthful_count,
            'hallucinated': hallucinated_count
        },
        'generation_results': generation_results
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved {len(generation_results)} Q&A pairs to: {output_path}")
    print(f"  Truthful: {truthful_count} ({truthful_count/len(generation_results)*100:.1f}%)")
    print(f"  Hallucinated: {hallucinated_count} ({hallucinated_count/len(generation_results)*100:.1f}%)")
    print(f"{'='*60}")
    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TruthfulQA to judge format')
    parser.add_argument(
        '--input',
        type=str,
        default='../spectral_hallucination_detection/truthfulqa_full_dataset.json',
        help='Input TruthfulQA dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='truthfulqa_judge_dataset_full.json',
        help='Output path for converted dataset'
    )
    parser.add_argument(
        '--max-per-type',
        type=int,
        default=None,
        help='Maximum samples per type (truthful/hallucinated)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    convert_truthqa_dataset(
        input_path=args.input,
        output_path=args.output,
        max_samples_per_type=args.max_per_type,
        seed=args.seed
    )
