"""
Download and prepare TruthfulQA dataset from HuggingFace.

The proper TruthfulQA dataset contains questions with both:
- Correct (truthful) reference answers
- Incorrect (false/misleading) reference answers

We'll use a model to generate answers and label them as truthful/hallucinated
based on semantic similarity to correct vs. incorrect references.
"""

import json
from datasets import load_dataset
from pathlib import Path
import argparse

def download_truthfulqa(
    output_path: str = "truthfulqa_generation_dataset.json",
    split: str = "validation",
    max_samples: int = None
):
    """
    Download TruthfulQA dataset from HuggingFace.
    
    Args:
        output_path: Path to save dataset
        split: Dataset split (validation recommended)
        max_samples: Maximum samples (None = all)
    """
    print("Downloading TruthfulQA from HuggingFace...")
    
    # Load dataset
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    
    # Get validation split (main eval set)
    data = dataset[split]
    
    print(f"Loaded {len(data)} questions from TruthfulQA")
    
    # Limit samples if requested
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
        print(f"Limited to {len(data)} samples")
    
    # Convert to our format
    processed_samples = []
    
    for idx, item in enumerate(data):
        sample = {
            'question_id': idx,
            'question': item['question'],
            'category': item.get('category', 'unknown'),
            'correct_answers': item.get('correct_answers', []),
            'incorrect_answers': item.get('incorrect_answers', []),
            'source': item.get('source', 'unknown'),
            # Model will generate answer, then we'll compare to references
            'best_answer': item.get('best_answer', ''),
        }
        processed_samples.append(sample)
    
    # Save dataset
    output = {
        'dataset': 'truthfulqa',
        'split': split,
        'source': 'huggingface:truthfulqa/truthful_qa',
        'num_samples': len(processed_samples),
        'samples': processed_samples
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(processed_samples)} questions to {output_path}")
    
    # Print statistics
    categories = {}
    for s in processed_samples:
        cat = s['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat}: {count}")
    
    print(f"\nSample structure:")
    print(f"  - Questions with correct_answers: {sum(1 for s in processed_samples if s['correct_answers'])}")
    print(f"  - Questions with incorrect_answers: {sum(1 for s in processed_samples if s['incorrect_answers'])}")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='truthfulqa_generation_dataset.json')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--split', default='validation')
    
    args = parser.parse_args()
    
    download_truthfulqa(
        output_path=args.output,
        split=args.split,
        max_samples=args.max_samples
    )
