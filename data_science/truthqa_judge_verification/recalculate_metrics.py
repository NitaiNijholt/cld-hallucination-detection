#!/usr/bin/env python3
"""
Recalculate TruthQA metrics using judge_score instead of judge_verdict
"""

import json
import sys

def recalculate_metrics(json_path):
    """Recalculate metrics using score instead of verdict"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize counters
    tp = 0  # True Positive: Actually hallucinated, predicted hallucinated (score < 0.5)
    fp = 0  # False Positive: Actually truthful, predicted hallucinated (score < 0.5)
    tn = 0  # True Negative: Actually truthful, predicted truthful (score >= 0.5)
    fn = 0  # False Negative: Actually hallucinated, predicted truthful (score >= 0.5)
    
    total_correct = 0
    
    for result in data['detailed_results']:
        is_truthful_actual = result['is_truthful_actual']
        judge_score = result['judge_score']
        
        # Predict based on score
        # score >= 0.5 means predicted TRUTHFUL (CORRECT)
        # score < 0.5 means predicted HALLUCINATED (INCORRECT)
        predicted_truthful = judge_score >= 0.5
        
        if is_truthful_actual:
            # Ground truth: TRUTHFUL
            if predicted_truthful:
                tn += 1  # Correctly predicted truthful
                total_correct += 1
            else:
                fp += 1  # Incorrectly predicted hallucinated
        else:
            # Ground truth: HALLUCINATED
            if predicted_truthful:
                fn += 1  # Incorrectly predicted truthful (missed hallucination)
            else:
                tp += 1  # Correctly predicted hallucinated
                total_correct += 1
    
    n_samples = len(data['detailed_results'])
    n_hallucinated = data['metrics']['n_hallucinated']
    n_truthful = data['metrics']['n_truthful']
    
    # Calculate metrics
    accuracy = total_correct / n_samples if n_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("=" * 80)
    print("RECALCULATED METRICS (using judge_score instead of judge_verdict)")
    print("=" * 80)
    print(f"\nDataset:")
    print(f"  Total samples: {n_samples}")
    print(f"  Hallucinated: {n_hallucinated} ({100*n_hallucinated/n_samples:.1f}%)")
    print(f"  Truthful: {n_truthful} ({100*n_truthful/n_samples:.1f}%)")
    
    print(f"\nConfusion Matrix (using score threshold = 0.5):")
    print(f"  TP (caught hallucinations): {tp}")
    print(f"  FP (false alarms): {fp}")
    print(f"  TN (correct truthful): {tn}")
    print(f"  FN (missed hallucinations): {fn}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {accuracy:.1%} ({total_correct}/{n_samples})")
    print(f"  Precision: {precision:.3f} ({tp}/{tp + fp} of flagged items were actual hallucinations)")
    print(f"  Recall: {recall:.1%} ({tp}/{tp + fn} of hallucinations caught)")
    print(f"  F1-Score: {f1_score:.3f}")
    print(f"  Specificity: {specificity:.1%} ({tn}/{tn + fp} of truthful answers correctly identified)")
    
    print(f"\nComparison with Original Results:")
    print(f"  Original Accuracy (using verdict): {data['metrics']['accuracy']:.1%}")
    print(f"  New Accuracy (using score): {accuracy:.1%}")
    print(f"  Improvement: {(accuracy - data['metrics']['accuracy']):.1%}")
    
    print("\n" + "=" * 80)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }

if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'truthqa_judge_verification_baseline/truthqa_judge_baseline_results_20251115_163550.json'
    recalculate_metrics(json_path)
