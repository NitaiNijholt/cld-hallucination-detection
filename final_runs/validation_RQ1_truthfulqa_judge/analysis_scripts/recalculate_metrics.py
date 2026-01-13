#!/usr/bin/env python3
"""
Recalculate TruthQA metrics using judge_score instead of judge_verdict
Generates truthqa_verification_table.tex for thesis
"""

import json
import sys
import argparse
from pathlib import Path


def generate_latex_table(metrics: dict, n_samples: int, output_path: Path) -> None:
    """Generate LaTeX table for thesis."""
    cm = metrics['confusion_matrix']
    
    latex = r"""\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{threeparttable}
\caption{TruthfulQA Judge Verification (n=""" + str(n_samples) + r""")}
\label{tab:truthqa_verification}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \multicolumn{2}{c}{\textbf{Value}} \\
\midrule
Accuracy & \multicolumn{2}{c}{""" + f"{metrics['accuracy']:.3f}" + r"""} \\
Precision & \multicolumn{2}{c}{""" + f"{metrics['precision']:.3f}" + r"""} \\
Recall & \multicolumn{2}{c}{""" + f"{metrics['recall']:.3f}" + r"""} \\
F1 Score & \multicolumn{2}{c}{""" + f"{metrics['f1_score']:.3f}" + r"""} \\
\midrule
 & \textbf{Pred: Halluc.} & \textbf{Pred: Truthful} \\
\textbf{Actual: Halluc.} & """ + str(cm['tp']) + r""" (TP) & """ + str(cm['fn']) + r""" (FN) \\
\textbf{Actual: Truthful} & """ + str(cm['fp']) + r""" (FP) & """ + str(cm['tn']) + r""" (TN) \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Note.} Judge: GPT-4.1, temp=0.0, seed=42, baseline correctness prompt. Dataset: 1634 samples (817 hallucinated, 817 truthful; 50\% base rate). Classification threshold: judge\_score $<$ 0.5 $\Rightarrow$ Hallucinated.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX table: {output_path}")


def recalculate_metrics(json_path, output_path=None):
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
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }
    
    # Generate LaTeX table if output path specified
    if output_path:
        generate_latex_table(metrics, n_samples, Path(output_path))
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recalculate TruthfulQA metrics")
    parser.add_argument("json_path", nargs="?", 
                       default="Data/truthqa_judge_baseline_results_20251115_204150.json",
                       help="Path to results JSON")
    parser.add_argument("--output", type=str, 
                       help="Output path for LaTeX table")
    args = parser.parse_args()
    
    recalculate_metrics(args.json_path, args.output)
