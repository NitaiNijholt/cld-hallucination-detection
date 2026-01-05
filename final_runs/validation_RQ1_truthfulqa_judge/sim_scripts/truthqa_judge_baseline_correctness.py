#!/usr/bin/env python3
"""
TruthQA Judge Verification Experiment - Using Baseline Prompt Structure

Tests whether LLM judges using the baseline prompt structure can detect 
hallucinations in TruthQA dataset.

This uses prompts_truthqa_baseline.yaml which mirrors prompts_correctness_baseline.yaml
for scientific validity - allows comparison of judge capability on factual vs causal reasoning.

Directory: truthqa_judge_verification/
"""

import json
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv
from jinja2 import Template

# Load environment variables from root directory
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent.parent
env_path = root_dir / '.env'

load_dotenv(env_path)

# Add backend directory to path to import LLM clients
backend_path = Path(__file__).parent.parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

from lm_clients.openai_client_working import OpenAIClient
from lm_clients.claude_client import ClaudeClient

# Add data_science to path to import the working judge models
data_science_path = Path(__file__).parent.parent
sys.path.insert(0, str(data_science_path))

from judge_models import CorrectnessJudgment

# API configuration
def get_api_config(provider: str, model: str) -> tuple:
    """Get API key and URL for the specified provider."""
    if provider.lower() == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        api_url = "https://api.anthropic.com/v1/messages"
    elif provider.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        api_url = "https://api.openai.com/v1/chat/completions"
    elif provider.lower() == "perplexity":
        api_key = os.getenv("PERPLEXITY_API_KEY")
        api_url = "https://api.perplexity.ai/chat/completions"
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    if not api_key:
        raise ValueError(f"API key not found for provider: {provider}. Check your .env file.")
    
    return api_key, api_url

@dataclass
class JudgeResult:
    """Result of a single judge evaluation."""
    question: str
    answer: str
    is_truthful_actual: bool
    judge_verdict: str  # "CORRECT", "PARTIALLY_CORRECT", or "INCORRECT"
    judge_score: float  # 0.0, 0.5, or 1.0
    judge_reasoning: str
    correct_prediction: bool


def load_prompts_from_yaml(yaml_path: str) -> Dict:
    """Load judge prompts from YAML file.
    
    Args:
        yaml_path: Path to YAML file with prompt configuration
        
    Returns:
        Dictionary with sys_prompt and usr_prompt templates
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract the judgeTruthQA prompt
    prompt_config = config['prompts'][0]['prompts']
    
    return {
        'sys_prompt': prompt_config['sys_prompt'],
        'usr_prompt_template': Template(prompt_config['usr_prompt'])
    }


def create_judge_prompt(question: str, answer: str, usr_prompt_template: Template) -> str:
    """Create the judge prompt for a single Q&A pair using YAML template.
    
    Args:
        question: The question being asked
        answer: The answer to evaluate
        usr_prompt_template: Jinja2 template for user prompt
        
    Returns:
        Rendered user prompt
    """
    return usr_prompt_template.render(question=question, answer=answer)


class TruthQAJudgeExperiment:
    """Experiment to test judge hallucination detection on TruthQA using baseline prompt structure."""
    
    def __init__(
        self,
        judge_provider: str = "anthropic",
        judge_model: str = "claude-sonnet-4-5-20250929",
        judge_temperature: float = 0.3,
        max_tokens: int = 250,
        seed: Optional[int] = None,
        yaml_path: str = None
    ):
        """Initialize the experiment.
        
        Args:
            judge_provider: LLM provider for judge (anthropic, openai, etc.)
            judge_model: Specific model to use as judge
            judge_temperature: Temperature for judge responses
            max_tokens: Maximum tokens for judge response
            seed: Random seed for reproducibility
            yaml_path: Path to YAML file with prompts (default: prompts_truthqa_baseline.yaml)
        """
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.max_tokens = max_tokens
        self.seed = seed
        
        # Load prompts from YAML
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent / 'parameter_tuning_experiments' / 'alternative_prompts' / 'prompts_truthqa_baseline.yaml'
        
        self.yaml_path = str(yaml_path)
        prompts = load_prompts_from_yaml(self.yaml_path)
        self.sys_prompt = prompts['sys_prompt']
        self.usr_prompt_template = prompts['usr_prompt_template']
        
        # Get API configuration
        api_key, api_url = get_api_config(judge_provider, judge_model)
        
        # Initialize judge client
        if judge_provider.lower() == "anthropic":
            self.judge_client = ClaudeClient(
                api_key=api_key,
                api_url=api_url,
                model=judge_model,
                dev_mode=False,
                temperature=judge_temperature,
                enable_web_search=False
            )
        else:
            self.judge_client = OpenAIClient(
                api_key=api_key,
                api_url=api_url,
                model=judge_model,
                temperature=judge_temperature,
                dev_mode=False,
                enable_websearch=False
            )
        
        print(f"Initialized TruthQA Judge Experiment (Baseline Prompt Structure)")
        print(f"Prompt YAML: {self.yaml_path}")
        print(f"Judge: {judge_provider}/{judge_model} (temp={judge_temperature}, max_tokens={max_tokens})")
    
    def load_truthqa_data(self, truthqa_path: str) -> List[Dict]:
        """Load TruthQA dataset from JSON file.
        
        Args:
            truthqa_path: Path to TruthQA JSON file
            
        Returns:
            List of Q&A dictionaries with ground truth labels
        """
        print(f"\nLoading TruthQA data from: {truthqa_path}")
        
        with open(truthqa_path, 'r') as f:
            data = json.load(f)
        
        # Extract generation results
        results = data.get('generation_results', data.get('results', []))
        
        # Normalize format
        normalized = []
        for r in results:
            # Clean up response (remove <think> tags if present)
            response = r.get('response', '')
            if '<think>' in response and '</think>' in response:
                parts = response.split('</think>')
                response = parts[-1].strip() if len(parts) > 1 else response
            
            item = {
                'question': r.get('question', ''),
                'answer': response,
                'is_truthful': bool(r.get('truthfulness', r.get('is_truthful', False))),
                'correct_answers': r.get('correct_answers', [])
            }
            normalized.append(item)
        
        print(f"Loaded {len(normalized)} Q&A pairs")
        
        # Count distribution
        truthful = sum(1 for item in normalized if item['is_truthful'])
        hallucinated = len(normalized) - truthful
        print(f"  Truthful: {truthful} ({truthful/len(normalized)*100:.1f}%)")
        print(f"  Hallucinated: {hallucinated} ({hallucinated/len(normalized)*100:.1f}%)")
        
        return normalized
    
    def judge_single_qa(self, question: str, answer: str) -> Tuple[str, float, str]:
        """Have the judge evaluate a single Q&A pair using structured outputs.
        
        Args:
            question: The question
            answer: The answer to evaluate
            
        Returns:
            Tuple of (verdict, score, reasoning)
        """
        user_prompt = create_judge_prompt(question, answer, self.usr_prompt_template)
        
        # Get judge response with structured output (only for OpenAI)
        if self.judge_provider.lower() == "openai":
            response_data = self.judge_client.send_message(
                sys_prompt=self.sys_prompt,
                usr_prompt=user_prompt,
                max_tokens=self.max_tokens,
                seed=self.seed,
                response_model=CorrectnessJudgment
            )
            
            # Extract structured response
            if 'choices' in response_data:
                # Parse the JSON from structured output
                content = response_data['choices'][0]['message']['content']
                judgment = CorrectnessJudgment.model_validate_json(content)
                return judgment.verdict, judgment.score, judgment.reason
        else:
            # Anthropic doesn't support structured outputs in the same way
            response_data = self.judge_client.send_message(
                sys_prompt=self.sys_prompt,
                usr_prompt=user_prompt,
                max_tokens=self.max_tokens
            )
        
        # Fallback text parsing for non-OpenAI or if structured output fails
        if 'content' in response_data and isinstance(response_data['content'], list):
            # Anthropic format
            text_blocks = [block.get('text', '') for block in response_data['content'] if block.get('type') == 'text']
            response = ' '.join(text_blocks) if text_blocks else ''
        elif 'choices' in response_data:
            # OpenAI format
            response = response_data['choices'][0]['message']['content']
        else:
            # Fallback
            response = response_data.get('text', response_data.get('message', {}).get('content', ''))
        
        # Parse response according to baseline format
        verdict = "UNKNOWN"
        score = 0.5
        reasoning = "Failed to parse judge response"
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('VERDICT:'):
                verdict_raw = line.replace('VERDICT:', '').strip().upper()
                # Map to our format
                if 'CORRECT' in verdict_raw and 'PARTIALLY' not in verdict_raw:
                    verdict = "CORRECT"
                elif 'INCORRECT' in verdict_raw:
                    verdict = "INCORRECT"
                elif 'PARTIALLY' in verdict_raw:
                    verdict = "PARTIALLY_CORRECT"
            elif line.startswith('SCORE:'):
                try:
                    score = float(line.replace('SCORE:', '').strip())
                except:
                    score = 0.5
            elif line.startswith('REASON:'):
                reasoning = line.replace('REASON:', '').strip()
        
        return verdict, score, reasoning
    
    def run_experiment(
        self,
        truthqa_path: str,
        max_samples: int = None,
        output_dir: str = "truthqa_judge_verification_baseline"
    ) -> Dict:
        """Run the full judge verification experiment.
        
        Args:
            truthqa_path: Path to TruthQA JSON file
            max_samples: Maximum samples to evaluate (None for all)
            output_dir: Directory to save results
            
        Returns:
            Dictionary with experiment results and metrics
        """
        print("\n" + "="*80)
        print("TruthQA Judge Verification Experiment - BASELINE PROMPT STRUCTURE")
        print("="*80)
        
        # Load data
        qa_pairs = self.load_truthqa_data(truthqa_path)
        
        if max_samples:
            qa_pairs = qa_pairs[:max_samples]
            print(f"\nLimited to {max_samples} samples")
        
        # Run judgments
        print(f"\nEvaluating {len(qa_pairs)} Q&A pairs...")
        results = []
        
        for i, item in enumerate(qa_pairs, 1):
            question = item['question']
            answer = item['answer']
            is_truthful = item['is_truthful']
            
            print(f"\n[{i}/{len(qa_pairs)}] Q: {question[:80]}...")
            print(f"Ground Truth: {'TRUTHFUL' if is_truthful else 'HALLUCINATED'}")
            
            # Get judge verdict
            try:
                verdict, score, reasoning = self.judge_single_qa(question, answer)
                
                # Determine if prediction is correct
                # CORRECT/PARTIALLY_CORRECT means judge says factually accurate (truthful)
                # INCORRECT means judge says hallucinated
                judge_says_truthful = (verdict in ["CORRECT", "PARTIALLY_CORRECT"])
                actually_truthful = is_truthful
                correct = (judge_says_truthful == actually_truthful)
                
                print(f"Judge Verdict: {verdict} (score: {score:.1f})")
                print(f"Correct: {'✓' if correct else '✗'}")
                
                result = JudgeResult(
                    question=question,
                    answer=answer,
                    is_truthful_actual=is_truthful,
                    judge_verdict=verdict,
                    judge_score=score,
                    judge_reasoning=reasoning,
                    correct_prediction=correct
                )
                results.append(result)
                
            except Exception as e:
                print(f"ERROR judging Q&A pair {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"truthqa_judge_baseline_results_{timestamp}.json"
        self._save_results(results, metrics, results_file)
        
        # Print summary
        self._print_summary(metrics)
        
        print(f"\n" + "="*80)
        print(f"Results saved to: {output_dir}/")
        print(f"  - {results_file.name}")
        print("="*80)
        
        return {
            'results': results,
            'metrics': metrics,
            'output_dir': str(output_dir),
            'timestamp': timestamp
        }
    
    def _calculate_metrics(self, results: List[JudgeResult]) -> Dict:
        """Calculate performance metrics."""
        if not results:
            return {}
        
        # Convert to binary predictions
        # Truthful = 0, Hallucinated = 1 (for positive class)
        y_true = [not r.is_truthful_actual for r in results]  # 1 = hallucinated
        y_pred = [r.judge_verdict == "INCORRECT" for r in results]  # 1 = judge says hallucinated (incorrect)
        
        # Calculate confusion matrix elements
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)  # Hallucinated, Judge says Hallucinated
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)  # Truthful, Judge says Hallucinated
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)  # Truthful, Judge says Truthful
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)  # Hallucinated, Judge says Truthful
        
        # Calculate metrics
        accuracy = (tp + tn) / len(results) if results else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Score analysis
        avg_score_all = np.mean([r.judge_score for r in results])
        avg_score_correct = np.mean([r.judge_score for r in results if r.correct_prediction])
        avg_score_incorrect = np.mean([r.judge_score for r in results if not r.correct_prediction])
        
        return {
            'n_samples': len(results),
            'n_hallucinated': sum(y_true),
            'n_truthful': sum(not yt for yt in y_true),
            'confusion_matrix': {
                'tp': tp,  # Correctly identified hallucinations
                'fp': fp,  # Falsely flagged as hallucination
                'tn': tn,  # Correctly identified truthful
                'fn': fn   # Missed hallucinations
            },
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'avg_score': avg_score_all,
            'avg_score_correct': avg_score_correct,
            'avg_score_incorrect': avg_score_incorrect
        }
    
    def _save_results(self, results: List[JudgeResult], metrics: Dict, output_file: Path):
        """Save results to JSON file."""
        output_data = {
            'experiment': 'TruthQA Judge Verification - Baseline Prompt Structure',
            'timestamp': datetime.now().isoformat(),
            'judge': {
                'provider': self.judge_provider,
                'model': self.judge_model,
                'temperature': self.judge_temperature,
                'max_tokens': self.max_tokens,
                'seed': self.seed,
                'prompt_variant': 'baseline',
                'yaml_path': self.yaml_path
            },
            'metrics': metrics,
            'detailed_results': [
                {
                    'question': r.question,
                    'answer': r.answer[:500] + '...' if len(r.answer) > 500 else r.answer,
                    'is_truthful_actual': r.is_truthful_actual,
                    'judge_verdict': r.judge_verdict,
                    'judge_score': r.judge_score,
                    'judge_reasoning': r.judge_reasoning,
                    'correct_prediction': r.correct_prediction
                }
                for r in results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def _print_summary(self, metrics: Dict):
        """Print experiment summary."""
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS")
        print("="*80)
        
        if not metrics or 'n_samples' not in metrics:
            print("\n⚠️  No results to display - all samples failed to process")
            print("Check the errors above for details")
            return
        
        print(f"\nDataset:")
        print(f"  Total samples: {metrics['n_samples']}")
        print(f"  Hallucinated: {metrics['n_hallucinated']} ({metrics['n_hallucinated']/metrics['n_samples']*100:.1f}%)")
        print(f"  Truthful: {metrics['n_truthful']} ({metrics['n_truthful']/metrics['n_samples']*100:.1f}%)")
        
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (caught hallucinations): {cm['tp']}")
        print(f"  False Positives (false alarms): {cm['fp']}")
        print(f"  True Negatives (correctly accepted): {cm['tn']}")
        print(f"  False Negatives (missed hallucinations): {cm['fn']}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        
        print(f"\nScore Analysis:")
        print(f"  Average score (all): {metrics['avg_score']:.2f}")
        print(f"  Avg score (correct predictions): {metrics['avg_score_correct']:.2f}")
        print(f"  Avg score (incorrect predictions): {metrics['avg_score_incorrect']:.2f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TruthQA Judge Verification using Baseline Prompt Structure (mirrors correctness baseline)"
    )
    parser.add_argument(
        '--truthqa-path',
        type=str,
        default='../qwen3_truthfulqa_evaluation_20250806_222711.json',
        help='Path to TruthQA JSON file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--judge-provider',
        type=str,
        default='openai',
        help='Judge LLM provider (default: openai)'
    )
    parser.add_argument(
        '--judge-model',
        type=str,
        default='gpt-4.1',
        help='Judge model (default: gpt-4.1)'
    )
    parser.add_argument(
        '--judge-temperature',
        type=float,
        default=0.3,
        help='Judge temperature (default: 0.3)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=250,
        help='Maximum tokens for judge response (default: 250)'
    )
    parser.add_argument(
        '--yaml-path',
        type=str,
        default=None,
        help='Path to YAML prompt file (default: prompts_truthqa_baseline.yaml)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='truthqa_judge_verification_baseline',
        help='Output directory (default: truthqa_judge_verification_baseline)'
    )
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = TruthQAJudgeExperiment(
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        yaml_path=args.yaml_path
    )
    
    # Run experiment
    experiment.run_experiment(
        truthqa_path=args.truthqa_path,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
