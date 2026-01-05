"""
vLLM-based Spectral Hallucination Detection
High-performance inference with access to model internals

vLLM provides faster inference than vanilla transformers while still
allowing access to hidden states and logits for analysis.
"""

import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Any
import json

class VLLMSpectralDetector:
    """
    Fast hallucination detection using vLLM for inference.
    Combines high-throughput with spectral analysis.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        tensor_parallel_size: int = 1,  # Number of GPUs
        max_model_len: int = 2048
    ):
        """
        Initialize vLLM with a model.
        
        Args:
            model_name: HuggingFace model to use
            tensor_parallel_size: Number of GPUs for parallel inference
            max_model_len: Maximum sequence length
        """
        print(f"Loading vLLM with model: {model_name}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate_with_logits(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Generate text and extract logit information.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of generation results with logit analysis
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=5  # Get top-5 log probabilities
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            
            # Analyze logprobs for uncertainty
            logprobs = output.outputs[0].logprobs
            
            if logprobs:
                # Calculate entropy from logprobs (uncertainty measure)
                entropies = []
                for token_logprobs in logprobs:
                    if token_logprobs:
                        # Convert logprobs to probabilities
                        probs = [np.exp(lp.logprob) for lp in token_logprobs.values()]
                        # Calculate entropy
                        probs = np.array(probs) / np.sum(probs)
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        entropies.append(entropy)
                
                avg_entropy = np.mean(entropies) if entropies else 0.0
                max_entropy = np.max(entropies) if entropies else 0.0
            else:
                avg_entropy = 0.0
                max_entropy = 0.0
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'avg_entropy': avg_entropy,
                'max_entropy': max_entropy,
                'num_tokens': len(generated_text.split())
            })
        
        return results
    
    def detect_hallucination_from_entropy(
        self,
        entropy_scores: Dict[str, float],
        threshold: float = 1.5
    ) -> bool:
        """
        Simple hallucination detection based on entropy.
        High entropy suggests uncertainty/potential hallucination.
        
        Args:
            entropy_scores: Dictionary with entropy metrics
            threshold: Threshold for hallucination detection
            
        Returns:
            Boolean indicating potential hallucination
        """
        # Combine average and max entropy
        risk_score = (entropy_scores['avg_entropy'] + 
                     0.5 * entropy_scores['max_entropy']) / 1.5
        
        return risk_score > threshold
    
    def batch_detect(
        self,
        prompts: List[str],
        continuation_test: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch hallucination detection for multiple prompts.
        
        Args:
            prompts: List of prompts to test
            continuation_test: Whether to test consistency of continuations
            
        Returns:
            List of detection results
        """
        results = []
        
        # Generate continuations
        generations = self.generate_with_logits(prompts, max_tokens=50)
        
        for i, gen in enumerate(generations):
            is_hallucinating = self.detect_hallucination_from_entropy(
                {
                    'avg_entropy': gen['avg_entropy'],
                    'max_entropy': gen['max_entropy']
                }
            )
            
            result = {
                'prompt': gen['prompt'],
                'is_hallucinating': is_hallucinating,
                'entropy_score': gen['avg_entropy'],
                'generated_text': gen['generated']
            }
            
            # Optional: Test consistency by regenerating
            if continuation_test:
                # Generate multiple times to check consistency
                multi_gen = self.generate_with_logits(
                    [gen['prompt']] * 3,
                    max_tokens=30,
                    temperature=0.3  # Lower temperature for consistency test
                )
                
                # Check if generations are similar
                texts = [g['generated'] for g in multi_gen]
                
                # Simple consistency check: do they start similarly?
                first_words = [t.split()[:5] for t in texts]
                consistency_score = len(set([' '.join(w) for w in first_words])) / 3.0
                
                result['consistency_score'] = 1.0 - consistency_score  # Higher = more consistent
                result['is_consistent'] = consistency_score < 0.5
            
            results.append(result)
        
        return results


def run_vllm_experiments():
    """Run experiments using vLLM for fast inference."""
    
    print("Initializing vLLM Spectral Detector...")
    print("Note: This requires vLLM to be installed: pip install vllm")
    
    try:
        detector = VLLMSpectralDetector(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            tensor_parallel_size=1
        )
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        print("Falling back to mock mode for demonstration...")
        return run_mock_experiments()
    
    # Test prompts that might trigger hallucinations
    test_prompts = [
        "The capital of France is",
        "The iPhone 27 Pro Max features",
        "According to recent studies, eating chocolate",
        "The speed of light in vacuum is",
        "The secret ingredient in Coca-Cola is",
        "Quantum computers can solve"
    ]
    
    print("\n" + "="*80)
    print("Running vLLM Hallucination Detection")
    print("="*80)
    
    results = detector.batch_detect(test_prompts, continuation_test=True)
    
    for i, result in enumerate(results, 1):
        print(f"\nTest {i}:")
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_text'][:100]}...")
        print(f"Hallucination Risk: {'HIGH' if result['is_hallucinating'] else 'LOW'}")
        print(f"Entropy Score: {result['entropy_score']:.3f}")
        
        if 'consistency_score' in result:
            print(f"Consistency Score: {result['consistency_score']:.3f}")
            print(f"Consistent: {'Yes' if result['is_consistent'] else 'No'}")


def run_mock_experiments():
    """Mock experiments for demonstration when vLLM is not available."""
    
    print("\nRunning in mock mode (vLLM not available)")
    print("="*80)
    
    mock_results = [
        {
            'prompt': "The capital of France is",
            'generated': "Paris, which has been the capital since...",
            'entropy': 0.3,
            'risk': 'LOW'
        },
        {
            'prompt': "The iPhone 27 Pro Max features",
            'generated': "a holographic display with quantum processing...",
            'entropy': 2.1,
            'risk': 'HIGH'
        },
        {
            'prompt': "The speed of light in vacuum is",
            'generated': "299,792,458 meters per second...",
            'entropy': 0.2,
            'risk': 'LOW'
        }
    ]
    
    for result in mock_results:
        print(f"\nPrompt: {result['prompt']}")
        print(f"Generated: {result['generated']}")
        print(f"Entropy: {result['entropy']:.2f}")
        print(f"Hallucination Risk: {result['risk']}")


if __name__ == "__main__":
    run_vllm_experiments()