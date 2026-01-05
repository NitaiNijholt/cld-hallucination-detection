#!/usr/bin/env python3
"""
Quick Start Guide for Spectral Hallucination Detection

This script demonstrates the basic usage of the spectral hallucination detector.
"""

import sys
import os

# Add parent directory to path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main demonstration function."""
    
    print("="*80)
    print("Spectral Hallucination Detection - Quick Start")
    print("="*80)
    
    # Option 1: Run simple demo (no GPU required)
    print("\n1. Running Simple Demo (No GPU Required)...")
    print("-" * 40)
    
    try:
        from simple_spectral_demo import demonstrate_spectral_detection
        import numpy as np
        np.random.seed(42)
        demonstrate_spectral_detection()
    except ImportError as e:
        print(f"Note: Simple demo requires numpy and matplotlib: {e}")
    
    # Option 2: Run with real models (requires GPU and models)
    print("\n2. To run with real models:")
    print("-" * 40)
    print("""
    from spectral_hallucination_detection import SpectralHallucinationDetector
    
    # Initialize detector
    detector = SpectralHallucinationDetector(
        model_name="microsoft/phi-2",  # Or any HuggingFace model
        device="cuda"  # Or "cpu"
    )
    
    # Test for hallucination
    text = "The iPhone 27 uses quantum processors"
    is_hallucinating, metrics = detector.detect_hallucination(text)
    
    print(f"Hallucination detected: {is_hallucinating}")
    print(f"Risk score: {metrics['scores']['overall_risk']:.3f}")
    """)
    
    # Option 3: vLLM high-performance
    print("\n3. For high-performance inference with vLLM:")
    print("-" * 40)
    print("""
    from vllm_spectral_detection import VLLMSpectralDetector
    
    # Initialize with vLLM
    detector = VLLMSpectralDetector(
        model_name="mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    # Batch detection
    prompts = ["The capital of France is", "The secret of time travel is"]
    results = detector.batch_detect(prompts)
    """)
    
    print("\n" + "="*80)
    print("For full documentation, see README.md")
    print("For examples, see demo.ipynb")
    print("="*80)


if __name__ == "__main__":
    main()