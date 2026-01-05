"""
Test script for Pure Synergistic Hidden State Analysis

This tests the CORRECT approach:
- Calculate H(Σ(X)) directly from hidden states
- Correlate synergistic entropy with hallucination labels
- No output logits involved
"""

import torch
import numpy as np
from pure_synergistic_hidden_state_analysis import PureSynergisticAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_toy_example():
    """
    Test with toy example similar to XOR from the paper.
    
    Create two scenarios:
    1. Synergistic: Information requires combining components (like XOR)
    2. Individual: Information in single component (like COPY)
    
    Expectation: Synergistic case has higher H(Σ(X))
    """
    logger.info("=" * 60)
    logger.info("TEST 1: Toy Example (XOR-like vs COPY-like)")
    logger.info("=" * 60)
    
    analyzer = PureSynergisticAnalyzer(n_components=4)
    
    # Create synergistic hidden states
    # Components must be combined to extract information
    torch.manual_seed(42)
    n_samples = 100
    hidden_dim = 768
    
    # Synergistic case: XOR-like pattern
    # Information is in the COMBINATION, not individuals
    comp1_syn = torch.randn(n_samples, hidden_dim // 4)
    comp2_syn = torch.randn(n_samples, hidden_dim // 4)
    comp3_syn = comp1_syn + comp2_syn + torch.randn(n_samples, hidden_dim // 4) * 0.1  # XOR-like
    comp4_syn = torch.randn(n_samples, hidden_dim // 4)
    hidden_syn = torch.cat([comp1_syn, comp2_syn, comp3_syn, comp4_syn], dim=-1)
    
    # Individual case: COPY-like pattern
    # Information is in ONE component
    comp1_ind = torch.randn(n_samples, hidden_dim // 4)
    comp2_ind = torch.randn(n_samples, hidden_dim // 4) * 0.1  # Noise
    comp3_ind = torch.randn(n_samples, hidden_dim // 4) * 0.1  # Noise
    comp4_ind = torch.randn(n_samples, hidden_dim // 4) * 0.1  # Noise
    hidden_ind = torch.cat([comp1_ind, comp2_ind, comp3_ind, comp4_ind], dim=-1)
    
    # Analyze both
    metrics_syn = analyzer.analyze_layer(hidden_syn)
    metrics_ind = analyzer.analyze_layer(hidden_ind)
    
    logger.info(f"\nSynergistic case:")
    logger.info(f"  H(Σ(X)) = {metrics_syn.synergistic_entropy:.3f}")
    logger.info(f"  H(X) = {metrics_syn.total_entropy:.3f}")
    logger.info(f"  Synergy ratio = {metrics_syn.synergy_ratio:.3f}")
    logger.info(f"  Hallucination risk = {metrics_syn.hallucination_risk:.3f}")
    
    logger.info(f"\nIndividual case:")
    logger.info(f"  H(Σ(X)) = {metrics_ind.synergistic_entropy:.3f}")
    logger.info(f"  H(X) = {metrics_ind.total_entropy:.3f}")
    logger.info(f"  Synergy ratio = {metrics_ind.synergy_ratio:.3f}")
    logger.info(f"  Hallucination risk = {metrics_ind.hallucination_risk:.3f}")
    
    # Expectation: Synergistic case has higher synergy
    if metrics_syn.synergy_ratio > metrics_ind.synergy_ratio:
        logger.info("\n✓ PASS: Synergistic case has higher synergy ratio")
    else:
        logger.warning("\n✗ FAIL: Individual case has higher synergy ratio")
    
    return metrics_syn, metrics_ind


def test_multi_layer():
    """
    Test with multiple layers.
    
    Expectation: Can analyze each layer independently
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Multi-Layer Analysis")
    logger.info("=" * 60)
    
    analyzer = PureSynergisticAnalyzer(n_components=4)
    
    # Create hidden states for multiple layers
    torch.manual_seed(42)
    n_layers = 6
    n_samples = 50
    hidden_dim = 768
    
    hidden_states_all_layers = []
    for i in range(n_layers):
        # Make later layers more synergistic (simulating hallucination pattern)
        synergy_strength = i / n_layers  # 0 to 1
        
        comp1 = torch.randn(n_samples, hidden_dim // 4)
        comp2 = torch.randn(n_samples, hidden_dim // 4)
        # More synergistic combination in later layers
        comp3 = synergy_strength * (comp1 + comp2) + (1 - synergy_strength) * torch.randn(n_samples, hidden_dim // 4)
        comp4 = torch.randn(n_samples, hidden_dim // 4)
        
        hidden_states = torch.cat([comp1, comp2, comp3, comp4], dim=-1)
        hidden_states_all_layers.append(hidden_states)
    
    # Analyze all layers
    layer_metrics = analyzer.analyze_all_layers(hidden_states_all_layers)
    
    logger.info("\nSynergy ratio by layer:")
    for layer_idx, metrics in layer_metrics.items():
        logger.info(f"  Layer {layer_idx}: {metrics.synergy_ratio:.3f}")
    
    # Expectation: Synergy increases with layer depth
    synergy_ratios = [m.synergy_ratio for m in layer_metrics.values()]
    if np.corrcoef(range(len(synergy_ratios)), synergy_ratios)[0, 1] > 0:
        logger.info("\n✓ PASS: Synergy increases with layer depth")
    else:
        logger.warning("\n✗ FAIL: Synergy doesn't increase with layer depth")
    
    return layer_metrics


def test_hallucination_correlation():
    """
    Test correlation between synergy and hallucination labels.
    
    Simulate:
    - Truthful samples: Low synergy (information in individual components)
    - Hallucinated samples: High synergy (information spread across components)
    
    Expectation: Positive correlation between synergy and hallucination
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Synergy-Hallucination Correlation")
    logger.info("=" * 60)
    
    analyzer = PureSynergisticAnalyzer(n_components=4)
    
    torch.manual_seed(42)
    n_samples = 20
    hidden_dim = 768
    
    synergy_scores = []
    hallucination_labels = []
    
    # Generate samples with varying synergy
    for i in range(n_samples):
        # Assign hallucination label
        is_hallucination = i >= n_samples // 2
        hallucination_labels.append(int(is_hallucination))
        
        # Create hidden states
        comp1 = torch.randn(1, hidden_dim // 4)
        comp2 = torch.randn(1, hidden_dim // 4)
        
        if is_hallucination:
            # High synergy: XOR-like combination
            comp3 = comp1 + comp2 + torch.randn(1, hidden_dim // 4) * 0.1
            comp4 = comp1 - comp2 + torch.randn(1, hidden_dim // 4) * 0.1
        else:
            # Low synergy: Independent components
            comp3 = torch.randn(1, hidden_dim // 4)
            comp4 = torch.randn(1, hidden_dim // 4)
        
        hidden_states = torch.cat([comp1, comp2, comp3, comp4], dim=-1)
        
        # Analyze
        metrics = analyzer.analyze_layer(hidden_states)
        synergy_scores.append(metrics.synergy_ratio)
    
    # Compute correlation
    correlation = np.corrcoef(synergy_scores, hallucination_labels)[0, 1]
    
    logger.info(f"\nGenerated {n_samples} samples:")
    logger.info(f"  {sum(hallucination_labels)} hallucinations")
    logger.info(f"  {n_samples - sum(hallucination_labels)} truthful")
    logger.info(f"\nCorrelation(synergy, hallucination): {correlation:.3f}")
    
    logger.info(f"\nMean synergy by label:")
    truthful_synergy = np.mean([s for s, h in zip(synergy_scores, hallucination_labels) if h == 0])
    halluc_synergy = np.mean([s for s, h in zip(synergy_scores, hallucination_labels) if h == 1])
    logger.info(f"  Truthful: {truthful_synergy:.3f}")
    logger.info(f"  Hallucination: {halluc_synergy:.3f}")
    
    # Expectation: Positive correlation
    if correlation > 0.3:
        logger.info("\n✓ PASS: Positive correlation between synergy and hallucination")
    else:
        logger.warning(f"\n✗ FAIL: Weak/negative correlation ({correlation:.3f})")
    
    return synergy_scores, hallucination_labels, correlation


def main():
    """Run all tests."""
    logger.info("Testing Pure Synergistic Hidden State Analysis")
    logger.info("=" * 60)
    
    # Test 1: Toy example
    test_toy_example()
    
    # Test 2: Multi-layer
    test_multi_layer()
    
    # Test 3: Hallucination correlation
    test_hallucination_correlation()
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
