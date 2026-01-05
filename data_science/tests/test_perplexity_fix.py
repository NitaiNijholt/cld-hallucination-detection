#!/usr/bin/env python3
"""
Unit tests for perplexity calculation with token-level capping.

Tests the fix for extreme perplexity values (e.g., 5.55E+28) caused by
tokens with near-zero probability (logprob ≈ -100 or worse).

Run: python -m pytest tests/test_perplexity_fix.py -v
"""

import sys
import math
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logit_metrics import compute_avg_nll, compute_perplexity


def test_normal_case():
    """Test with normal log probabilities (no extreme values)."""
    log_data = {
        "tokens": ["The", " cat", " sat"],
        "token_logprobs": [-0.1, -0.5, -0.3]  # Normal range
    }
    
    avg_nll = compute_avg_nll(log_data)
    perplexity = compute_perplexity(log_data)
    
    # Expected: avg_nll = (0.1 + 0.5 + 0.3) / 3 = 0.3
    # Expected perplexity = exp(0.3) ≈ 1.35
    assert avg_nll is not None
    assert abs(avg_nll - 0.3) < 0.001, f"Expected avg_nll=0.3, got {avg_nll}"
    assert perplexity is not None
    assert abs(perplexity - math.exp(0.3)) < 0.01, f"Expected perplexity≈1.35, got {perplexity}"
    print(f"✓ Normal case: avg_nll={avg_nll:.4f}, perplexity={perplexity:.4f}")


def test_single_extreme_token():
    """Test with ONE extreme token (logprob=-100) among normal tokens."""
    log_data = {
        "tokens": ["The", " cat", " <EXTREME>", " sat"],
        "token_logprobs": [-0.1, -0.5, -100.0, -0.3]  # One extreme outlier
    }
    
    avg_nll = compute_avg_nll(log_data)
    perplexity = compute_perplexity(log_data)
    
    # Without capping: avg_nll = (0.1 + 0.5 + 100 + 0.3) / 4 = 25.225
    # Without capping: perplexity = exp(25.225) ≈ 9.5E+10 (EXPLOSION!)
    
    # With capping: extreme token capped at 20.0
    # avg_nll = (0.1 + 0.5 + 20.0 + 0.3) / 4 = 5.225
    # perplexity = exp(5.225) ≈ 186
    
    assert avg_nll is not None
    assert avg_nll < 10.0, f"avg_nll should be capped below 10.0, got {avg_nll}"
    assert perplexity is not None
    assert perplexity < 25000, f"Perplexity should be reasonable (<25k), got {perplexity}"
    print(f"✓ Single extreme token: avg_nll={avg_nll:.4f}, perplexity={perplexity:.4f}")


def test_multiple_extreme_tokens():
    """Test with MULTIPLE extreme tokens (reproducing actual bug scenario)."""
    log_data = {
        "tokens": ["Token1", "Token2", "Token3", "Token4", "Token5"],
        "token_logprobs": [-50.0, -100.0, -0.5, -75.0, -0.3]  # Multiple outliers
    }
    
    avg_nll = compute_avg_nll(log_data)
    perplexity = compute_perplexity(log_data)
    
    # Without capping: avg_nll = (50 + 100 + 0.5 + 75 + 0.3) / 5 = 45.16
    # Without capping: perplexity = exp(45.16) = 3.4E+19 (MASSIVE EXPLOSION!)
    
    # With capping: all extreme tokens capped at 20.0
    # avg_nll = (20 + 20 + 0.5 + 20 + 0.3) / 5 = 12.16
    # perplexity = min(exp(12.16), exp(10)) = exp(10) ≈ 22026 (final cap)
    
    assert avg_nll is not None
    assert avg_nll < 25.0, f"avg_nll should be well below 25.0, got {avg_nll}"
    assert perplexity is not None
    assert perplexity < 25000, f"Perplexity should be capped at ~22026, got {perplexity}"
    assert perplexity <= math.exp(10.0) + 1, f"Perplexity exceeds final cap, got {perplexity}"
    print(f"✓ Multiple extreme tokens: avg_nll={avg_nll:.4f}, perplexity={perplexity:.4f}")


def test_all_extreme_tokens():
    """Test with ALL tokens being extreme (worst case scenario)."""
    log_data = {
        "tokens": ["Extreme1", "Extreme2", "Extreme3"],
        "token_logprobs": [-200.0, -150.0, -100.0]  # All extremely low
    }
    
    avg_nll = compute_avg_nll(log_data)
    perplexity = compute_perplexity(log_data)
    
    # Without capping: avg_nll = (200 + 150 + 100) / 3 = 150
    # Without capping: perplexity = exp(150) = INFINITY (overflow!)
    
    # With capping: all tokens capped at 20.0
    # avg_nll = (20 + 20 + 20) / 3 = 20.0
    # perplexity = min(exp(20), exp(10)) = exp(10) ≈ 22026
    
    assert avg_nll is not None
    assert avg_nll == 20.0, f"avg_nll should be exactly 20.0 when all tokens capped, got {avg_nll}"
    assert perplexity is not None
    assert perplexity <= math.exp(10.0) + 1, f"Perplexity should be final capped, got {perplexity}"
    print(f"✓ All extreme tokens: avg_nll={avg_nll:.4f}, perplexity={perplexity:.4f}")


def test_infinity_and_nan_handling():
    """Test with -inf and NaN values (API errors)."""
    log_data = {
        "tokens": ["Token1", "Token2", "Token3", "Token4"],
        "token_logprobs": [-0.5, -np.inf, np.nan, -1.0]  # Pathological cases
    }
    
    avg_nll = compute_avg_nll(log_data)
    perplexity = compute_perplexity(log_data)
    
    # -inf and NaN should be capped at 20.0
    # avg_nll = (0.5 + 20.0 + 20.0 + 1.0) / 4 = 10.375
    # perplexity = min(exp(10.375), exp(10)) = exp(10) ≈ 22026
    
    assert avg_nll is not None
    assert np.isfinite(avg_nll), f"avg_nll should be finite, got {avg_nll}"
    assert perplexity is not None
    assert np.isfinite(perplexity), f"Perplexity should be finite, got {perplexity}"
    print(f"✓ Infinity/NaN handling: avg_nll={avg_nll:.4f}, perplexity={perplexity:.4f}")


def test_empty_input():
    """Test with empty token_logprobs."""
    log_data = {
        "tokens": [],
        "token_logprobs": []
    }
    
    avg_nll = compute_avg_nll(log_data)
    perplexity = compute_perplexity(log_data)
    
    assert avg_nll is None, "Should return None for empty input"
    assert perplexity is None, "Should return None for empty input"
    print("✓ Empty input handled correctly")


def test_reproduces_bug_scenario():
    """
    Test that reproduces the actual bug scenario from user's data:
    Perplexity values like 5.55E+28, 8.69E+12, 1.76E+18
    
    Working backwards from observed perplexities to estimate token logprobs:
    - 5.55E+28 → ln(5.55E+28) ≈ 65.6 avg_nll
    - 8.69E+12 → ln(8.69E+12) ≈ 29.9 avg_nll
    - 1.76E+18 → ln(1.76E+18) ≈ 42.1 avg_nll
    """
    # Simulate sequences that would produce such extreme perplexities
    test_cases = [
        {
            "name": "Case 1: 5.55E+28",
            "logprobs": [-65.6, -0.5, -0.3],  # One token with ~-65.6 average contribution
            "expected_max_perplexity": 25000
        },
        {
            "name": "Case 2: 8.69E+12",
            "logprobs": [-30.0, -29.8, -0.5],  # Multiple moderate-extreme tokens
            "expected_max_perplexity": 25000
        },
        {
            "name": "Case 3: 1.76E+18",
            "logprobs": [-42.0, -42.2, -0.5],  # Multiple extreme tokens
            "expected_max_perplexity": 25000
        }
    ]
    
    for case in test_cases:
        log_data = {
            "tokens": [f"T{i}" for i in range(len(case["logprobs"]))],
            "token_logprobs": case["logprobs"]
        }
        
        perplexity = compute_perplexity(log_data)
        
        assert perplexity is not None
        assert perplexity < case["expected_max_perplexity"], \
            f"{case['name']}: Perplexity {perplexity} should be < {case['expected_max_perplexity']}"
        assert np.isfinite(perplexity), \
            f"{case['name']}: Perplexity should be finite, got {perplexity}"
        
        print(f"✓ {case['name']}: perplexity={perplexity:.2f} (was {case['name'].split()[2]})")


def test_threshold_boundary():
    """Test behavior at the exact capping threshold (logprob=-20)."""
    log_data = {
        "tokens": ["T1", "T2", "T3", "T4"],
        "token_logprobs": [-19.9, -20.0, -20.1, -0.5]  # Around the threshold
    }
    
    avg_nll = compute_avg_nll(log_data)
    perplexity = compute_perplexity(log_data)
    
    # Expected: (19.9 + 20.0 + 20.0 + 0.5) / 4 = 15.1
    # perplexity = min(exp(15.1), exp(10)) = exp(10) ≈ 22026
    
    assert avg_nll is not None
    assert 15.0 < avg_nll < 15.2, f"avg_nll should be ~15.1, got {avg_nll}"
    assert perplexity is not None
    assert perplexity <= math.exp(10.0) + 1, f"Perplexity should be final capped, got {perplexity}"
    print(f"✓ Threshold boundary: avg_nll={avg_nll:.4f}, perplexity={perplexity:.4f}")


def run_all_tests():
    """Run all tests and provide summary."""
    tests = [
        ("Normal case", test_normal_case),
        ("Single extreme token", test_single_extreme_token),
        ("Multiple extreme tokens", test_multiple_extreme_tokens),
        ("All extreme tokens", test_all_extreme_tokens),
        ("Infinity and NaN handling", test_infinity_and_nan_handling),
        ("Empty input", test_empty_input),
        ("Bug reproduction", test_reproduces_bug_scenario),
        ("Threshold boundary", test_threshold_boundary),
    ]
    
    print("=" * 80)
    print("PERPLEXITY FIX - UNIT TESTS")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✓ ALL TESTS PASSED - Fix is working correctly!")
        return 0
    else:
        print(f"✗ {failed} test(s) failed - please review")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
