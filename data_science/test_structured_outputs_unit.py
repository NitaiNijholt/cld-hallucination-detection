#!/usr/bin/env python3
"""
Unit tests for structured outputs functionality.

Tests the Pydantic models and adapter without requiring the full pipeline.
This is a quick sanity check before running full integration tests.
"""

import sys
import json
from typing import Dict, Any

def test_imports():
    """Test 1: Verify all imports work"""
    print("Test 1: Checking imports...")
    try:
        from judge_models import CorrectnessJudgment, CitationJudgment
        from judge_response_adapter import JudgeResponseAdapter
        print("  ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        print("  💡 Run: pip install 'pydantic>=2.0.0'")
        return False


def test_correctness_model():
    """Test 2: Verify CorrectnessJudgment model works"""
    print("\nTest 2: Testing CorrectnessJudgment model...")
    try:
        from judge_models import CorrectnessJudgment
        
        # Create a judgment
        judgment = CorrectnessJudgment(
            reason="The causal mechanism is well-explained and scientifically sound.",
            score=1.0,
            verdict="CORRECT"
        )
        
        # Verify fields
        assert judgment.reason == "The causal mechanism is well-explained and scientifically sound."
        assert judgment.score == 1.0
        assert judgment.verdict == "CORRECT"
        
        # Verify JSON schema generation
        schema = CorrectnessJudgment.model_json_schema()
        assert "properties" in schema
        assert "reason" in schema["properties"]
        assert "score" in schema["properties"]
        assert "verdict" in schema["properties"]
        
        print("  ✅ CorrectnessJudgment model works correctly")
        print(f"     reason: {judgment.reason[:50]}...")
        print(f"     score: {judgment.score}")
        print(f"     verdict: {judgment.verdict}")
        return True
        
    except Exception as e:
        print(f"  ❌ CorrectnessJudgment model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citation_model():
    """Test 3: Verify CitationJudgment model works"""
    print("\nTest 3: Testing CitationJudgment model...")
    try:
        from judge_models import CitationJudgment
        
        # Create a judgment
        judgment = CitationJudgment(
            reason="The citations provide direct evidence for the proposed relationship.",
            score=1.0,
            verdict="SUPPORTED"
        )
        
        # Verify fields
        assert judgment.reason == "The citations provide direct evidence for the proposed relationship."
        assert judgment.score == 1.0
        assert judgment.verdict == "SUPPORTED"
        
        # Verify JSON schema generation
        schema = CitationJudgment.model_json_schema()
        assert "properties" in schema
        assert "reason" in schema["properties"]
        assert "score" in schema["properties"]
        assert "verdict" in schema["properties"]
        
        print("  ✅ CitationJudgment model works correctly")
        print(f"     reason: {judgment.reason[:50]}...")
        print(f"     score: {judgment.score}")
        print(f"     verdict: {judgment.verdict}")
        return True
        
    except Exception as e:
        print(f"  ❌ CitationJudgment model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_adapter():
    """Test 4: Verify response adapter converts correctly"""
    print("\nTest 4: Testing JudgeResponseAdapter...")
    try:
        from judge_models import CorrectnessJudgment, CitationJudgment
        from judge_response_adapter import JudgeResponseAdapter
        
        # Test with CorrectnessJudgment
        judgment1 = CorrectnessJudgment(
            reason="Test reason",
            score=0.5,
            verdict="PARTIALLY_CORRECT"
        )
        
        result1 = JudgeResponseAdapter.to_legacy_format(judgment1)
        assert result1["verdict"] == "PARTIALLY_CORRECT"
        assert result1["score"] == 0.5
        assert result1["reason"] == "Test reason"
        
        # Test with CitationJudgment
        judgment2 = CitationJudgment(
            reason="Citations partially support the claim",
            score=0.5,
            verdict="PARTIALLY_SUPPORTED"
        )
        
        result2 = JudgeResponseAdapter.to_legacy_format(judgment2)
        assert result2["verdict"] == "PARTIALLY_SUPPORTED"
        assert result2["score"] == 0.5
        assert result2["reason"] == "Citations partially support the claim"
        
        print("  ✅ Response adapter works correctly")
        print(f"     Correctness: {result1}")
        print(f"     Citation: {result2}")
        return True
        
    except Exception as e:
        print(f"  ❌ Response adapter failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compatibility_check():
    """Test 5: Verify model compatibility checker"""
    print("\nTest 5: Testing model compatibility checker...")
    try:
        from judge_response_adapter import JudgeResponseAdapter
        
        # Test compatible models
        compatible_models = [
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-09-15",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18"
        ]
        
        for model in compatible_models:
            assert JudgeResponseAdapter.supports_structured_outputs(model), \
                f"Model {model} should be compatible"
        
        # Test incompatible models
        incompatible_models = [
            "gpt-4.1",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-sonnet"
        ]
        
        for model in incompatible_models:
            assert not JudgeResponseAdapter.supports_structured_outputs(model), \
                f"Model {model} should NOT be compatible"
        
        print("  ✅ Model compatibility checker works correctly")
        print(f"     Compatible models: {len(compatible_models)}")
        print(f"     Incompatible models: {len(incompatible_models)}")
        return True
        
    except Exception as e:
        print(f"  ❌ Model compatibility check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_schema_format():
    """Test 6: Verify JSON schema is correctly formatted for OpenAI"""
    print("\nTest 6: Testing JSON schema format...")
    try:
        from judge_models import CorrectnessJudgment
        
        schema = CorrectnessJudgment.model_json_schema()
        
        # Check required fields
        assert "properties" in schema
        assert "required" in schema
        
        # Verify field order (REASON → SCORE → VERDICT)
        props = schema["properties"]
        assert "reason" in props
        assert "score" in props
        assert "verdict" in props
        
        # Verify score constraints
        assert props["score"]["type"] == "number"
        assert props["score"]["minimum"] == 0.0
        assert props["score"]["maximum"] == 1.0
        
        # Verify verdict enum
        assert props["verdict"]["type"] == "string"
        assert "enum" in props["verdict"]
        assert set(props["verdict"]["enum"]) == {"CORRECT", "PARTIALLY_CORRECT", "INCORRECT"}
        
        print("  ✅ JSON schema format is correct")
        print(f"     Properties: {list(props.keys())}")
        print(f"     Required: {schema['required']}")
        return True
        
    except Exception as e:
        print(f"  ❌ JSON schema format check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_field_ordering():
    """Test 7: Verify field ordering (REASON → SCORE → VERDICT)"""
    print("\nTest 7: Testing field ordering...")
    try:
        from judge_models import CorrectnessJudgment, CitationJudgment
        
        # Get field names in definition order
        correctness_fields = list(CorrectnessJudgment.model_fields.keys())
        citation_fields = list(CitationJudgment.model_fields.keys())
        
        # Verify order: reason, score, verdict
        assert correctness_fields == ["reason", "score", "verdict"], \
            f"CorrectnessJudgment field order is wrong: {correctness_fields}"
        assert citation_fields == ["reason", "score", "verdict"], \
            f"CitationJudgment field order is wrong: {citation_fields}"
        
        print("  ✅ Field ordering is correct (REASON → SCORE → VERDICT)")
        print(f"     CorrectnessJudgment: {correctness_fields}")
        print(f"     CitationJudgment: {citation_fields}")
        return True
        
    except Exception as e:
        print(f"  ❌ Field ordering check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("STRUCTURED OUTPUTS UNIT TESTS")
    print("="*80)
    
    tests = [
        test_imports,
        test_correctness_model,
        test_citation_model,
        test_response_adapter,
        test_model_compatibility_check,
        test_json_schema_format,
        test_field_ordering
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"Test {i} ({test_func.__name__}): {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL UNIT TESTS PASSED!")
        print("\n📝 Next steps:")
        print("   1. Run full integration test: python parameter_tuning_experiments/test_structured_outputs.py")
        print("   2. Or enable in your experiments: set use_structured_outputs=True in judge_config")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        if not results[0]:  # Import test failed
            print("\n💡 Install Pydantic: pip install 'pydantic>=2.0.0'")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
