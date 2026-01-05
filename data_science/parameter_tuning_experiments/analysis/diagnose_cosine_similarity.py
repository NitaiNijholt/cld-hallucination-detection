"""
Diagnostic script to test why cosine similarity is not being computed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from pathlib import Path

# Try to import the CI metrics functions
try:
    from logit_metrics import compute_alignment_batch
    print("✓ Imported compute_alignment_batch from logit_metrics")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test cosine similarity computation
def test_cosine_similarity():
    """Test if we can compute cosine similarity."""
    
    print("\n" + "="*70)
    print("TESTING COSINE SIMILARITY COMPUTATION")
    print("="*70)
    
    # Get API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("\n❌ OPENAI_API_KEY not found in environment!")
        print("   Cosine similarity requires OpenAI embeddings API")
        return False
    
    print(f"\n✓ OpenAI API key found: {openai_key[:20]}...")
    
    # Test data
    motivation = "Physical Activity Level causes Total Daily Energy Expenditure to increase."
    citation_text = """
    Physical activity increases energy expenditure through metabolic processes.
    Regular exercise raises total daily energy expenditure (TDEE) significantly.
    Studies show that active individuals have 20-30% higher TDEE than sedentary individuals.
    """
    
    chunks = [citation_text]
    
    print(f"\n Testing with:")
    print(f"  Motivation: {motivation[:60]}...")
    print(f"  Citation chunks: {len(chunks)}")
    
    try:
        similarity = compute_alignment_batch(
            narrative=motivation,
            chunks=chunks,
            openai_key=openai_key,
            model="text-embedding-3-small",
            batch_size=64
        )
        
        if similarity is not None:
            print(f"\n✅ SUCCESS! Cosine similarity computed: {similarity:.3f}")
            return True
        else:
            print(f"\n❌ Returned None")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_why_not_saved():
    """Check why computed values aren't being saved."""
    
    print("\n" + "="*70)
    print("CHECKING EXCEL EXPORT FLOW")
    print("="*70)
    
    print("""
The flow should be:

1. compare_session_graph_to_validation() is called with embedding_enable=True
2. Inside that, _compute_edge_level_context_insensitive_metrics() is called
3. That function:
   - Queries Neo4j for edges
   - Gets motivation + citations
   - Fetches citation texts
   - Chunks citation texts
   - Embeds motivation + chunks
   - Computes cosine similarity
   - Returns dict with ci metrics per edge

4. export_edges_comparison_to_excel() is called
5. It calls _compute_edge_level_context_insensitive_metrics() AGAIN (line 6439)
6. Results are added to Excel rows (line 6620-6640)

ISSUE: If step 5 fails (exception), it silently returns {} (line 6447)
       Then all CI metrics including cosine_similarity are None/NaN

DIAGNOSIS NEEDED:
   - Is _compute_edge_level_context_insensitive_metrics() being called?
   - Is it throwing an exception?
   - Is the exception being caught and silenced?
   - What specific error is occurring?
    """)


if __name__ == "__main__":
    print("="*70)
    print("COSINE SIMILARITY DIAGNOSTIC")
    print("="*70)
    
    # Test 1: Can we compute cosine similarity at all?
    success = test_cosine_similarity()
    
    # Test 2: Explain the flow
    check_why_not_saved()
    
    if success:
        print("\n" + "="*70)
        print("✅ CONCLUSION: Cosine similarity CAN be computed")
        print("="*70)
        print("""
Likely issues:
1. Exception in _compute_edge_level_context_insensitive_metrics() being silenced (line 6447)
2. Citation fetch failing (403 errors, timeouts)
3. Neo4j query returning empty citations
4. OPENAI_API_KEY not available during experiment run

RECOMMENDED FIX:
Replace line 6447 in modules.py:
    except Exception:
        edge_ci_metrics = {}

With:
    except Exception as e:
        logger.error(f"CI metrics computation failed: {e}")
        logger.exception("Full traceback:")
        edge_ci_metrics = {}

This will show us what's actually failing.
        """)
    else:
        print("\n" + "="*70)
        print("❌ CONCLUSION: Cosine similarity computation failed")
        print("="*70)
        print("Check OPENAI_API_KEY and network connectivity")