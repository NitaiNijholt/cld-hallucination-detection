"""
Simple demonstration of Pydantic models for structured CoT outputs
No API calls required - just shows the models and validates them
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import json


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED CoT
# ============================================================================

class CorrectnessJudgmentCoT(BaseModel):
    """
    Structured output for correctness judgment with Chain-of-Thought.
    
    KEY INSIGHT: Field order matters for CoT reasoning!
    The model must generate reasoning BEFORE the verdict.
    Research shows this increases accuracy by 60% (GSM8k benchmark).
    """
    
    # STEP 1: Chain of Thought reasoning (comes FIRST)
    chain_of_thought: str = Field(
        description=(
            "Step-by-step reasoning process evaluating the causal explanation. "
            "Consider: (1) logical consistency, (2) temporal ordering, "
            "(3) mechanism specificity, (4) scientific plausibility"
        )
    )
    
    # STEP 2: Structured analysis
    strengths: list[str] = Field(
        description="Key strengths of the causal explanation"
    )
    
    weaknesses: list[str] = Field(
        description="Key weaknesses or gaps in the causal explanation"
    )
    
    # STEP 3: Final verdict (comes LAST after reasoning)
    verdict: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"] = Field(
        description="Final judgment on the causal explanation quality"
    )
    
    score: float = Field(
        description="Numerical score: 1.0 for CORRECT, 0.5 for PARTIALLY_CORRECT, 0.0 for INCORRECT",
        ge=0.0,
        le=1.0
    )
    
    reason: str = Field(
        description="One-sentence summary of the judgment"
    )


class CitationJudgmentCoT(BaseModel):
    """Structured output for citation verification with Chain-of-Thought"""
    
    # STEP 1: Reasoning first
    chain_of_thought: str = Field(
        description=(
            "Step-by-step analysis of how well citations support the causal claim. "
            "Consider: (1) evidence for mechanisms, (2) temporality evidence, "
            "(3) strength of association, (4) study design quality"
        )
    )
    
    # STEP 2: Evidence assessment
    evidence_for_claim: list[str] = Field(
        description="Specific evidence from citations that supports the claim"
    )
    
    evidence_against_claim: list[str] = Field(
        description="Evidence from citations that contradicts or weakens the claim"
    )
    
    missing_evidence: list[str] = Field(
        description="What key evidence is missing from the citations"
    )
    
    # STEP 3: Final verdict
    verdict: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"] = Field(
        description="Final judgment on citation support"
    )
    
    score: float = Field(
        description="Numerical score: 1.0 for SUPPORTED, 0.5 for PARTIALLY_SUPPORTED, 0.0 for NOT_SUPPORTED",
        ge=0.0,
        le=1.0
    )
    
    reason: str = Field(
        description="One-sentence summary of the judgment"
    )


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_correctness_model():
    """Demonstrate the correctness judgment model with validation"""
    
    print("=" * 80)
    print("DEMO 1: Correctness Judgment with Structured CoT")
    print("=" * 80)
    
    # Example: Valid judgment
    print("\n✅ Example 1: Valid Judgment\n" + "-" * 80)
    
    valid_judgment = {
        "chain_of_thought": (
            "First, I assess logical consistency: The explanation provides a clear "
            "causal pathway from sleep deprivation to reduced cognitive performance. "
            "Second, temporal ordering: The explanation correctly positions sleep "
            "deprivation as preceding cognitive decline. "
            "Third, mechanism specificity: It identifies specific neural mechanisms "
            "(neural plasticity, prefrontal cortex activity). "
            "Fourth, scientific plausibility: These mechanisms are well-established "
            "in neuroscience literature. The explanation is comprehensive and scientifically sound."
        ),
        "strengths": [
            "Identifies specific neural mechanism (prefrontal cortex)",
            "Explains both the pathway (neural plasticity) and effects (attention, memory)",
            "Respects temporal causality (deprivation → impairment)"
        ],
        "weaknesses": [
            "Could provide more detail on dose-response relationship",
            "Does not address potential confounders (e.g., stress, caffeine)"
        ],
        "verdict": "CORRECT",
        "score": 1.0,
        "reason": "Provides mechanistically specific, temporally sound causal explanation with clear neural pathways"
    }
    
    try:
        judgment = CorrectnessJudgmentCoT(**valid_judgment)
        print("✅ Validation passed!")
        print(f"\nVerdict: {judgment.verdict}")
        print(f"Score: {judgment.score}")
        print(f"\nChain of Thought:\n{judgment.chain_of_thought}")
        print(f"\nStrengths: {', '.join(judgment.strengths)}")
        print(f"Weaknesses: {', '.join(judgment.weaknesses)}")
        
        print("\n📊 JSON Output:")
        print(json.dumps(judgment.model_dump(), indent=2))
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Example: Invalid judgment (will fail validation)
    print("\n\n❌ Example 2: Invalid Judgment (score out of range)\n" + "-" * 80)
    
    invalid_judgment = {
        "chain_of_thought": "Some reasoning",
        "strengths": ["Something good"],
        "weaknesses": [],
        "verdict": "CORRECT",
        "score": 1.5,  # INVALID: > 1.0
        "reason": "Test"
    }
    
    try:
        judgment = CorrectnessJudgmentCoT(**invalid_judgment)
        print("This shouldn't print - validation should fail")
    except ValidationError as e:
        print("✅ Pydantic correctly caught the error!")
        print(f"Error: {e.errors()[0]['msg']}")
        print(f"Field: {e.errors()[0]['loc']}")


def demo_citation_model():
    """Demonstrate the citation judgment model"""
    
    print("\n\n" + "=" * 80)
    print("DEMO 2: Citation Judgment with Structured CoT")
    print("=" * 80 + "\n")
    
    valid_citation = {
        "chain_of_thought": (
            "Analyzing citation support: Citation 1 provides meta-analytic evidence "
            "showing a significant negative effect (SMD = -0.62). This addresses "
            "strength of association. Citation 2 provides mechanistic evidence "
            "(BDNF, neurogenesis in hippocampus). Together, they cover both "
            "association strength and plausibility of mechanism. However, neither "
            "citation explicitly addresses temporality or rules out reverse causation."
        ),
        "evidence_for_claim": [
            "Meta-analysis shows significant reduction in depression (SMD = -0.62)",
            "BDNF mechanism provides biological plausibility",
            "Effect consistent across different exercise types"
        ],
        "evidence_against_claim": [],
        "missing_evidence": [
            "Longitudinal studies showing exercise precedes depression reduction",
            "Evidence ruling out reverse causation (depression → reduced exercise)",
            "Discussion of confounding variables (social support, outdoor exposure)"
        ],
        "verdict": "PARTIALLY_SUPPORTED",
        "score": 0.5,
        "reason": "Strong evidence for association and mechanism, but lacks temporality evidence"
    }
    
    try:
        citation = CitationJudgmentCoT(**valid_citation)
        print("✅ Validation passed!")
        print(f"\nVerdict: {citation.verdict}")
        print(f"Score: {citation.score}")
        print(f"\nChain of Thought:\n{citation.chain_of_thought}")
        print(f"\nEvidence For: {len(citation.evidence_for_claim)} items")
        print(f"Evidence Against: {len(citation.evidence_against_claim)} items")
        print(f"Missing Evidence: {len(citation.missing_evidence)} items")
        
        print("\n📊 JSON Output:")
        print(json.dumps(citation.model_dump(), indent=2))
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")


def compare_with_without_cot():
    """Compare structured models with and without CoT"""
    
    print("\n\n" + "=" * 80)
    print("DEMO 3: Comparison - With vs Without Chain of Thought")
    print("=" * 80)
    
    # Without CoT (old approach)
    class JudgmentNoCoT(BaseModel):
        verdict: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
        score: float
        reason: str
    
    # With CoT (new approach)
    # (Using CorrectnessJudgmentCoT defined above)
    
    print("\n❌ WITHOUT CoT (old approach):")
    print("-" * 80)
    no_cot = JudgmentNoCoT(
        verdict="CORRECT",
        score=1.0,
        reason="The explanation is good"
    )
    print(json.dumps(no_cot.model_dump(), indent=2))
    print("\n⚠️  Issues:")
    print("  - No transparency into reasoning process")
    print("  - Can't verify HOW the judgment was made")
    print("  - Lower accuracy (research: -60% on reasoning tasks)")
    print("  - Harder to debug errors")
    print("  - No structured breakdown of strengths/weaknesses")
    
    print("\n\n✅ WITH CoT (new approach):")
    print("-" * 80)
    with_cot = CorrectnessJudgmentCoT(
        chain_of_thought=(
            "I first check logical consistency - the explanation flows logically. "
            "Then temporal ordering - cause precedes effect. "
            "Then mechanism specificity - clear neural pathways identified. "
            "Finally, plausibility - mechanisms are scientifically established."
        ),
        strengths=["Clear mechanism", "Temporal ordering correct"],
        weaknesses=["Missing dose-response discussion"],
        verdict="CORRECT",
        score=1.0,
        reason="Mechanistically specific and scientifically sound explanation"
    )
    print(json.dumps(with_cot.model_dump(), indent=2))
    print("\n✅ Benefits:")
    print("  - Complete transparency into reasoning")
    print("  - Can verify each step of judgment")
    print("  - Higher accuracy (+60% on reasoning tasks)")
    print("  - Easier to debug and improve")
    print("  - Structured breakdown helps identify specific issues")
    print("  - 100% schema compliance with OpenAI structured outputs")


def show_schema():
    """Show the JSON schema for documentation"""
    
    print("\n\n" + "=" * 80)
    print("DEMO 4: JSON Schema for OpenAI Structured Outputs")
    print("=" * 80)
    
    print("\n📋 Correctness Judgment Schema:")
    print("-" * 80)
    schema = CorrectnessJudgmentCoT.model_json_schema()
    print(json.dumps(schema, indent=2))
    
    print("\n\n💡 Usage with OpenAI API:")
    print("-" * 80)
    print("""
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",  # Required for structured outputs
    messages=[
        {"role": "system", "content": "You are a precise judge..."},
        {"role": "user", "content": "Evaluate this explanation..."}
    ],
    response_format=CorrectnessJudgmentCoT,  # ← Pydantic model
)

# Get structured output (100% schema compliance)
judgment = completion.choices[0].message.parsed
print(judgment.verdict)  # Type-safe access!
print(judgment.chain_of_thought)
""")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STRUCTURED CHAIN-OF-THOUGHT WITH PYDANTIC")
    print("Demonstrating Models for LLM-as-Judge")
    print("=" * 80)
    
    print("\n📚 Research Findings:")
    print("  • Adding reasoning field increases accuracy by 60% (GSM8k)")
    print("  • Field order matters: reasoning BEFORE verdict")
    print("  • OpenAI structured outputs: 100% schema compliance")
    print("  • Instructor library: 3M+ downloads/month for this pattern")
    
    # Run demonstrations
    demo_correctness_model()
    demo_citation_model()
    compare_with_without_cot()
    show_schema()
    
    print("\n\n" + "=" * 80)
    print("✅ DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    
    print("\n📋 Summary:")
    print("  1. ✅ Pydantic models provide 100% schema compliance")
    print("  2. ✅ Chain-of-thought field increases accuracy significantly")
    print("  3. ✅ Field order matters (reasoning → analysis → verdict)")
    print("  4. ✅ Automatic validation catches errors")
    print("  5. ✅ Structured output ready for database storage")
    print("  6. ✅ Type-safe access to all fields")
    
    print("\n🎯 Recommendation:")
    print("  → Update your YAML prompts to use Pydantic models")
    print("  → Use OpenAI's structured outputs API")
    print("  → Include 'chain_of_thought' field FIRST")
    print("  → Add structured fields (strengths/weaknesses/evidence)")
    print("  → Model: gpt-4o-2024-08-06 or later")
    
    print("\n📂 Next Steps:")
    print("  1. Convert your YAML prompts to use these Pydantic models")
    print("  2. Update your evaluation code to use structured outputs")
    print("  3. Run comparative tests (baseline vs CoT vs Mechanistic CoT)")
    print("  4. Measure accuracy improvement with structured outputs")
    
    print("\n")
