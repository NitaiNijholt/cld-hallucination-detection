"""
Shared Pydantic models for LLM-as-Judge experiments.

These models enforce structured outputs for both:
- Correctness judgments (causal reasoning quality)
- Factual accuracy judgments (hallucination detection)

Usage:
    from alternative_prompts.judge_models import BaselineJudgment, CorrectnessJudgmentCoT
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


# ============================================================================
# BASELINE MODELS (No Chain-of-Thought)
# ============================================================================

class BaselineJudgment(BaseModel):
    """
    Baseline judgment model without Chain-of-Thought.
    
    Used for:
    - Correctness baseline (causal reasoning)
    - TruthQA baseline (factual accuracy)
    
    This is the simplest structured output format for fair comparison
    across factual and causal evaluation tasks.
    """
    model_config = ConfigDict(extra="forbid")  # Required for OpenAI structured outputs
    
    verdict: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"] = Field(
        description="The judgment: CORRECT, PARTIALLY_CORRECT, or INCORRECT"
    )
    
    score: Literal[0.0, 0.5, 1.0] = Field(
        description="Numerical score: 1.0 for CORRECT, 0.5 for PARTIALLY_CORRECT, 0.0 for INCORRECT"
    )
    
    reason: str = Field(
        description="Brief explanation of the judgment"
    )


# ============================================================================
# CHAIN-OF-THOUGHT MODELS
# ============================================================================

class CorrectnessJudgmentCoT(BaseModel):
    """
    Structured output for correctness judgment with Chain-of-Thought.
    
    KEY INSIGHT: Field order matters for CoT reasoning!
    The model must generate reasoning BEFORE the verdict.
    Research shows this increases accuracy by 60% (GSM8k benchmark).
    
    Used for: Evaluating causal reasoning quality in CLDs
    """
    model_config = ConfigDict(extra="forbid")  # Required for OpenAI structured outputs
    
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
    
    score: Literal[0.0, 0.5, 1.0] = Field(
        description="Numerical score: 1.0 for CORRECT, 0.5 for PARTIALLY_CORRECT, 0.0 for INCORRECT"
    )
    
    reason: str = Field(
        description="One-sentence summary of the judgment"
    )


class CitationJudgmentCoT(BaseModel):
    """Structured output for citation verification with Chain-of-Thought"""
    model_config = ConfigDict(extra="forbid")  # Required for OpenAI structured outputs
    
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
    
    score: Literal[0.0, 0.5, 1.0] = Field(
        description="Numerical score: 1.0 for SUPPORTED, 0.5 for PARTIALLY_SUPPORTED, 0.0 for NOT_SUPPORTED"
    )
    
    reason: str = Field(
        description="One-sentence summary of the judgment"
    )


# ============================================================================
# LEGACY ALIASES (for backward compatibility)
# ============================================================================

# Alias for TruthQA scripts
TruthQAJudgment = BaselineJudgment

# Alias for correctness scripts  
CorrectnessJudgment = BaselineJudgment
