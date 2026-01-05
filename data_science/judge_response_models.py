"""
Pydantic models for structured judge responses.
Ensures LLMs output properly formatted verdicts.
"""

from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class CorrectnessVerdict(str, Enum):
    """Verdict options for correctness-based judging"""
    CORRECT = "CORRECT"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    INCORRECT = "INCORRECT"


class CitationVerdict(str, Enum):
    """Verdict options for citation-based judging"""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNADDRESSED = "UNADDRESSED"


class CorrectnessJudgeResponse(BaseModel):
    """
    Structured response for correctness-based judging.
    
    The judge evaluates causal reasoning quality without requiring citations.
    """
    # Optional: Allow Chain-of-Thought reasoning steps
    reasoning_steps: str | None = Field(
        None,
        description="Optional: Step-by-step reasoning process (for CoT prompts)"
    )
    
    verdict: CorrectnessVerdict = Field(
        ...,
        description="Final verdict on the causal explanation quality"
    )
    
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numeric score: 1.0 for CORRECT, 0.5 for PARTIALLY_CORRECT, 0.0 for INCORRECT"
    )
    
    reason: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief explanation of the judgment (1-3 sentences)"
    )
    
    class Config:
        use_enum_values = True


class CitationJudgeResponse(BaseModel):
    """
    Structured response for citation-based judging.
    
    The judge evaluates whether scientific claims are supported by cited sources.
    """
    # Optional: Allow Chain-of-Thought reasoning steps
    reasoning_steps: str | None = Field(
        None,
        description="Optional: Step-by-step reasoning process (for CoT prompts)"
    )
    
    verdict: CitationVerdict = Field(
        ...,
        description="Final verdict on citation support"
    )
    
    reason: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief explanation of the judgment (1-3 sentences)"
    )
    
    class Config:
        use_enum_values = True


# Example usage and validation
if __name__ == "__main__":
    # Test correctness response
    correctness_example = CorrectnessJudgeResponse(
        reasoning_steps="Step 1: Mechanism is clearly described\nStep 2: Logic is sound",
        verdict=CorrectnessVerdict.CORRECT,
        score=1.0,
        reason="The causal mechanism is well-defined and logically coherent."
    )
    print("Correctness Example:")
    print(correctness_example.model_dump_json(indent=2))
    
    # Test citation response
    citation_example = CitationJudgeResponse(
        reasoning_steps="Step 1: Citations mention the variables\nStep 2: Evidence is indirect",
        verdict=CitationVerdict.PARTIALLY_SUPPORTED,
        reason="Citations provide correlational evidence but lack mechanistic support."
    )
    print("\n\nCitation Example:")
    print(citation_example.model_dump_json(indent=2))
