"""
Pydantic models for structured LLM judge outputs.

These models match the exact output format of current prompts (VERDICT, SCORE, REASON)
but order fields as REASON → SCORE → VERDICT to encourage reasoning-first generation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class CorrectnessJudgment(BaseModel):
    """
    Structured output for correctness judgments.
    
    Matches prompt format: VERDICT, SCORE, REASON
    Field order: reason → score → verdict (reasoning-first for better accuracy)
    """
    model_config = ConfigDict(extra="forbid")  # Required for OpenAI structured outputs
    
    reason: str = Field(description="Brief explanation of your judgment")
    score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="1.0 for CORRECT, 0.5 for PARTIALLY_CORRECT, 0.0 for INCORRECT"
    )
    verdict: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]


class CitationJudgment(BaseModel):
    """
    Structured output for citation judgments.
    
    Matches prompt format: VERDICT, SCORE, REASON
    Field order: reason → score → verdict (reasoning-first for better accuracy)
    """
    model_config = ConfigDict(extra="forbid")  # Required for OpenAI structured outputs
    
    reason: str = Field(description="Brief explanation of your judgment")
    score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="1.0 for SUPPORTED, 0.5 for PARTIALLY_SUPPORTED, 0.0 for NOT_SUPPORTED"
    )
    verdict: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]
