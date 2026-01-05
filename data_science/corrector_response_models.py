"""
Pydantic models for structured corrector responses.
Ensures LLMs output properly formatted correction decisions.

Used with OpenAI's structured outputs feature to guarantee valid JSON responses
that match the expected schema, eliminating parsing errors.
"""

from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class CorrectorAction(str, Enum):
    """Available correction actions"""
    REVISE = "revise"           # Revise motivation text (keep edge type)
    CHANGE_TYPE = "change_type"  # Change edge type (POSITIVE/NEGATIVE/NONE)
    NONE = "none"               # No correction needed


class EdgeType(str, Enum):
    """Edge type options for causal relationships"""
    POSITIVE = "POSITIVE"   # Positive causal relationship
    NEGATIVE = "NEGATIVE"   # Negative causal relationship
    NONE = "NONE"          # No causal relationship


class CorrectorResponse(BaseModel):
    """
    Structured response for causal edge correction.
    
    The corrector analyzes judge feedback and decides how to fix problematic edges.
    """
    # Required: Reasoning for the decision (Chain-of-Thought)
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Step-by-step reasoning explaining why this correction is needed"
    )
    
    # Required: The action to take
    action: CorrectorAction = Field(
        ...,
        description="The correction action: 'revise' to fix motivation, 'change_type' to change edge type, 'none' if no correction needed"
    )
    
    # Optional: New motivation text (required for revise and change_type)
    new_motivation: str | None = Field(
        None,
        max_length=2000,
        description="New motivation text explaining the causal relationship (required for revise/change_type actions)"
    )
    
    # Optional: New edge type (required for change_type action)
    new_type: EdgeType | None = Field(
        None,
        description="New edge type: POSITIVE, NEGATIVE, or NONE (required for change_type action)"
    )
    
    class Config:
        use_enum_values = True


class CorrectorResponseWithConfidence(CorrectorResponse):
    """
    Extended corrector response with confidence score.
    Useful for filtering low-confidence corrections.
    """
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the correction decision (0.0 to 1.0)"
    )


# Example usage and validation
if __name__ == "__main__":
    # Test revise action
    revise_example = CorrectorResponse(
        reasoning="The current motivation lacks mechanistic detail. The judge noted that the causal pathway is unclear.",
        action=CorrectorAction.REVISE,
        new_motivation="Increased physical activity leads to higher basal metabolic rate through enhanced muscle mass and mitochondrial density.",
        new_type=None
    )
    print("Revise Example:")
    print(revise_example.model_dump_json(indent=2))
    
    # Test change_type action
    change_type_example = CorrectorResponse(
        reasoning="The edge is marked as NONE but the judge found evidence of a causal relationship. Changing to POSITIVE.",
        action=CorrectorAction.CHANGE_TYPE,
        new_motivation="Social norms influence individual BMI through peer pressure and behavioral conformity.",
        new_type=EdgeType.POSITIVE
    )
    print("\n\nChange Type Example:")
    print(change_type_example.model_dump_json(indent=2))
    
    # Test none action
    none_example = CorrectorResponse(
        reasoning="The current edge is correctly specified. The judge verdict is CORRECT with high confidence.",
        action=CorrectorAction.NONE,
        new_motivation=None,
        new_type=None
    )
    print("\n\nNone Example:")
    print(none_example.model_dump_json(indent=2))
    
    # Verify JSON schema for OpenAI
    print("\n\nJSON Schema for OpenAI structured outputs:")
    print(CorrectorResponse.model_json_schema())




















