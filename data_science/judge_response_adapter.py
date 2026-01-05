"""
Backward-compatible adapter for converting structured Pydantic outputs to legacy dict format.

This allows the existing codebase to work unchanged while using structured outputs internally.
"""

import json
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel
from judge_models import CorrectnessJudgment, CitationJudgment


class JudgeResponseAdapter:
    """Converts between Pydantic models and legacy dict format"""
    
    @staticmethod
    def parse_structured_response(
        response: Dict[str, Any],
        model_class: Type[BaseModel]
    ) -> BaseModel:
        """
        Parse OpenAI structured output into Pydantic model.
        
        Args:
            response: Raw OpenAI API response dict
            model_class: Pydantic model class to parse into
            
        Returns:
            Parsed Pydantic model instance
        """
        content = response['choices'][0]['message']['content']
        return model_class(**json.loads(content))
    
    @staticmethod
    def to_legacy_format(judgment: BaseModel) -> Dict[str, Any]:
        """
        Convert Pydantic model to legacy dict format.
        
        This ensures existing code continues to work unchanged - it still gets
        the expected verdict, score, reason fields.
        
        Args:
            judgment: Pydantic model instance (CorrectnessJudgment or CitationJudgment)
            
        Returns:
            Dict with keys: verdict, score, reason
        """
        return {
            "verdict": judgment.verdict,
            "score": judgment.score,
            "reason": judgment.reason
        }
    
    @staticmethod
    def supports_structured_outputs(model_name: str) -> bool:
        """
        Check if model supports OpenAI structured outputs.
        
        Structured outputs require:
        - gpt-4o-2024-08-06 or later
        - gpt-4o-mini (any version)
        
        Args:
            model_name: OpenAI model name
            
        Returns:
            True if model supports structured outputs
        """
        compatible_prefixes = [
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-08",
            "gpt-4o-2024-09",
            "gpt-4o-2024-10",
            "gpt-4o-2024-11",
            "gpt-4o-2024-12",
            "gpt-4o-2025",  # Future-proof
            "gpt-4o-mini"
        ]
        return any(model_name.startswith(prefix) for prefix in compatible_prefixes)
