"""
Spectral Hallucination Detection Package

A novel approach to detecting hallucinations in Large Language Models (LLMs)
based on spectral analysis of model internals, inspired by WeightWatcher's
Heavy-Tailed Self-Regularization theory.
"""

from .spectral_hallucination_detection import (
    SpectralHallucinationDetector,
    SpectralMetrics
)

__version__ = "1.0.0"
__all__ = ["SpectralHallucinationDetector", "SpectralMetrics"]