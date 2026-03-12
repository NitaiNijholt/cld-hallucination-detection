"""Lightweight evaluation utilities (no torch/transformers deps)."""

import re

VERDICT_LABELS = ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
INCORRECT = "INCORRECT"


def parse_verdict(text: str) -> str:
    """Extract VERDICT token from model output."""
    m = re.search(
        r"VERDICT\s*[:=]?\s*(CORRECT|PARTIALLY_CORRECT|INCORRECT)",
        text.upper(),
    )
    return m.group(1) if m else "UNKNOWN"
