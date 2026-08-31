"""Implicit-only CoT mode utilities for latent reasoning training."""

from typing import Dict

IMPLICIT_FLAGS = {
    "enable_latent_reasoning": True,
    "emit_thinking_tokens": False,
    "use_iterative_forward": True,
    "generate_thinking": True,
    "reasoning_stage": 4,
}


def get_implicit_flags() -> Dict[str, object]:
    """Return the fixed flag set for implicit latent reasoning."""
    return dict(IMPLICIT_FLAGS)
