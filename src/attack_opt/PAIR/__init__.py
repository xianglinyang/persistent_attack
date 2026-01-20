"""
PAIR Attack Module
==================

Prompt Automatic Iterative Refinement for evading guardrails.
"""

from .main import (
    pair_attack_exposure,
    pair_attack_trigger,
    pair_attack_multiple_triggers,
    main_pair_experiment,
    prompt_template,
    extract_guard_feedback,
    format_attack_history,
    pair_refine_prompt,
    pair_attack_single_iteration,
)

__all__ = [
    "pair_attack_exposure",
    "pair_attack_trigger",
    "pair_attack_multiple_triggers",
    "main_pair_experiment",
    "prompt_template",
    "extract_guard_feedback",
    "format_attack_history",
    "pair_refine_prompt",
    "pair_attack_single_iteration",
]
