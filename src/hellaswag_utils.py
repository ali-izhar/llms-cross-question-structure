"""HellaSwag prompt builder."""

from __future__ import annotations

CHOICE_LETTERS = ["A", "B", "C", "D"]


def build_hellaswag_prompt(context: str, endings: list[str]) -> str:
    """A/B/C/D prompt in the style of mmlu_utils.build_mmlu_prompt (letter-only instruction, `Answer:` cue)."""
    if len(endings) != 4:
        raise ValueError(f"HellaSwag expects 4 endings, got {len(endings)}")

    prompt = (
        "Choose the most plausible continuation of the scenario below. "
        "Reply with ONLY the letter (A, B, C, or D).\n\n"
        f"Scenario: {context.strip()}\n"
    )
    for letter, ending in zip(CHOICE_LETTERS, endings):
        prompt += f"{letter}. {ending.strip()}\n"
    prompt += "Answer:"
    return prompt
