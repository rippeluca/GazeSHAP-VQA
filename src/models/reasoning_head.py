"""Prompt construction and answer parsing for the ``<think>``/``<answer>`` format.

Stage 1 never generates. The prompt ends on the opening ``<answer>`` tag and
the ground-truth answer is teacher-forced after it. Attribution targets the sum
of the logits of every answer token. Stage 2 uses the same system prompt but
lets the model produce the whole chain.
"""

from __future__ import annotations

import re
import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import ProcessorMixin

SYSTEM_PROMPT = (
    "You are a careful visual reasoner. When answering a question about an image, "
    "first describe your visual reasoning step by step inside <think>...</think> "
    "tags, focusing only on the image regions relevant to the question. Then give "
    "your final answer inside <answer>...</answer> tags, using as few words as "
    "possible."
)

USER_TEMPLATE = "Question: {question}"

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
FORMAT_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)

GQA_ARTICLES = frozenset({"a", "an", "the"})


def build_messages(question: str) -> list[dict]:
    """Build the chat messages for one question.

    Args:
        question: The natural-language question.

    Returns:
        Messages in the format ``processor.apply_chat_template`` expects, with a
        single image placeholder in the user turn.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_TEMPLATE.format(question=question)},
            ],
        },
    ]


def stage1_prompt(processor: ProcessorMixin, question: str) -> str:
    """Render the Stage 1 prompt, ending with the opening ``<answer>`` tag.

    The ground-truth answer tokens are appended by the dataset, so each answer
    token has a logit without a reasoning chain that Stage 1 has no supervision
    for.

    Args:
        processor: A Qwen3-VL processor with a chat template.
        question: The natural-language question.

    Returns:
        The rendered prompt string.
    """
    prefix = processor.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True
    )
    return f"{prefix}{ANSWER_OPEN}"


def generation_prompt(processor: ProcessorMixin, question: str) -> str:
    """Render the Stage 2 prompt, stopping before the model's turn content.

    Args:
        processor: A Qwen3-VL processor with a chat template.
        question: The natural-language question.

    Returns:
        The rendered prompt string.
    """
    return processor.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True
    )


def extract_answer(completion: str) -> str | None:
    """Pull the answer out of a completion.

    Args:
        completion: Raw model output.

    Returns:
        The answer text, or ``None`` if the tags are missing or empty.
    """
    match = ANSWER_PATTERN.search(completion)
    if match is None:
        return None
    answer = match.group(1).strip()
    return answer or None


def has_valid_format(completion: str) -> bool:
    """Whether a completion matches ``<think>...</think><answer>...</answer>``.

    Args:
        completion: Raw model output.

    Returns:
        ``True`` if the tag structure is well formed and each tag appears once.
    """
    if FORMAT_PATTERN.match(completion) is None:
        return False
    counts = (
        completion.count(THINK_OPEN),
        completion.count(THINK_CLOSE),
        completion.count(ANSWER_OPEN),
        completion.count(ANSWER_CLOSE),
    )
    return counts == (1, 1, 1, 1)


def normalise_gqa_answer(answer: str) -> str:
    """Apply GQA's answer normalisation.

    Lowercases, strips punctuation and drops leading articles, so that
    ``"A dog."`` and ``"dog"`` compare equal.

    Args:
        answer: Raw answer string.

    Returns:
        The normalised answer.
    """
    text = answer.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t]
    while tokens and tokens[0] in GQA_ARTICLES:
        tokens.pop(0)
    return " ".join(tokens)


def answers_match(predicted: str | None, target: str) -> bool:
    """Compare a predicted answer to the ground truth under GQA normalisation.

    Args:
        predicted: Extracted answer, or ``None`` for a format violation.
        target: Ground-truth answer.

    Returns:
        ``True`` if they match after normalisation.
    """
    if predicted is None:
        return False
    return normalise_gqa_answer(predicted) == normalise_gqa_answer(target)
