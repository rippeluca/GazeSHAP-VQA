from __future__ import annotations

import pytest

from src.models.reasoning_head import (
    ANSWER_OPEN,
    SYSTEM_PROMPT,
    answers_match,
    build_messages,
    extract_answer,
    generation_prompt,
    has_valid_format,
    normalise_gqa_answer,
    stage1_prompt,
)
from tests.fixtures import FakeProcessor

WELL_FORMED = "<think>The dog is left of the tree.</think><answer>dog</answer>"


# --- prompts ----------------------------------------------------------------


def test_messages_carry_the_system_prompt_and_one_image():
    messages = build_messages("What colour is the car?")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["text"] == SYSTEM_PROMPT
    assert sum(c["type"] == "image" for c in messages[1]["content"]) == 1
    assert "What colour is the car?" in messages[1]["content"][1]["text"]


def test_stage1_prompt_ends_with_the_answer_tag():
    """The prompt stops at <answer>; the dataset appends the answer tokens."""
    prompt = stage1_prompt(FakeProcessor(), "Is it raining?")
    assert prompt.endswith(ANSWER_OPEN)


def test_generation_prompt_does_not_open_the_answer_tag():
    assert not generation_prompt(FakeProcessor(), "Is it raining?").endswith(ANSWER_OPEN)


# --- extraction -------------------------------------------------------------


def test_extract_answer():
    assert extract_answer(WELL_FORMED) == "dog"
    assert extract_answer("<answer>  left  </answer>") == "left"
    assert extract_answer("<answer>on the\nright</answer>") == "on the\nright"


@pytest.mark.parametrize(
    "completion",
    [
        "no tags at all",
        "<think>only thinking</think>",
        "<answer></answer>",
        "<answer>   </answer>",
        "<answer>unclosed",
    ],
)
def test_extract_answer_returns_none_on_bad_input(completion):
    assert extract_answer(completion) is None


def test_extract_answer_takes_the_first_of_several():
    assert extract_answer("<answer>a</answer><answer>b</answer>") == "a"


# --- format compliance ------------------------------------------------------


def test_valid_format_accepted():
    assert has_valid_format(WELL_FORMED)
    assert has_valid_format("\n<think>\nreasoning\n</think>\n<answer>yes</answer>\n")


@pytest.mark.parametrize(
    "completion",
    [
        "<answer>yes</answer>",  # missing think block
        "<think>reasoning</think>",  # missing answer
        "<answer>yes</answer><think>reasoning</think>",  # wrong order
        "<think>a</think><think>b</think><answer>y</answer>",  # duplicated tags
        "<think>a</think><answer>y</answer> trailing text",
        "prefix <think>a</think><answer>y</answer>",
        "",
    ],
)
def test_invalid_format_rejected(completion):
    assert not has_valid_format(completion)


def test_nested_answer_tag_is_rejected():
    assert not has_valid_format("<think>a</think><answer>y<answer>z</answer></answer>")


# --- GQA normalisation ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A dog.", "dog"),
        ("the tree", "tree"),
        ("An apple!", "apple"),
        ("  YES  ", "yes"),
        ("on the left", "on the left"),
        ("skateboard", "skateboard"),
        ("", ""),
        ("...", ""),
        ("the the dog", "dog"),
    ],
)
def test_normalise_gqa_answer(raw, expected):
    assert normalise_gqa_answer(raw) == expected


def test_articles_are_only_stripped_from_the_front():
    assert normalise_gqa_answer("man with a hat") == "man with a hat"


# --- matching ---------------------------------------------------------------


def test_answers_match_under_normalisation():
    assert answers_match("A dog.", "dog")
    assert answers_match("Yes", "yes")
    assert not answers_match("cat", "dog")


def test_missing_answer_never_matches():
    assert not answers_match(None, "dog")
    assert not answers_match(None, "")
