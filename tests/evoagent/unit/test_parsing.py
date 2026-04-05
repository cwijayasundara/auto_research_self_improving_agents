"""Tests for LLM JSON response parsing."""

from evoagent.core.parsing import parse_llm_json


def test_parse_plain_json():
    assert parse_llm_json('{"score": 0.8}') == {"score": 0.8}


def test_parse_markdown_json_block():
    raw = '```json\n{"score": 0.8, "reasoning": "good"}\n```'
    result = parse_llm_json(raw)
    assert result == {"score": 0.8, "reasoning": "good"}


def test_parse_markdown_block_no_lang():
    raw = '```\n{"score": 0.5}\n```'
    assert parse_llm_json(raw) == {"score": 0.5}


def test_parse_invalid_json_returns_empty():
    assert parse_llm_json("not json at all") == {}


def test_parse_empty_string():
    assert parse_llm_json("") == {}


def test_parse_with_trailing_text():
    raw = '```json\n{"score": 0.9}\n```\nSome trailing text'
    result = parse_llm_json(raw)
    assert result["score"] == 0.9
