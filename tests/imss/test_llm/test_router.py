"""Test LLM router JSON parsing."""

from imss.llm.router import parse_agent_json, strip_json_fences


def test_strip_json_fences_removes_backticks():
    raw = '```json\n{"action": "BUY"}\n```'
    assert strip_json_fences(raw) == '{"action": "BUY"}'


def test_strip_json_fences_handles_no_fences():
    raw = '{"action": "BUY"}'
    assert strip_json_fences(raw) == '{"action": "BUY"}'


def test_parse_agent_json_valid():
    text = '{"action": "BUY", "stock": "BBRI", "quantity": 1000, "confidence": 0.8, "reasoning": "strong earnings"}'
    result = parse_agent_json(text)
    assert result is not None
    assert result["action"] == "BUY"
    assert result["quantity"] == 1000


def test_parse_agent_json_with_fences():
    text = '```json\n{"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "wait"}\n```'
    result = parse_agent_json(text)
    assert result is not None
    assert result["action"] == "HOLD"


def test_parse_agent_json_missing_keys():
    text = '{"action": "BUY"}'
    result = parse_agent_json(text)
    assert result is None


def test_parse_agent_json_invalid_action():
    text = '{"action": "WAIT", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "x"}'
    result = parse_agent_json(text)
    assert result is None


def test_parse_agent_json_garbage_input():
    result = parse_agent_json("this is not json at all")
    assert result is None
