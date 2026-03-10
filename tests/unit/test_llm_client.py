"""Tests for LLM abstraction layer."""

import json
import pytest
from unittest.mock import MagicMock, patch

from llm.base_client import (
    LLMConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMAPIError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMValidationError,
    ModelInfo,
)
from llm.response_validator import ResponseValidator
from llm.cost_tracker import CostTracker
from llm.retry_handler import RetryHandler, RetryConfig, CircuitBreakerState
from llm.prompt_manager import PromptManager


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_total_tokens_calculated(self):
        """Test that total_tokens is calculated from input + output."""
        response = LLMResponse(
            content="test",
            input_tokens=100,
            output_tokens=50,
        )
        assert response.total_tokens == 150

    def test_total_tokens_explicit(self):
        """Test explicit total_tokens value."""
        response = LLMResponse(
            content="test",
            input_tokens=100,
            output_tokens=50,
            total_tokens=200,
        )
        assert response.total_tokens == 200


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_defaults(self):
        """Test default config values."""
        config = LLMConfig()
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.system_prompt is None

    def test_custom_config(self):
        """Test custom config values."""
        config = LLMConfig(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.3,
            system_prompt="You are a financial analyst",
        )
        assert config.model == "claude-3-opus-20240229"
        assert config.temperature == 0.3


class TestResponseValidator:
    """Tests for ResponseValidator."""

    def test_extract_json_pure(self):
        """Test extracting pure JSON."""
        content = '{"key": "value"}'
        result = ResponseValidator.extract_json(content)
        assert json.loads(result) == {"key": "value"}

    def test_extract_json_markdown_fenced(self):
        """Test extracting JSON from markdown code fences."""
        content = '```json\n{"key": "value"}\n```'
        result = ResponseValidator.extract_json(content)
        assert json.loads(result) == {"key": "value"}

    def test_extract_json_embedded(self):
        """Test extracting JSON embedded in text."""
        content = 'Here is the result: {"score": 75, "recommendation": "BUY"} end.'
        result = ResponseValidator.extract_json(content)
        assert json.loads(result)["score"] == 75

    def test_parse_json(self):
        """Test parsing JSON content."""
        data = ResponseValidator.parse_json('{"a": 1}')
        assert data == {"a": 1}

    def test_parse_json_invalid(self):
        """Test parsing invalid JSON raises ValueError."""
        with pytest.raises(ValueError):
            ResponseValidator.parse_json("not json at all")

    def test_validate_required_fields(self):
        """Test required field validation."""
        data = {"a": 1, "b": 2}
        missing = ResponseValidator.validate_required_fields(data, ["a", "b", "c"])
        assert missing == ["c"]

    def test_validate_score_range_valid(self):
        """Test valid score passes through."""
        score = ResponseValidator.validate_score_range(75.0)
        assert score == 75.0

    def test_validate_score_range_clamp(self):
        """Test out-of-range score is clamped."""
        score = ResponseValidator.validate_score_range(150.0)
        assert score == 100.0

        score = ResponseValidator.validate_score_range(-20.0)
        assert score == 0.0

    def test_validate_score_range_invalid_type(self):
        """Test invalid type returns midpoint."""
        score = ResponseValidator.validate_score_range("not a number")
        assert score == 50.0

    def test_validate_agent_report(self):
        """Test agent report validation."""
        data = {
            "overall_score": 82,
            "findings": [
                {"category": "growth", "description": "Strong revenue", "severity": "info", "score": 85}
            ],
            "recommendation": "BUY",
            "confidence": 0.8,
        }
        result = ResponseValidator.validate_agent_report(data)
        assert result["overall_score"] == 82.0
        assert len(result["findings"]) == 1
        assert result["confidence"] == 0.8

    def test_validate_agent_report_missing_fields(self):
        """Test validation with missing fields uses defaults."""
        result = ResponseValidator.validate_agent_report({})
        assert result["overall_score"] == 50.0
        assert result["findings"] == []
        assert result["confidence"] == 0.5

    def test_validate_stop_loss_valid(self):
        """Test valid stop loss passes through."""
        stop = ResponseValidator.validate_stop_loss(8550.0, None, 9000.0)
        assert stop == 8550.0

    def test_validate_stop_loss_above_entry(self):
        """Test stop above entry is corrected."""
        stop = ResponseValidator.validate_stop_loss(9500.0, None, 9000.0)
        assert stop < 9000.0

    def test_validate_stop_loss_no_downward_move(self):
        """Test stop cannot move down."""
        stop = ResponseValidator.validate_stop_loss(8000.0, 8500.0, 9000.0)
        assert stop == 8500.0


class TestCostTracker:
    """Tests for CostTracker."""

    def test_record_and_get_daily_cost(self):
        """Test recording calls and getting daily cost."""
        tracker = CostTracker(daily_budget=10.0)
        response = LLMResponse(
            content="test",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
        )
        tracker.record(response, purpose="test")
        assert tracker.get_daily_cost() == 0.05

    def test_within_budget(self):
        """Test budget checking."""
        tracker = CostTracker(daily_budget=1.0)
        assert tracker.is_within_budget() is True

        for _ in range(100):
            response = LLMResponse(content="test", cost_usd=0.02)
            tracker.record(response)

        assert tracker.is_within_budget() is False

    def test_daily_summary(self):
        """Test daily summary generation."""
        tracker = CostTracker()
        response = LLMResponse(
            content="test",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        tracker.record(response, provider=LLMProvider.CLAUDE)
        summary = tracker.get_daily_summary()
        assert summary.total_calls == 1
        assert summary.total_cost_usd == 0.01


class TestRetryHandler:
    """Tests for retry logic."""

    def test_successful_call(self):
        """Test that successful call returns normally."""
        handler = RetryHandler(config=RetryConfig(max_retries=3))
        result = handler.execute(lambda: "success")
        assert result == "success"

    def test_retry_on_error(self):
        """Test that retries happen on retryable errors."""
        call_count = 0

        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMAPIError("temporary error")
            return "success"

        handler = RetryHandler(
            config=RetryConfig(max_retries=3, base_delay=0.01)
        )
        result = handler.execute(failing_then_success)
        assert result == "success"
        assert call_count == 3

    def test_exhausted_retries(self):
        """Test that exhausted retries raise error."""
        handler = RetryHandler(
            config=RetryConfig(max_retries=2, base_delay=0.01)
        )
        with pytest.raises(LLMAPIError):
            handler.execute(lambda: (_ for _ in ()).throw(LLMAPIError("fail")))

    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreakerState(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True
        assert cb.can_proceed() is False

    def test_circuit_breaker_closes_on_success(self):
        """Test circuit breaker closes after success."""
        cb = CircuitBreakerState(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True
        cb.record_success()
        assert cb.is_open is False


class TestPromptManager:
    """Tests for PromptManager."""

    def test_builtin_templates_exist(self):
        """Test that built-in templates are loaded."""
        pm = PromptManager()
        templates = pm.list_templates()
        assert "auditor_analysis" in templates
        assert "growth_analysis" in templates
        assert "sentiment_analysis" in templates

    def test_render_template(self):
        """Test rendering a template with variables."""
        pm = PromptManager()
        result = pm.render("auditor_analysis", financial_data="test data")
        assert "test data" in result

    def test_render_missing_template(self):
        """Test rendering missing template raises KeyError."""
        pm = PromptManager()
        with pytest.raises(KeyError):
            pm.render("nonexistent_template")

    def test_register_custom_template(self):
        """Test registering custom templates."""
        pm = PromptManager()
        pm.register("custom", "Hello $name")
        result = pm.render("custom", name="World")
        assert result == "Hello World"

    def test_get_template(self):
        """Test getting raw template string."""
        pm = PromptManager()
        template = pm.get_template("synthesis")
        assert "$analyst_reports" in template
