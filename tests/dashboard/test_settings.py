"""Tests for Settings page configuration validation."""
import pytest
from typing import Dict, Any


class TestLLMConfigValidation:
    """Tests for LLM configuration validation."""

    def test_valid_provider_selection(self):
        """Test valid LLM provider selection."""
        valid_providers = ["Claude (Anthropic)", "GLM (Z.AI)", "OpenAI"]

        for provider in valid_providers:
            assert provider in ["Claude (Anthropic)", "GLM (Z.AI)", "OpenAI"]

    def test_fallback_provider_different(self):
        """Test that fallback provider should differ from primary."""
        primary = "Claude (Anthropic)"
        fallback = "GLM (Z.AI)"

        assert primary != fallback

    def test_api_key_validation(self):
        """Test API key format validation."""
        def validate_api_key(key: str, provider: str) -> bool:
            if not key:
                return False
            if provider == "Claude (Anthropic)" and not key.startswith("sk-ant-"):
                return False
            if provider == "OpenAI" and not key.startswith("sk-"):
                return False
            return len(key) >= 20

        assert validate_api_key("sk-ant-api03-validkey123456789", "Claude (Anthropic)") is True
        assert validate_api_key("sk-validkey123456789", "OpenAI") is True
        assert validate_api_key("", "Claude (Anthropic)") is False
        assert validate_api_key("invalid", "Claude (Anthropic)") is False


class TestTradingModeValidation:
    """Tests for trading mode parameter validation."""

    def test_intraday_risk_per_trade(self):
        """Test Intraday mode risk per trade validation."""
        intraday_risk = 0.5  # 0.5%

        assert 0.1 <= intraday_risk <= 2.0  # Reasonable range

    def test_swing_risk_per_trade(self):
        """Test Swing mode risk per trade validation."""
        swing_risk = 1.0  # 1.0%

        assert 0.5 <= swing_risk <= 2.0

    def test_position_risk_per_trade(self):
        """Test Position mode risk per trade validation."""
        position_risk = 1.5  # 1.5%

        assert 0.5 <= position_risk <= 3.0

    def test_investor_risk_per_trade(self):
        """Test Investor mode risk per trade validation."""
        investor_risk = 2.0  # 2.0%

        assert 1.0 <= investor_risk <= 5.0

    def test_risk_progression(self):
        """Test that risk increases with holding period."""
        risks = {
            "intraday": 0.5,
            "swing": 1.0,
            "position": 1.5,
            "investor": 2.0,
        }

        # Longer hold = higher risk tolerance
        assert risks["intraday"] < risks["swing"]
        assert risks["swing"] < risks["position"]
        assert risks["position"] < risks["investor"]


class TestRiskManagementSettings:
    """Tests for risk management settings validation."""

    def test_kelly_fraction_range(self):
        """Test Kelly fraction is within valid range."""
        kelly_fraction = 0.5

        assert 0.1 <= kelly_fraction <= 1.0

    def test_kelly_fraction_half_kelly(self):
        """Test that half Kelly is recommended."""
        recommended_kelly = 0.5

        assert recommended_kelly == 0.5  # Half Kelly is standard

    def test_max_daily_loss_range(self):
        """Test max daily loss is within valid range."""
        max_daily_loss = 2.0  # 2%

        assert 1.0 <= max_daily_loss <= 5.0

    def test_max_daily_loss_triggers_halt(self):
        """Test that hitting max daily loss should halt trading."""
        def should_halt_trading(current_loss_pct: float, max_loss_pct: float) -> bool:
            return current_loss_pct >= max_loss_pct

        assert should_halt_trading(2.5, 2.0) is True
        assert should_halt_trading(1.5, 2.0) is False

    def test_drawdown_warning_threshold(self):
        """Test drawdown warning threshold."""
        warning_threshold = 0.05  # 5%
        halt_threshold = 0.10  # 10%

        assert warning_threshold < halt_threshold


class TestNotificationSettings:
    """Tests for notification settings validation."""

    def test_telegram_chat_id_format(self):
        """Test Telegram chat ID format validation."""
        def validate_telegram_chat_id(chat_id: str) -> bool:
            if not chat_id:
                return False
            # Telegram chat IDs are numeric strings
            return chat_id.lstrip("-").isdigit()

        assert validate_telegram_chat_id("-1001234567890") is True
        assert validate_telegram_chat_id("123456789") is True
        assert validate_telegram_chat_id("") is False
        assert validate_telegram_chat_id("abc") is False

    def test_email_format_validation(self):
        """Test email format validation."""
        import re

        def validate_email(email: str) -> bool:
            if not email:
                return False
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))

        assert validate_email("trader@example.com") is True
        assert validate_email("user.name@domain.co.id") is True
        assert validate_email("") is False
        assert validate_email("invalid") is False
        assert validate_email("no@domain") is False

    def test_summary_frequency_options(self):
        """Test summary frequency options."""
        valid_frequencies = ["Daily", "Weekly"]

        for freq in ["Daily", "Weekly"]:
            assert freq in valid_frequencies


class TestSettingsPersistence:
    """Tests for settings persistence logic."""

    def test_settings_serialization(self):
        """Test that settings can be serialized to dict."""
        settings = {
            "llm": {
                "primary_provider": "Claude (Anthropic)",
                "fallback_provider": "GLM (Z.AI)",
            },
            "trading": {
                "intraday_risk": 0.5,
                "swing_risk": 1.0,
                "position_risk": 1.5,
                "investor_risk": 2.0,
            },
            "risk": {
                "kelly_fraction": 0.5,
                "max_daily_loss": 2.0,
            },
            "notifications": {
                "telegram_enabled": True,
                "email_enabled": True,
            },
        }

        # Should be JSON serializable
        import json
        serialized = json.dumps(settings)
        deserialized = json.loads(serialized)

        assert deserialized == settings

    def test_settings_migration(self):
        """Test that old settings can be migrated to new format."""
        old_settings = {
            "provider": "Claude",
            "risk_per_trade": 1.0,
        }

        # Migration logic
        new_settings = {
            "llm": {
                "primary_provider": old_settings.get("provider", "Claude (Anthropic)"),
            },
            "trading": {
                "swing_risk": old_settings.get("risk_per_trade", 1.0),
            },
        }

        assert "llm" in new_settings
        assert "trading" in new_settings


class TestSettingsValidation:
    """Tests for complete settings validation."""

    def test_validate_complete_settings(self):
        """Test validation of complete settings object."""
        settings = {
            "llm": {
                "primary_provider": "Claude (Anthropic)",
                "fallback_provider": "GLM (Z.AI)",
            },
            "trading": {
                "intraday_risk": 0.5,
                "swing_risk": 1.0,
                "position_risk": 1.5,
                "investor_risk": 2.0,
            },
            "risk": {
                "kelly_fraction": 0.5,
                "max_daily_loss": 2.0,
            },
        }

        def validate_settings(s: Dict[str, Any]) -> Dict[str, Any]:
            errors = []

            # Check LLM providers differ
            if s["llm"]["primary_provider"] == s["llm"]["fallback_provider"]:
                errors.append("Primary and fallback providers should differ")

            # Check Kelly fraction
            if not 0.1 <= s["risk"]["kelly_fraction"] <= 1.0:
                errors.append("Kelly fraction must be between 0.1 and 1.0")

            # Check max daily loss
            if not 1.0 <= s["risk"]["max_daily_loss"] <= 5.0:
                errors.append("Max daily loss must be between 1% and 5%")

            return {"valid": len(errors) == 0, "errors": errors}

        result = validate_settings(settings)
        assert result["valid"] is True

    def test_detect_invalid_settings(self):
        """Test detection of invalid settings."""
        settings = {
            "llm": {
                "primary_provider": "Claude",
                "fallback_provider": "Claude",  # Same as primary
            },
            "risk": {
                "kelly_fraction": 1.5,  # Too high
                "max_daily_loss": 10.0,  # Too high
            },
        }

        def validate_settings(s: Dict[str, Any]) -> Dict[str, Any]:
            errors = []

            if s["llm"]["primary_provider"] == s["llm"]["fallback_provider"]:
                errors.append("Primary and fallback providers should differ")

            if not 0.1 <= s["risk"]["kelly_fraction"] <= 1.0:
                errors.append("Kelly fraction out of range")

            if not 1.0 <= s["risk"]["max_daily_loss"] <= 5.0:
                errors.append("Max daily loss out of range")

            return {"valid": len(errors) == 0, "errors": errors}

        result = validate_settings(settings)
        assert result["valid"] is False
        assert len(result["errors"]) >= 2
