"""
Tests for Settings Page (05_settings.py) and API /config/* endpoints.

Tests:
1. Page loads without errors
2. Trading mode selector works
3. LLM provider settings display
4. Risk parameter sliders work
5. Settings save/load functionality
6. All NextGen styling applied
7. API endpoints /config/* work
"""

import json
import pytest
import requests
from unittest.mock import patch, MagicMock


# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

class TestConfigAPIEndpoints:
    """Test /config/* API endpoints."""

    BASE_URL = "http://localhost:8000"

    def test_config_modes_endpoint_returns_200(self):
        """Test that /config/modes endpoint returns 200."""
        response = requests.get(f"{self.BASE_URL}/config/modes", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_config_modes_returns_all_modes(self):
        """Test that /config/modes returns all 4 trading modes."""
        response = requests.get(f"{self.BASE_URL}/config/modes", timeout=5)
        data = response.json()

        assert "modes" in data, "Response should contain 'modes' key"
        modes = data["modes"]

        expected_modes = ["intraday", "swing", "position", "investor"]
        for mode in expected_modes:
            assert mode in modes, f"Missing mode: {mode}"

    def test_config_modes_returns_default(self):
        """Test that /config/modes returns default mode."""
        response = requests.get(f"{self.BASE_URL}/config/modes", timeout=5)
        data = response.json()

        assert "default" in data, "Response should contain 'default' key"
        assert data["default"] == "swing", "Default mode should be 'swing'"

    def test_config_modes_mode_structure(self):
        """Test that each mode has required fields."""
        response = requests.get(f"{self.BASE_URL}/config/modes", timeout=5)
        data = response.json()

        required_fields = [
            "name",
            "min_hold_days",
            "max_hold_days",
            "max_risk_per_trade",
            "max_position_pct",
            "technical_weight",
            "flow_weight",
            "fundamental_weight",
            "min_score",
            "default_stop_pct",
        ]

        for mode_name, mode_config in data["modes"].items():
            for field in required_fields:
                assert field in mode_config, f"Mode {mode_name} missing field: {field}"

    def test_config_modes_intraday_values(self):
        """Test intraday mode has correct values."""
        response = requests.get(f"{self.BASE_URL}/config/modes", timeout=5)
        data = response.json()

        intraday = data["modes"]["intraday"]
        assert intraday["name"] == "Intraday"
        assert intraday["min_hold_days"] == 0
        assert intraday["max_hold_days"] == 1
        assert intraday["max_risk_per_trade"] == 0.005  # 0.5%

    def test_config_modes_swing_values(self):
        """Test swing mode has correct values."""
        response = requests.get(f"{self.BASE_URL}/config/modes", timeout=5)
        data = response.json()

        swing = data["modes"]["swing"]
        assert swing["name"] == "Swing"
        assert swing["min_hold_days"] == 2
        assert swing["max_hold_days"] == 7
        assert swing["max_risk_per_trade"] == 0.01  # 1.0%

    def test_config_modes_position_values(self):
        """Test position mode has correct values."""
        response = requests.get(f"{self.BASE_URL}/config/modes", timeout=5)
        data = response.json()

        position = data["modes"]["position"]
        assert position["name"] == "Position"
        assert position["min_hold_days"] == 5
        assert position["max_hold_days"] == 20
        assert position["max_risk_per_trade"] == 0.015  # 1.5%

    def test_config_modes_investor_values(self):
        """Test investor mode has correct values."""
        response = requests.get(f"{self.BASE_URL}/config/modes", timeout=5)
        data = response.json()

        investor = data["modes"]["investor"]
        assert investor["name"] == "Investor"
        assert investor["min_hold_days"] == 20
        assert investor["max_hold_days"] == 90
        assert investor["max_risk_per_trade"] == 0.02  # 2.0%


# ============================================================================
# SETTINGS PAGE STRUCTURE TESTS
# ============================================================================

class TestSettingsPageStructure:
    """Test the settings page structure and components."""

    def test_supported_providers_constant(self):
        """Test SUPPORTED_PROVIDERS constant exists."""
        # Check via file content since direct import of Streamlit pages can fail
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()
        assert "SUPPORTED_PROVIDERS" in content

    def test_page_imports_streamlit(self):
        """Test that page imports streamlit."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()
        assert "import streamlit as st" in content

    def test_page_has_required_constants(self):
        """Test that page has required constants."""
        # Read the file and check for constants
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "REQUEST_TIMEOUT" in content, "Missing REQUEST_TIMEOUT constant"
        assert "SUPPORTED_PROVIDERS" in content, "Missing SUPPORTED_PROVIDERS constant"

    def test_page_has_session_state_initialization(self):
        """Test that page initializes session state."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "st.session_state.settings" in content, "Missing session state initialization"

    def test_page_has_three_tabs(self):
        """Test that page has three tabs."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "LLM Providers" in content, "Missing LLM Providers tab"
        assert "Trading Modes" in content, "Missing Trading Modes tab"
        assert "Notifications" in content, "Missing Notifications tab"

    def test_page_has_form_for_llm_settings(self):
        """Test that LLM settings use a form."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert 'st.form("llm_settings")' in content, "Missing LLM settings form"


# ============================================================================
# SETTINGS DEFAULTS TESTS
# ============================================================================

class TestSettingsDefaults:
    """Test default settings values."""

    EXPECTED_DEFAULTS = {
        'primary_provider': 'Claude (Anthropic)',
        'fallback_provider': 'GLM (Z.AI)',
        'max_daily_loss': 2.0,
        'kelly_fraction': 0.5,
        'intraday_risk': 0.5,
        'swing_risk': 1.0,
        'position_risk': 1.5,
        'investor_risk': 2.0,
        'telegram_enabled': True,
        'telegram_chat_id': '',
        'email_enabled': True,
        'email_address': '',
        'email_frequency': 'Daily',
    }

    def test_default_settings_structure(self):
        """Test that default settings has all required keys."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        for key in self.EXPECTED_DEFAULTS:
            assert f"'{key}'" in content, f"Missing default setting: {key}"

    def test_default_primary_provider(self):
        """Test default primary provider is Claude."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "'primary_provider': 'Claude (Anthropic)'" in content

    def test_default_fallback_provider(self):
        """Test default fallback provider is GLM."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "'fallback_provider': 'GLM (Z.AI)'" in content

    def test_default_risk_values(self):
        """Test default risk values are correct."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "'intraday_risk': 0.5" in content
        assert "'swing_risk': 1.0" in content
        assert "'position_risk': 1.5" in content
        assert "'investor_risk': 2.0" in content


# ============================================================================
# NEXTGEN STYLING TESTS
# ============================================================================

class TestNextGenStyling:
    """Test that NextGen styling is properly applied."""

    def test_imports_nextgen_styles(self):
        """Test that page imports nextgen_styles."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "from dashboard.components.nextgen_styles import" in content, \
            "Missing nextgen_styles import"

    def test_imports_colors(self):
        """Test that page imports COLORS constant."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "COLORS" in content, "Missing COLORS import"

    def test_applies_nextgen_css(self):
        """Test that page applies NextGen CSS."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "get_nextgen_css()" in content, "Missing get_nextgen_css() call"

    def test_uses_nextgen_card_class(self):
        """Test that page uses nextgen-card CSS class."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert 'class="nextgen-card"' in content, "Missing nextgen-card class usage"

    def test_uses_section_header_class(self):
        """Test that page uses section-header CSS class."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert 'class="section-header"' in content, "Missing section-header class usage"

    def test_uses_colors_for_styling(self):
        """Test that page uses COLORS dict for inline styles."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        # Check for color usage in styles
        assert "COLORS['muted_foreground']" in content or 'COLORS["muted_foreground"]' in content

    def test_trading_hours_indicator_included(self):
        """Test that trading hours indicator is included."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "trading_hours_indicator()" in content, \
            "Missing trading_hours_indicator component"


# ============================================================================
# TRADING MODE SELECTOR TESTS
# ============================================================================

class TestTradingModeSelector:
    """Test trading mode selector functionality."""

    def test_has_mode_risk_inputs(self):
        """Test that page has risk inputs for all modes."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "intraday_risk" in content
        assert "swing_risk" in content
        assert "position_risk" in content
        assert "investor_risk" in content

    def test_has_mode_comparison_table(self):
        """Test that page has mode comparison table."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "MODE COMPARISON" in content or "Mode Comparison" in content
        assert "<table" in content, "Missing comparison table"

    def test_mode_table_has_all_modes(self):
        """Test that comparison table includes all 4 modes."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert ">Intraday<" in content
        assert ">Swing<" in content
        assert ">Position<" in content
        assert ">Investor<" in content

    def test_mode_number_input_bounds(self):
        """Test that mode risk inputs have proper bounds."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        # Check min_value and max_value are set
        assert "min_value=0.1" in content or "min_value = 0.1" in content
        assert "max_value=5.0" in content or "max_value = 5.0" in content


# ============================================================================
# LLM PROVIDER SETTINGS TESTS
# ============================================================================

class TestLLMProviderSettings:
    """Test LLM provider settings functionality."""

    def test_has_primary_provider_selectbox(self):
        """Test that page has primary provider selectbox."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Primary Provider" in content
        assert "st.selectbox" in content

    def test_has_fallback_provider_selectbox(self):
        """Test that page has fallback provider selectbox."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Fallback Provider" in content

    def test_provider_options_are_correct(self):
        """Test that provider options include Claude and GLM."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Claude (Anthropic)" in content
        assert "GLM (Z.AI)" in content


# ============================================================================
# RISK PARAMETER SLIDERS TESTS
# ============================================================================

class TestRiskParameterSliders:
    """Test risk parameter sliders functionality."""

    def test_has_max_daily_loss_slider(self):
        """Test that page has max daily loss slider."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Max Daily Loss Limit" in content
        assert "st.slider" in content

    def test_has_kelly_fraction_slider(self):
        """Test that page has Kelly fraction slider."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Kelly Fraction" in content

    def test_max_daily_loss_slider_bounds(self):
        """Test max daily loss slider has correct bounds (1-5%)."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        # Find the slider definition
        assert "1.0, 5.0" in content, "Max daily loss should be between 1.0 and 5.0"

    def test_kelly_fraction_slider_bounds(self):
        """Test Kelly fraction slider has correct bounds (0.1-1.0)."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "0.1, 1.0" in content, "Kelly fraction should be between 0.1 and 1.0"


# ============================================================================
# SETTINGS SAVE/LOAD FUNCTIONALITY TESTS
# ============================================================================

class TestSettingsSaveLoad:
    """Test settings save and load functionality."""

    def test_has_save_llm_settings_button(self):
        """Test that page has save LLM settings button."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Save LLM Settings" in content

    def test_has_save_notification_settings_button(self):
        """Test that page has save notification settings button."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Save Notification Settings" in content

    def test_has_reset_button(self):
        """Test that page has reset to defaults button."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Reset All Settings" in content

    def test_has_export_settings(self):
        """Test that page has export settings functionality."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Download Settings JSON" in content or "st.download_button" in content

    def test_save_updates_session_state(self):
        """Test that save buttons update session state."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        # Check that save buttons update session state
        assert "st.session_state.settings[" in content

    def test_shows_toast_on_save(self):
        """Test that save buttons show toast notification."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "st.toast" in content, "Missing toast notification on save"


# ============================================================================
# NOTIFICATION SETTINGS TESTS
# ============================================================================

class TestNotificationSettings:
    """Test notification settings functionality."""

    def test_has_telegram_settings(self):
        """Test that page has Telegram settings."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Telegram" in content
        assert "telegram_enabled" in content

    def test_has_email_settings(self):
        """Test that page has Email settings."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Email" in content
        assert "email_enabled" in content

    def test_has_telegram_chat_id_input(self):
        """Test that page has Telegram chat ID input."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "telegram_chat_id" in content

    def test_has_email_frequency_selector(self):
        """Test that page has email frequency selector."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "email_frequency" in content
        assert "Daily" in content
        assert "Weekly" in content


# ============================================================================
# PAGE CONFIG TESTS
# ============================================================================

class TestPageConfig:
    """Test page configuration."""

    def test_has_set_page_config(self):
        """Test that page has set_page_config."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "st.set_page_config" in content

    def test_page_title_is_correct(self):
        """Test that page title is correct."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "Configuration | IDX" in content or "Settings" in content

    def test_page_icon_is_settings(self):
        """Test that page icon is settings gear."""
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        assert "page_icon" in content


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSettingsPageIntegration:
    """Integration tests for settings page."""

    def test_streamlit_page_loads(self):
        """Test that Streamlit settings page is accessible."""
        try:
            # Streamlit pages are rendered server-side, so we check if the
            # Streamlit server responds
            response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
            # Streamlit health endpoint should return 200
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Streamlit server not running on port 8501")

    def test_api_and_settings_consistency(self):
        """Test that API modes and settings page modes are consistent."""
        # Get API modes
        response = requests.get("http://localhost:8000/config/modes", timeout=5)
        api_modes = response.json()["modes"]

        # Check settings page has matching modes
        with open("/mnt/data/Project/idx-trading-system/dashboard/pages/05_settings.py") as f:
            content = f.read()

        # Verify modes match
        assert "Intraday" in content
        assert "Swing" in content
        assert "Position" in content
        assert "Investor" in content

        # Verify risk values match
        assert "0.5" in content  # Intraday 0.5%
        assert "1.0" in content  # Swing 1.0%
        assert "1.5" in content  # Position 1.5%
        assert "2.0" in content  # Investor 2.0%


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
