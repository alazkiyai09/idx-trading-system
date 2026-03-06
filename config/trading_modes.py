"""
Trading mode configurations.

Each mode has different parameters for:
- Position sizing
- Hold periods
- Risk limits
- Signal thresholds
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class TradingMode(Enum):
    """Trading mode enumeration.

    Attributes:
        INTRADAY: Same-day trading, quick momentum plays.
        SWING: 2-7 day holds, foreign flow + technical focus.
        POSITION: 1-4 week holds, trend following.
        INVESTOR: Month+ holds, fundamentals focused.
    """

    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"
    INVESTOR = "investor"


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a trading mode.

    This dataclass contains all parameters specific to a trading mode,
    including risk limits, hold periods, and signal thresholds.

    Attributes:
        name: Human-readable name of the mode.
        mode: TradingMode enum value.
        max_risk_per_trade: Maximum risk per trade as decimal.
        max_position_pct: Maximum position size as decimal of capital.
        min_hold_days: Minimum holding period in days.
        max_hold_days: Maximum holding period in days.
        min_score: Minimum composite score required for entry.
        min_flow_signal: Minimum foreign flow signal required.
        rsi_oversold: RSI level considered oversold.
        rsi_overbought: RSI level considered overbought.
        min_volume_ratio: Minimum volume ratio vs average.
        default_stop_pct: Default stop loss percentage.
        target_1_ratio: First target as R-multiple.
        target_2_ratio: Second target as R-multiple.
        target_3_ratio: Third target as R-multiple.
        technical_weight: Weight for technical score in composite.
        flow_weight: Weight for foreign flow score in composite.
        fundamental_weight: Weight for fundamental score in composite.
    """

    name: str
    mode: TradingMode

    # Position sizing
    max_risk_per_trade: float
    max_position_pct: float

    # Hold periods
    min_hold_days: int
    max_hold_days: int

    # Signal thresholds
    min_score: int
    min_flow_signal: str  # "strong_buy", "buy", "neutral"

    # Technical filters
    rsi_oversold: int
    rsi_overbought: int
    min_volume_ratio: float

    # Targets and stops
    default_stop_pct: float
    target_1_ratio: float  # R multiple
    target_2_ratio: float
    target_3_ratio: float

    # Weights for composite score
    technical_weight: float
    flow_weight: float
    fundamental_weight: float


# Mode configurations dictionary
MODE_CONFIGS: Dict[TradingMode, ModeConfig] = {
    TradingMode.INTRADAY: ModeConfig(
        name="Intraday",
        mode=TradingMode.INTRADAY,
        max_risk_per_trade=0.005,  # 0.5%
        max_position_pct=0.20,
        min_hold_days=0,
        max_hold_days=1,
        min_score=70,
        min_flow_signal="buy",
        rsi_oversold=30,
        rsi_overbought=70,
        min_volume_ratio=1.5,
        default_stop_pct=0.02,
        target_1_ratio=1.5,
        target_2_ratio=2.0,
        target_3_ratio=3.0,
        technical_weight=0.50,
        flow_weight=0.40,
        fundamental_weight=0.10,
    ),
    TradingMode.SWING: ModeConfig(
        name="Swing",
        mode=TradingMode.SWING,
        max_risk_per_trade=0.01,  # 1%
        max_position_pct=0.25,
        min_hold_days=2,
        max_hold_days=7,
        min_score=65,
        min_flow_signal="buy",
        rsi_oversold=35,
        rsi_overbought=70,
        min_volume_ratio=1.2,
        default_stop_pct=0.05,
        target_1_ratio=1.5,
        target_2_ratio=2.5,
        target_3_ratio=4.0,
        technical_weight=0.40,
        flow_weight=0.45,
        fundamental_weight=0.15,
    ),
    TradingMode.POSITION: ModeConfig(
        name="Position",
        mode=TradingMode.POSITION,
        max_risk_per_trade=0.015,  # 1.5%
        max_position_pct=0.30,
        min_hold_days=5,
        max_hold_days=20,
        min_score=60,
        min_flow_signal="neutral",
        rsi_oversold=40,
        rsi_overbought=65,
        min_volume_ratio=1.0,
        default_stop_pct=0.08,
        target_1_ratio=2.0,
        target_2_ratio=3.5,
        target_3_ratio=5.0,
        technical_weight=0.35,
        flow_weight=0.35,
        fundamental_weight=0.30,
    ),
    TradingMode.INVESTOR: ModeConfig(
        name="Investor",
        mode=TradingMode.INVESTOR,
        max_risk_per_trade=0.02,  # 2%
        max_position_pct=0.35,
        min_hold_days=20,
        max_hold_days=90,
        min_score=55,
        min_flow_signal="neutral",
        rsi_oversold=45,
        rsi_overbought=60,
        min_volume_ratio=0.8,
        default_stop_pct=0.12,
        target_1_ratio=2.5,
        target_2_ratio=4.0,
        target_3_ratio=6.0,
        technical_weight=0.20,
        flow_weight=0.20,
        fundamental_weight=0.60,
    ),
}


def get_mode_config(mode: TradingMode) -> ModeConfig:
    """Get configuration for a trading mode.

    Args:
        mode: TradingMode enum value.

    Returns:
        ModeConfig for the specified mode.

    Raises:
        KeyError: If mode is not found in configurations.
    """
    return MODE_CONFIGS[mode]


def get_mode_from_string(mode_str: str) -> TradingMode:
    """Convert string to TradingMode enum.

    Args:
        mode_str: String representation of trading mode.

    Returns:
        TradingMode enum value.

    Raises:
        ValueError: If mode string is not recognized.
    """
    mode_map = {
        "intraday": TradingMode.INTRADAY,
        "swing": TradingMode.SWING,
        "position": TradingMode.POSITION,
        "investor": TradingMode.INVESTOR,
    }
    mode_lower = mode_str.lower().strip()
    if mode_lower not in mode_map:
        valid_modes = ", ".join(mode_map.keys())
        raise ValueError(f"Invalid mode '{mode_str}'. Valid modes: {valid_modes}")
    return mode_map[mode_lower]


def get_all_modes() -> Dict[TradingMode, ModeConfig]:
    """Get all mode configurations.

    Returns:
        Dictionary mapping TradingMode to ModeConfig.
    """
    return MODE_CONFIGS.copy()
