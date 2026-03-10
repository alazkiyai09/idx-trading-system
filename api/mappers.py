"""Domain-to-API mappers.

These helpers keep the API layer decoupled from internal dataclass aliases.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def map_signal_dict(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a signal-like dict for API responses."""
    targets = signal.get("targets")
    if not targets:
        targets = [
            target
            for target in [
                signal.get("target_1"),
                signal.get("target_2"),
                signal.get("target_3"),
            ]
            if target is not None
        ]

    return {
        "symbol": signal.get("symbol", ""),
        "signal_type": _enum_value(signal.get("signal_type", "")),
        "setup_type": _enum_value(signal.get("setup_type", "")),
        "entry_price": signal.get("entry_price", 0),
        "stop_loss": signal.get("stop_loss", 0),
        "targets": targets,
        "composite_score": signal.get("composite_score", signal.get("score", 0)),
        "key_factors": signal.get("key_factors", []),
        "risks": signal.get("risks", signal.get("risk_factors", [])),
    }


def map_position_dict(position: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a portfolio position dict for API responses."""
    target = position.get("target")
    if target is None:
        target = position.get("target_1")

    shares = position.get("shares")
    if shares is None:
        shares = position.get("quantity", 0)

    return {
        "symbol": position.get("symbol", ""),
        "entry_price": position.get("entry_price", 0),
        "current_price": position.get("current_price"),
        "shares": shares,
        "entry_date": position.get("entry_date"),
        "unrealized_pnl": position.get("unrealized_pnl", 0),
        "unrealized_pnl_pct": position.get("unrealized_pnl_pct", 0),
        "stop_loss": position.get("stop_loss"),
        "target": target,
    }
