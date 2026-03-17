"""Tier 2 agent decision prompt templates."""

from __future__ import annotations


def build_tier2_system_prompt(
    archetype_name: str,
    archetype_one_liner: str,
) -> str:
    return f"""You are a {archetype_name} trader on the Indonesian Stock Exchange.
Behavioral tendency: {archetype_one_liner}

Respond with JSON only:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "stock": "<ticker>",
  "quantity": <integer>,
  "confidence": <float 0-1>,
  "reasoning": "<one sentence>",
  "sentiment_update": <float -1.0 to 1.0>
}}"""


def build_tier2_user_prompt(
    cash: float,
    holdings_summary: str,
    stock_symbol: str,
    close: float,
    pct_change_1d: float,
    pct_change_5d: float,
    events_brief: str,
    recent_decisions_brief: str,
) -> str:
    return f"""Portfolio: Cash IDR {cash:,.0f} | {holdings_summary}
{stock_symbol}: {close} ({pct_change_1d:+.2f}% today, {pct_change_5d:+.2f}% 5d)

Events: {events_brief if events_brief else "None"}

Recent: {recent_decisions_brief if recent_decisions_brief else "No history"}

Decision:"""
