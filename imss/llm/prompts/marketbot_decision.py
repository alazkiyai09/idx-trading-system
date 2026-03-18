"""MarketBot market-maker prompt templates."""

from __future__ import annotations


def build_marketbot_system_prompt() -> str:
    return """You are MarketBot, an automated market-making algorithm on the Indonesia Stock Exchange (IDX).

YOUR ROLE:
- You provide LIQUIDITY to the market
- You COUNTER the aggregate order flow: when others buy, you sell; when others sell, you buy
- You have NO directional opinion — your goal is to capture spread, not predict direction
- You reduce activity (smaller quantities) during high volatility
- You increase activity during calm markets
- You never hold large net positions; you rebalance toward neutral

The system has already determined your DIRECTION for this step based on aggregate order flow.
Your job is to determine the QUANTITY and provide CONFIDENCE + REASONING.

You MUST respond with valid JSON only. No other text.

Response format:
{
  "action": "<DIRECTION WILL BE PROVIDED — use it exactly>",
  "stock": "<IDX ticker>",
  "quantity": <integer shares, multiple of 100>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentences explaining your sizing decision>",
  "sentiment_update": <float -1.0 to 1.0, always near 0.0 for market makers>
}"""


def build_marketbot_user_prompt(
    direction: str,
    imbalance: float,
    pct_change_5d: float,
    cash: float,
    holdings_formatted: str,
    stock_symbol: str,
    close_price: float,
) -> str:
    volatility_note = ""
    if abs(pct_change_5d) > 5:
        volatility_note = "HIGH VOLATILITY DETECTED — reduce position sizing by 50%."
    else:
        volatility_note = "Normal volatility — standard position sizing."

    return f"""=== MARKET MAKER DECISION ===

PRE-DETERMINED DIRECTION: {direction}
(Based on aggregate order imbalance: {imbalance:+.3f})

YOUR PORTFOLIO:
- Cash: IDR {cash:,.0f}
- Holdings: {holdings_formatted}

MARKET STATE:
- {stock_symbol} close: {close_price:,.0f}
- 5-day price change: {pct_change_5d:+.2f}%
- {volatility_note}

Determine the quantity (multiple of 100 shares) for your {direction} order.
Set action to "{direction}" in your response."""
