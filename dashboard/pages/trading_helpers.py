"""
Helper functions for Virtual Trading page - extracted for testability.
"""
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime


def calculate_order_value(
    quantity: int,
    price: float,
    side: str = "BUY"
) -> Dict[str, float]:
    """Calculate order value including fees.

    Args:
        quantity: Number of shares
        price: Price per share
        side: BUY or SELL

    Returns:
        Dict with order value breakdown
    """
    gross_value = quantity * price

    # IDX fees: 0.15% buy, 0.25% sell (includes tax)
    fee_rate = 0.0015 if side == "BUY" else 0.0025
    fees = gross_value * fee_rate

    net_value = gross_value + fees if side == "BUY" else gross_value - fees

    return {
        "gross_value": round(gross_value, 2),
        "fee_rate": fee_rate,
        "fees": round(fees, 2),
        "net_value": round(net_value, 2),
    }


def validate_order(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    capital: float,
    existing_positions: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Validate order parameters.

    Args:
        symbol: Stock symbol
        side: BUY or SELL
        quantity: Number of shares
        price: Order price
        capital: Available capital
        existing_positions: List of existing positions

    Returns:
        Dict with validation result
    """
    errors = []

    # Basic validation
    if not symbol:
        errors.append("Symbol is required")

    if side not in ["BUY", "SELL"]:
        errors.append("Side must be BUY or SELL")

    if quantity <= 0:
        errors.append("Quantity must be positive")

    if quantity % 100 != 0:
        errors.append("Quantity must be a multiple of 100 (lot size)")

    if price <= 0:
        errors.append("Price must be positive")

    # Check capital for buy orders
    if side == "BUY" and price > 0 and quantity > 0:
        order_calc = calculate_order_value(quantity, price, "BUY")
        if order_calc["net_value"] > capital:
            errors.append(f"Insufficient capital. Need {order_calc['net_value']:,.0f}")

    # Check position for sell orders
    if side == "SELL":
        if not existing_positions:
            errors.append(f"No position in {symbol} to sell")
        else:
            position = next(
                (p for p in existing_positions if p.get("symbol") == symbol),
                None
            )
            if not position:
                errors.append(f"No position in {symbol} to sell")
            elif position.get("quantity", 0) < quantity:
                errors.append(f"Insufficient shares. Have {position['quantity']}, want to sell {quantity}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def calculate_portfolio_value(
    capital: float,
    positions: List[Dict],
    current_prices: Dict[str, float]
) -> Dict[str, float]:
    """Calculate total portfolio value.

    Args:
        capital: Cash capital
        positions: List of position dicts with symbol and quantity
        current_prices: Dict mapping symbol to current price

    Returns:
        Dict with portfolio metrics
    """
    cash = capital
    positions_value = 0
    unrealized_pnl = 0

    for pos in positions:
        symbol = pos.get("symbol")
        quantity = pos.get("quantity", 0)
        avg_price = pos.get("avg_price", 0)

        if symbol in current_prices:
            current_price = current_prices[symbol]
            market_value = quantity * current_price
            cost_basis = quantity * avg_price

            positions_value += market_value
            unrealized_pnl += (market_value - cost_basis)

    total_value = cash + positions_value

    return {
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_value": round(total_value, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
    }


def calculate_performance_metrics(
    trades: List[Dict]
) -> Dict[str, Any]:
    """Calculate trading performance metrics.

    Args:
        trades: List of trade dicts with pnl

    Returns:
        Dict with performance metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
        }

    pnls = [t.get("pnl", 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    win_rate = len(wins) / len(trades) if trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
        "profit_factor": round(profit_factor, 2),
    }


def process_trade_history(
    trades: List[Dict]
) -> pd.DataFrame:
    """Process trade history for display.

    Args:
        trades: List of trade dicts

    Returns:
        DataFrame with formatted trade history
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    # Ensure required columns exist
    required_cols = ["symbol", "side", "quantity", "price", "pnl", "timestamp"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Format timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df.sort_values('timestamp', ascending=False) if 'timestamp' in df.columns else df


def generate_equity_curve(
    initial_capital: float,
    trades: List[Dict]
) -> List[Dict]:
    """Generate equity curve data points.

    Args:
        initial_capital: Starting capital
        trades: List of trade dicts with timestamp and cumulative_pnl

    Returns:
        List of equity curve points with date and value
    """
    if not trades:
        return [{"date": datetime.now().isoformat(), "value": initial_capital}]

    curve = []
    running_capital = initial_capital

    for trade in sorted(trades, key=lambda x: x.get('timestamp', '')):
        pnl = trade.get('pnl', 0)
        running_capital += pnl

        curve.append({
            "date": trade.get('timestamp', ''),
            "value": round(running_capital, 2),
        })

    return curve
