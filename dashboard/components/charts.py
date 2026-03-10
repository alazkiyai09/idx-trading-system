"""
Reusable Chart Components for the IDX Trading Dashboard.

Provides factory functions for building common Plotly charts used across
multiple dashboard pages. Uses NextGen design system colors.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Import NextGen colors
from dashboard.components.nextgen_styles import get_chart_colors, COLORS


def build_candlestick_chart(
    df: pd.DataFrame,
    title: str = "",
    show_volume: bool = True,
    show_ma: bool = True,
    show_rsi: bool = False,
    show_macd: bool = False,
    show_bbands: bool = False,
    prediction_data: dict = None,
) -> go.Figure:
    """Build a full interactive candlestick chart with optional overlays.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        title: Chart title
        show_volume: Show volume sub-bars
        show_ma: Show MA20/MA50 overlay
        show_rsi: Show RSI(14) subplot
        show_macd: Show MACD(12,26,9) subplot
        show_bbands: Show Bollinger Bands overlay
        prediction_data: Optional dict with 'predictions' list and 'is_mock' flag

    Returns:
        Plotly Figure object
    """
    chart_colors = get_chart_colors()

    # Determine number of subplot rows
    rows = 1
    row_heights = [0.6]
    if show_volume:
        rows += 1
        row_heights.append(0.15)
    if show_rsi:
        rows += 1
        row_heights.append(0.15)
    if show_macd:
        rows += 1
        row_heights.append(0.15)

    # Normalize heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # --- Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name="Price",
            increasing_line_color=chart_colors['up'],
            decreasing_line_color=chart_colors['down'],
        ),
        row=1, col=1,
    )

    # --- Moving Averages ---
    if show_ma:
        ma20 = df['close'].rolling(20).mean()
        ma50 = df['close'].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=df['date'], y=ma20,
            line=dict(color=chart_colors['ma20'], width=1),
            name="MA20"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=ma50,
            line=dict(color=chart_colors['ma50'], width=1),
            name="MA50"
        ), row=1, col=1)

    # --- Bollinger Bands ---
    if show_bbands:
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        fig.add_trace(go.Scatter(
            x=df['date'], y=upper,
            line=dict(color='rgba(161, 161, 170, 0.3)', width=1),
            name="BB Upper", showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=lower,
            line=dict(color='rgba(161, 161, 170, 0.3)', width=1),
            fill='tonexty', fillcolor='rgba(161, 161, 170, 0.08)',
            name="BB Lower", showlegend=False
        ), row=1, col=1)

    # --- ML Prediction Overlay ---
    if prediction_data and "predictions" in prediction_data:
        pred_df = pd.DataFrame(prediction_data["predictions"])
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        last_date = df['date'].iloc[-1]
        last_price = df['close'].iloc[-1]
        pred_dates = [last_date] + pred_df['date'].tolist()
        pred_prices = [last_price] + pred_df['predicted_price'].tolist()
        is_mock = prediction_data.get("is_mock", False)
        label = "7-Day Forecast (Mock)" if is_mock else "7-Day ML Forecast"
        color = 'rgba(245, 158, 11, 0.8)' if is_mock else 'rgba(16, 185, 129, 0.8)'
        fig.add_trace(go.Scatter(
            x=pred_dates, y=pred_prices,
            line=dict(color=color, width=3, dash='dash'),
            name=label
        ), row=1, col=1)

        # Confidence bands if available
        if "upper_band" in pred_df.columns and "lower_band" in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df['date'], y=pred_df['upper_band'],
                line=dict(width=0), showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=pred_df['date'], y=pred_df['lower_band'],
                line=dict(width=0), fill='tonexty',
                fillcolor='rgba(16, 185, 129, 0.1)',
                name="Confidence Band"
            ), row=1, col=1)

    current_row = 2

    # --- Volume ---
    if show_volume:
        colors = [
            chart_colors['volume_up'] if c >= o else chart_colors['volume_down']
            for c, o in zip(df['close'], df['open'])
        ]
        fig.add_trace(go.Bar(
            x=df['date'], y=df['volume'],
            marker_color=colors, name="Volume", showlegend=False
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="Vol", row=current_row, col=1)
        current_row += 1

    # --- RSI ---
    if show_rsi:
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(
            x=df['date'], y=rsi,
            line=dict(color=chart_colors['rsi'], width=1.5),
            name="RSI(14)"
        ), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=chart_colors['down'], opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=chart_colors['up'], opacity=0.5, row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
        current_row += 1

    # --- MACD ---
    if show_macd:
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        hist_colors = [
            chart_colors['volume_up'] if v >= 0 else chart_colors['volume_down']
            for v in histogram
        ]
        fig.add_trace(go.Bar(
            x=df['date'], y=histogram,
            marker_color=hist_colors, name="MACD Hist", showlegend=False
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=macd_line,
            line=dict(color=chart_colors['macd'], width=1.5),
            name="MACD"
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=signal_line,
            line=dict(color=chart_colors['signal'], width=1.5),
            name="Signal"
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)

    # Apply NextGen theme
    fig.update_layout(
        title=title,
        paper_bgcolor=chart_colors['background'],
        plot_bgcolor=chart_colors['background'],
        font=dict(color=chart_colors['text']),
        height=200 + rows * 200,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
        yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
    )

    return fig


def build_sentiment_gauge(score: float, title: str = "Market Sentiment") -> go.Figure:
    """Build a semicircular gauge chart for sentiment score (0-100)."""
    chart_colors = get_chart_colors()

    # Determine color based on score
    if score >= 70:
        bar_color = COLORS['primary']
    elif score >= 50:
        bar_color = COLORS['warning']
    else:
        bar_color = COLORS['destructive']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={'text': title, 'font': {'size': 18, 'color': chart_colors['text']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': chart_colors['muted_text']},
            'bar': {'color': bar_color},
            'bgcolor': chart_colors['background'],
            'steps': [
                {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [30, 50], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [50, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'},
            ],
            'threshold': {
                'line': {'color': chart_colors['text'], 'width': 2},
                'thickness': 0.8,
                'value': score,
            },
        },
        number={'font': {'color': chart_colors['text']}},
    ))
    fig.update_layout(
        paper_bgcolor=chart_colors['background'],
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def build_equity_curve(dates: list, values: list, title: str = "Equity Curve") -> go.Figure:
    """Build a line chart for portfolio equity over time."""
    chart_colors = get_chart_colors()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.15)',
        line=dict(color=chart_colors['primary'], width=2),
        name="Portfolio Value",
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor=chart_colors['background'],
        plot_bgcolor=chart_colors['background'],
        font=dict(color=chart_colors['text']),
        yaxis_title='Value (IDR)',
        height=350,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
        yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
    )
    return fig


def build_foreign_flow_chart(dates: list, flows: list, title: str = "Foreign Flow") -> go.Figure:
    """Build a bar chart for foreign flow data."""
    chart_colors = get_chart_colors()

    colors = [
        chart_colors['volume_up'] if f >= 0 else chart_colors['volume_down']
        for f in flows
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=flows,
        marker_color=colors,
        name="Foreign Flow",
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor=chart_colors['background'],
        plot_bgcolor=chart_colors['background'],
        font=dict(color=chart_colors['text']),
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
        yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='IDR (Billions)'),
    )
    return fig


def build_conviction_gauge(score: float, title: str = "Conviction Score") -> go.Figure:
    """Build a gauge for conviction score."""
    chart_colors = get_chart_colors()

    # Color based on score
    if score >= 80:
        bar_color = COLORS['primary']
        rating = "Strong Buy"
    elif score >= 60:
        bar_color = COLORS['primary_light']
        rating = "Buy"
    elif score >= 40:
        bar_color = COLORS['warning']
        rating = "Neutral"
    else:
        bar_color = COLORS['destructive']
        rating = "Sell"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': f"{title}<br><span style='font-size:0.8em'>{rating}</span>", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color},
            'bgcolor': chart_colors['background'],
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [60, 80], 'color': 'rgba(16, 185, 129, 0.1)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.2)'},
            ],
        },
        number={'font': {'color': chart_colors['text'], 'size': 48}, 'suffix': ''},
    ))
    fig.update_layout(
        paper_bgcolor=chart_colors['background'],
        height=200,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def build_treemap(labels: list, parents: list, values: list, colors: list = None, title: str = "") -> go.Figure:
    """Build a treemap for market overview."""
    chart_colors = get_chart_colors()

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors if colors else values,
            colorscale=[
                [0, 'rgba(239, 68, 68, 0.6)'],
                [0.5, 'rgba(161, 161, 170, 0.3)'],
                [1, 'rgba(16, 185, 129, 0.6)'],
            ],
        ),
        textfont=dict(color=chart_colors['text']),
        hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>%{color:.2f}%<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor=chart_colors['background'],
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
