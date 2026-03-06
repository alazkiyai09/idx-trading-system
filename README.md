# IDX Trading System

Institutional-grade trading system for Indonesia Stock Exchange (IDX).

## Key Features

- **Multi-mode trading**: Intraday, Swing, Position, and Investor modes
- **Foreign flow analysis**: Track and analyze foreign investor activity (critical for IDX)
- **Quantitative risk management**: Empirical Kelly Criterion with Monte Carlo uncertainty quantification
- **Calibration-based exits**: Dynamic exit timing based on calibration surface analysis
- **Backtesting engine**: Event-driven simulation with IDX-realistic conditions (fees, slippage, ARA/ARB)
- **Fundamental analysis**: Multi-agent analysis pipeline with fraud detection (Benford's Law)
- **TimesFM forecasting**: Google's Time Series Foundation Model integration (optional)
- **Notifications**: Telegram bot for signal alerts and daily summaries

## Trading Modes

| Mode | Hold Period | Risk/Trade | Focus |
|------|-------------|------------|-------|
| **Intraday** | Same day | 0.5% | Quick momentum |
| **Swing** | 2-7 days | 1.0% | Foreign flow + technical |
| **Position** | 1-4 weeks | 1.5% | Trend following |
| **Investor** | Months | 2.0% | Fundamentals |

## IDX Market Specifics

- **Lot size**: 100 shares
- **Fees**: 0.15% buy, 0.25% sell (includes 0.1% transaction tax)
- **Daily price limit**: +/-7% (ARA/ARB)
- **Settlement**: T+2
- **Trading hours**: 09:00-15:30 WIB (Asia/Jakarta)
- **Universe**: LQ45 (45 most liquid stocks), IDX30 (30 largest)

## Installation

### Prerequisites

- Python 3.10+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/alazkiyai09/idx-trading-system.git
cd idx-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

```bash
# Run daily scan (swing mode)
python scripts/daily_scan.py --mode swing

# Run backtest
python scripts/run_backtest.py --strategy swing --period 2023

# Run tests
pytest tests/ -v
```

## Project Structure

```
idx-trading-system/
├── config/              # Configuration (settings, trading modes, constants)
├── core/                # Core business logic
│   ├── data/            # Data layer (scraper, database, cache)
│   ├── analysis/        # Technical analysis
│   ├── signals/         # Signal generation
│   ├── risk/            # Risk management (Empirical Kelly, pattern matching)
│   ├── execution/       # Paper trading, order management
│   ├── portfolio/       # Portfolio tracking
│   └── forecasting/     # TimesFM integration
├── research/            # Monte Carlo, calibration surfaces, return analysis
├── backtest/            # Backtesting engine, metrics, walk-forward
├── fundamental/         # Fundamental analysis with multi-agent pipeline
├── agents/              # Trading system coordinator
├── notifications/       # Telegram notifications
├── scripts/             # CLI scripts (daily scan, backtest, data fetch)
└── tests/               # Unit, integration, and E2E tests
```

## Architecture

```
Data Layer → Analysis Layer → Signal Layer → Risk Layer → Execution Layer → Output Layer
```

- **Risk Manager** has veto power over all trades
- **LLM outputs** are validated before affecting trading decisions
- **Backtesting is mandatory** before live trading

## Risk Management

The system uses three institutional-grade methods:

1. **Empirical Kelly Criterion** - Position sizing adjusted for edge uncertainty
2. **Calibration Surface Analysis** - Dynamic exit timing based on signal strength decay
3. **Monte Carlo Simulation** - 10,000 path resampling for drawdown distribution analysis

## Configuration

Key settings in `.env`:

```bash
ANTHROPIC_API_KEY=your_key    # For Claude LLM
GLM_API_KEY=your_key          # For GLM analysis
TELEGRAM_BOT_TOKEN=your_token # For notifications
INITIAL_CAPITAL=100000000     # Starting capital (IDR)
DEFAULT_MODE=swing            # Default trading mode
PAPER_TRADING=true            # Paper trading mode
```

## Tech Stack

- **Language**: Python 3.10+
- **Database**: SQLite (via SQLAlchemy)
- **Data**: Yahoo Finance, IDX website scraping
- **Analysis**: pandas, numpy, scipy, pandas-ta
- **LLM**: Claude (Anthropic), GLM
- **Forecasting**: TimesFM 2.5 (Google, optional)
- **Notifications**: python-telegram-bot
- **Testing**: pytest

## Disclaimer

This trading system is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and never invest more than you can afford to lose.

## License

Private repository. All rights reserved.
