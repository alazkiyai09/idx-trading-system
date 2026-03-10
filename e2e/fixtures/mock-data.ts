/**
 * Mock Data Generators
 *
 * Provides functions to generate mock data for testing.
 */

/**
 * Generate sample OHLCV data
 */
export function generateMockOHLCV(
  symbol: string,
  days: number = 30,
  startPrice: number = 10000
): Array<{
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}> {
  const data = [];
  let price = startPrice;
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);

    // Skip weekends
    if (date.getDay() === 0 || date.getDay() === 6) continue;

    // Random price movement (max 7% for IDX)
    const change = (Math.random() - 0.5) * 0.10 * price;
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) * (1 + Math.random() * 0.02);
    const low = Math.min(open, close) * (1 - Math.random() * 0.02);
    const volume = Math.floor(Math.random() * 10000000) + 100000;

    data.push({
      date: date.toISOString().split('T')[0],
      open: Math.round(open),
      high: Math.round(high),
      low: Math.round(low),
      close: Math.round(close),
      volume,
    });

    price = close;
  }

  return data;
}

/**
 * Generate sample signal data
 */
export function generateMockSignal(symbol: string, overrides: Partial<any> = {}) {
  const entryPrice = 10000 + Math.random() * 5000;
  return {
    symbol,
    signal_type: Math.random() > 0.5 ? 'LONG' : 'SHORT',
    setup_type: ['breakout', 'pullback', 'reversal'][Math.floor(Math.random() * 3)],
    entry_price: entryPrice,
    stop_loss: entryPrice * 0.95,
    targets: [entryPrice * 1.05, entryPrice * 1.10, entryPrice * 1.15],
    composite_score: Math.floor(Math.random() * 40) + 60, // 60-100
    key_factors: ['Strong momentum', 'Volume confirmation', 'Sector support'],
    risks: ['Market volatility', 'Earnings coming up'],
    mode: 'swing',
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * Generate sample portfolio state
 */
export function generateMockPortfolio(overrides: Partial<any> = {}) {
  return {
    total_value: 150_000_000,
    cash: 100_000_000,
    invested: 50_000_000,
    unrealized_pnl: 5_000_000,
    realized_pnl: 3_000_000,
    total_pnl: 8_000_000,
    total_pnl_pct: 5.33,
    num_positions: 3,
    positions: [
      {
        symbol: 'BBCA',
        entry_price: 9500,
        current_price: 9800,
        shares: 500,
        entry_date: '2024-03-01',
        unrealized_pnl: 150_000,
        unrealized_pnl_pct: 3.16,
        stop_loss: 9000,
        target: 10500,
      },
      {
        symbol: 'TLKM',
        entry_price: 3800,
        current_price: 4000,
        shares: 2000,
        entry_date: '2024-03-05',
        unrealized_pnl: 400_000,
        unrealized_pnl_pct: 5.26,
        stop_loss: 3600,
        target: 4500,
      },
    ],
    ...overrides,
  };
}

/**
 * Generate sample sentiment data
 */
export function generateMockSentiment(symbol?: string) {
  return {
    symbol: symbol || 'MARKET',
    sentiment_score: (Math.random() * 2 - 1).toFixed(2), // -1 to 1
    confidence: (0.6 + Math.random() * 0.4).toFixed(2), // 0.6 to 1.0
    key_topics: ['Earnings', 'Dividend', 'Market Outlook'],
    themes: ['Oil Prices', 'Interest Rates', 'Inflation'],
    sector: 'Financials',
    analyzed_at: new Date().toISOString(),
  };
}

/**
 * Generate sample backtest request
 */
export function generateMockBacktestRequest(overrides: Partial<any> = {}) {
  const today = new Date();
  const startDate = new Date(today);
  startDate.setFullYear(startDate.getFullYear() - 1);

  return {
    symbols: ['BBCA', 'BBRI', 'TLKM'],
    mode: 'swing',
    start_date: startDate.toISOString().split('T')[0],
    end_date: today.toISOString().split('T')[0],
    initial_capital: 100_000_000,
    ...overrides,
  };
}

/**
 * Generate sample simulation session
 */
export function generateMockSimulation(overrides: Partial<any> = {}) {
  return {
    name: 'Test Simulation',
    mode: 'live',
    trading_mode: 'swing',
    initial_capital: 100_000_000,
    ...overrides,
  };
}

/**
 * Generate sample order request
 */
export function generateMockOrderRequest(overrides: Partial<any> = {}) {
  return {
    symbol: 'BBCA',
    side: 'LONG',
    quantity: 100, // 1 lot
    order_type: 'MARKET',
    price: 0,
    targets: [10000, 10500, 11000],
    ...overrides,
  };
}

/**
 * Common IDX stock symbols for testing
 */
export const TEST_SYMBOLS = {
  BLUE_CHIPS: ['BBCA', 'BBRI', 'TLKM', 'ASII', 'UNVR'],
  TECH: ['GOTO', 'BUKA', 'EMTK'],
  BANKING: ['BBCA', 'BBRI', 'BMTR', 'BBNI'],
  COMMODITIES: ['ANTM', 'INCO', 'MDKA'],
};

/**
 * Trading mode configurations
 */
export const TRADING_MODE_CONFIGS = {
  intraday: {
    min_hold_days: 0,
    max_hold_days: 1,
    max_risk_per_trade: 0.005,
  },
  swing: {
    min_hold_days: 2,
    max_hold_days: 7,
    max_risk_per_trade: 0.01,
  },
  position: {
    min_hold_days: 7,
    max_hold_days: 28,
    max_risk_per_trade: 0.015,
  },
  investor: {
    min_hold_days: 30,
    max_hold_days: 365,
    max_risk_per_trade: 0.02,
  },
} as const;
