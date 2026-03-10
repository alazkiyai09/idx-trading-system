/**
 * Virtual Trading Page Object
 *
 * Handles interactions with the Virtual Trading dashboard page.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';

interface OrderParams {
  symbol: string;
  side: string;
  quantity: number;
  orderType?: string;
}

interface OrderResult {
  success: boolean;
  error?: string;
}

export class VirtualTradingPage extends BasePage {
  readonly pageUrl = '/04_virtual_trading';

  constructor(page: Page) {
    super(page);
  }

  async goto(): Promise<void> {
    await this.page.goto(this.pageUrl);
    await this.waitForLoad();
  }

  /**
   * Get portfolio summary section
   */
  getPortfolioSummary(): Locator {
    return this.page.locator('[data-testid="stMetric"], .stMetric').first();
  }

  /**
   * Get order entry form
   */
  getOrderForm(): Locator {
    return this.page.locator('[data-testid="stForm"], form, .stForm');
  }

  /**
   * Get positions table
   */
  getPositionsTable(): Locator {
    return this.page.locator('[data-testid="stDataFrame"], table, .stDataFrame');
  }

  /**
   * Get equity curve chart
   */
  getEquityCurve(): Locator {
    return this.page.locator('[data-testid="stVegaLiteChart"], .stVegaLiteChart, canvas');
  }

  /**
   * Get trade history section
   */
  getTradeHistory(): Locator {
    return this.page.locator('[data-testid="stDataFrame"], table').nth(1);
  }

  /**
   * Get create session button
   */
  getCreateSessionButton(): Locator {
    return this.page.locator('button:has-text("New Session"), button:has-text("Create Session"), button:has-text("Create")');
  }

  /**
   * Get submit order button
   */
  getSubmitOrderButton(): Locator {
    return this.page.locator('button:has-text("Submit Order"), button:has-text("Execute"), button:has-text("Submit")');
  }

  /**
   * Click create session button
   */
  async clickCreateSession(): Promise<void> {
    await this.getCreateSessionButton().click();
    await this.page.waitForTimeout(1000);
  }

  /**
   * Create new trading session
   */
  async createSession(name: string, tradingMode: string, initialCapital: number): Promise<void> {
    // Fill session name
    const nameInput = this.page.locator('input[placeholder*="name" i], input[name="name"], [data-testid="session-name"]').first();
    if (await nameInput.count() > 0) {
      await nameInput.fill(name);
    }

    // Select trading mode
    const modeSelect = this.page.locator('select[name="trading_mode"], [data-testid="trading-mode"], select').first();
    if (await modeSelect.count() > 0) {
      await modeSelect.selectOption({ label: tradingMode });
    }

    // Set initial capital
    const capitalInput = this.page.locator('input[name="initial_capital"], [data-testid="capital-input"], input[type="number"]').first();
    if (await capitalInput.count() > 0) {
      await capitalInput.fill(initialCapital.toString());
    }

    // Click create
    await this.getCreateSessionButton().click();
    await this.page.waitForTimeout(1000);
    await this.streamlitHelper.waitForAppReady();
  }

  /**
   * Submit a buy order
   */
  async submitBuyOrder(symbol: string, quantity: number): Promise<void> {
    await this.submitOrderInternal(symbol, 'BUY', quantity);
  }

  /**
   * Submit a sell order
   */
  async submitSellOrder(symbol: string, quantity: number): Promise<void> {
    await this.submitOrderInternal(symbol, 'SELL', quantity);
  }

  /**
   * Submit an order with object params (for tests)
   */
  async submitOrder(params: OrderParams): Promise<void> {
    await this.submitOrderInternal(params.symbol, params.side, params.quantity);
  }

  /**
   * Submit order and return result (for validation tests)
   */
  async submitOrderRaw(params: OrderParams): Promise<OrderResult> {
    try {
      await this.submitOrderInternal(params.symbol, params.side, params.quantity);

      // Check for error
      const errorLocator = this.page.locator('[data-testid="stError"], .stNotification, .error');
      const hasError = await errorLocator.count() > 0;

      if (hasError) {
        const errorText = await errorLocator.first().textContent();
        return { success: false, error: errorText || 'Unknown error' };
      }

      return { success: true };
    } catch (error) {
      return { success: false, error: String(error) };
    }
  }

  /**
   * Submit an order (internal implementation)
   */
  private async submitOrderInternal(symbol: string, side: string, quantity: number): Promise<void> {
    // Select symbol
    const symbolSelect = this.page.locator('select[name="symbol"], [data-testid="symbol-select"], select').first();
    if (await symbolSelect.count() > 0) {
      await symbolSelect.selectOption({ label: symbol });
    }

    // Select side
    const sideSelect = this.page.locator('select[name="side"], [data-testid="side-select"], select').nth(1);
    if (await sideSelect.count() > 0) {
      await sideSelect.selectOption({ label: side });
    }

    // Set quantity (must be multiple of 100 for IDX lot size)
    const quantityInput = this.page.locator('input[name="quantity"], [data-testid="quantity-input"], input[type="number"]').first();
    if (await quantityInput.count() > 0) {
      await quantityInput.fill(quantity.toString());
    }

    // Click submit
    await this.getSubmitOrderButton().click();
    await this.page.waitForTimeout(1000);
    await this.streamlitHelper.waitForAppReady();
  }

  /**
   * Get current capital
   */
  async getCurrentCapital(): Promise<number> {
    const capitalText = await this.page.locator('[data-testid="current-capital"], .capital-value, .stMetric label:has-text("Capital") + div').textContent();
    const capital = parseFloat(capitalText?.replace(/[^0-9.-]/g, '') || '0');
    return capital;
  }

  /**
   * Get total P&L
   */
  async getTotalPnL(): Promise<number> {
    const pnlText = await this.page.locator('[data-testid="total-pnl"], .pnl-value, .stMetric label:has-text("P&L") + div').textContent();
    const pnl = parseFloat(pnlText?.replace(/[^0-9.-]/g, '') || '0');
    return pnl;
  }

  /**
   * Get open positions count
   */
  async getOpenPositionsCount(): Promise<number> {
    return this.getPositionsTable().locator('tbody tr, [data-testid="position-row"], tr').count();
  }

  /**
   * Get performance metrics section (returns Locator for visibility check)
   */
  getPerformanceMetricsSection(): Locator {
    return this.page.locator('.stMetric, [data-testid="stMetric"]').first();
  }

  /**
   * Get performance metrics values
   */
  async getPerformanceMetrics(): Promise<{
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
  }> {
    const sharpeText = await this.page.locator('[data-testid="sharpe-ratio"], .sharpe-value, .stMetric:has-text("Sharpe") div').first().textContent();
    const drawdownText = await this.page.locator('[data-testid="max-drawdown"], .drawdown-value, .stMetric:has-text("Drawdown") div').first().textContent();
    const winRateText = await this.page.locator('[data-testid="win-rate"], .winrate-value, .stMetric:has-text("Win") div').first().textContent();

    return {
      sharpeRatio: parseFloat(sharpeText || '0'),
      maxDrawdown: parseFloat(drawdownText?.replace('%', '') || '0'),
      winRate: parseFloat(winRateText?.replace('%', '') || '0'),
    };
  }

  /**
   * Verify lot size validation (must be multiple of 100)
   */
  async verifyLotSizeValidation(): Promise<boolean> {
    // Try to submit an invalid quantity (not multiple of 100)
    const quantityInput = this.page.locator('input[name="quantity"], [data-testid="quantity-input"], input[type="number"]').first();
    if (await quantityInput.count() > 0) {
      await quantityInput.fill('50'); // Invalid: not multiple of 100
      await this.getSubmitOrderButton().click();
      await this.page.waitForTimeout(500);

      // Check for error message
      const errorVisible = await this.page.locator('[data-testid="stError"], .stNotification, .error').isVisible();
      return errorVisible;
    }
    return false;
  }

  /**
   * Wait for equity curve to render
   */
  async waitForEquityCurveRender(): Promise<void> {
    await this.streamlitHelper.waitForChartRender('[data-testid="stVegaLiteChart"], canvas');
  }
}
