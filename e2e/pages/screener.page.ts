/**
 * Stock Screener Page Object
 *
 * Handles interactions with the Stock Screener dashboard page.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';

export class ScreenerPage extends BasePage {
  readonly pageUrl = '/01_screener';

  constructor(page: Page) {
    super(page);
  }

  async goto(): Promise<void> {
    await this.page.goto(this.pageUrl);
    await this.waitForLoad();
  }

  /**
   * Get filter controls section
   */
  getFilterControls(): Locator {
    return this.page.locator('text=FILTER CONFIGURATION');
  }

  /**
   * Get scan button
   */
  getScanButton(): Locator {
    return this.page.locator('button:has-text("Scan Market"), button:has-text("Scan")');
  }

  /**
   * Get results table
   */
  getResultsTable(): Locator {
    return this.page.locator('[data-testid="stDataFrame"], table');
  }

  /**
   * Toggle LQ45 filter
   */
  async toggleLQ45(): Promise<void> {
    const checkbox = this.page.locator('label:has-text("LQ45") input, [data-testid="lq45-filter"]');
    if (await checkbox.count() > 0) {
      await checkbox.first().check();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Toggle IDX30 filter
   */
  async toggleIDX30(): Promise<void> {
    const checkbox = this.page.locator('label:has-text("IDX30") input, [data-testid="idx30-filter"]');
    if (await checkbox.count() > 0) {
      await checkbox.first().check();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Select sector filter
   */
  async selectSector(sector: string): Promise<void> {
    const select = this.page.locator('select[name="sector"], [data-testid="sector-select"]').first();
    if (await select.count() > 0) {
      await select.selectOption({ label: sector });
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Set minimum market cap filter
   */
  async setMinMarketCap(value: number): Promise<void> {
    const input = this.page.locator('input[name="min_market_cap"], [data-testid="market-cap-input"]').first();
    if (await input.count() > 0) {
      await input.fill(value.toString());
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Set RSI range filter
   */
  async setRSIRange(min: number, max: number): Promise<void> {
    const minInput = this.page.locator('input[name="rsi_min"], [data-testid="rsi-min"]').first();
    const maxInput = this.page.locator('input[name="rsi_max"], [data-testid="rsi-max"]').first();

    if (await minInput.count() > 0) {
      await minInput.fill(min.toString());
    }
    if (await maxInput.count() > 0) {
      await maxInput.fill(max.toString());
    }
    await this.page.waitForTimeout(500);
  }

  /**
   * Click the scan market button
   */
  async clickScanMarket(): Promise<void> {
    await this.getScanButton().click();
    // Wait for analysis to complete
    await this.page.waitForTimeout(2000);
    await this.streamlitHelper.waitForAppReady();
  }

  /**
   * Get results count
   */
  async getResultsCount(): Promise<number> {
    const rows = await this.page.locator('table tbody tr, [data-testid="stDataFrame"] tbody tr').count();
    return rows;
  }

  /**
   * Verify results are displayed
   */
  async verifyResultsDisplayed(): Promise<void> {
    await expect(this.getResultsTable()).toBeVisible();
  }

  /**
   * Get composite score column values
   */
  async getCompositeScores(): Promise<number[]> {
    const cells = await this.page.locator('td:has-text("Score"), [data-testid="score-cell"]').allTextContents();
    return cells.map(text => parseFloat(text) || 0);
  }

  /**
   * Navigate to stock detail page
   */
  async navigateToStockDetail(symbol: string): Promise<void> {
    // Click on the symbol row
    await this.page.click(`text="${symbol}"`);
    await this.page.waitForTimeout(500);
  }

  /**
   * Get found stocks message
   */
  async getFoundStocksMessage(): Promise<string> {
    const message = await this.page.locator('[data-testid="stMarkdownContainer"]').first().textContent();
    return message || '';
  }
}
