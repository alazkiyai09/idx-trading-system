/**
 * Market Overview Page Object
 *
 * Handles interactions with the Market Overview dashboard page.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';

export class MarketOverviewPage extends BasePage {
  readonly pageUrl = '/06_market_overview';

  constructor(page: Page) {
    super(page);
  }

  async goto(): Promise<void> {
    await this.page.goto(this.pageUrl);
    await this.waitForLoad();
  }

  /**
   * Get sector heatmap container
   */
  getSectorHeatmap(): Locator {
    return this.page.locator('[data-testid="stVegaLiteChart"], .stVegaLiteChart, canvas').first();
  }

  /**
   * Get stock list container
   */
  getStockList(): Locator {
    return this.page.locator('[data-testid="stDataFrame"], table, .stDataFrame').first();
  }

  /**
   * Get treemap visualization
   */
  getTreemap(): Locator {
    return this.page.locator('[data-testid="stVegaLiteChart"], .stVegaLiteChart, canvas').first();
  }

  /**
   * Click on a sector in the heatmap
   */
  async clickSector(sectorName: string): Promise<void> {
    await this.page.click(`text="${sectorName}"`);
    await this.page.waitForTimeout(500);
  }

  /**
   * Get stocks count from the list
   */
  async getStocksCount(): Promise<number> {
    const rows = await this.page.locator('table tbody tr').count();
    return rows;
  }

  /**
   * Filter stocks by sector
   */
  async filterBySector(sector: string): Promise<void> {
    const sectorFilter = this.page.locator('select[name="sector"], [data-testid="sector-filter"]');
    if (await sectorFilter.count() > 0) {
      await sectorFilter.first().selectOption({ label: sector });
      await this.page.waitForTimeout(1000);
    }
  }

  /**
   * Verify market data is displayed
   */
  async verifyMarketDataLoaded(): Promise<void> {
    await expect(this.getStockList()).toBeVisible();
    const count = await this.getStocksCount();
    expect(count).toBeGreaterThan(0);
  }

  /**
   * Get top gainers section
   */
  getTopGainers(): Locator {
    return this.page.locator('[data-testid="top-gainers"], .gainers-list');
  }

  /**
   * Get top losers section
   */
  getTopLosers(): Locator {
    return this.page.locator('[data-testid="top-losers"], .losers-list');
  }

  /**
   * Get most active by volume section
   */
  getMostActive(): Locator {
    return this.page.locator('[data-testid="most-active"], .active-list');
  }
}
