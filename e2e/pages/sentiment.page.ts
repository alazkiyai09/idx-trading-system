/**
 * Sentiment Page Object
 *
 * Handles interactions with the Sentiment dashboard page.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';

export class SentimentPage extends BasePage {
  readonly pageUrl = '/03_sentiment';

  constructor(page: Page) {
    super(page);
  }

  async goto(): Promise<void> {
    await this.page.goto(this.pageUrl);
    await this.waitForLoad();
  }

  /**
   * Get market sentiment gauge
   */
  getSentimentGauge(): Locator {
    return this.page.locator('[data-testid="stVegaLiteChart"], .stVegaLiteChart, canvas').first();
  }

  /**
   * Get sector sentiment heatmap
   */
  getSectorHeatmap(): Locator {
    return this.page.locator('[data-testid="stDataFrame"], table, .stDataFrame').first();
  }

  /**
   * Get theme cards container
   */
  getThemeCards(): Locator {
    return this.page.locator('[data-testid="stMarkdownContainer"], .stMarkdownContainer');
  }

  /**
   * Get news articles list
   */
  getNewsArticles(): Locator {
    return this.page.locator('[data-testid="stExpander"], .stExpander, [data-testid="stDataFrame"]').first();
  }

  /**
   * Get fetch latest button
   */
  getFetchLatestButton(): Locator {
    return this.page.locator('button:has-text("Fetch Latest"), button:has-text("Refresh")');
  }

  /**
   * Click fetch latest button
   */
  async clickFetchLatest(): Promise<void> {
    await this.getFetchLatestButton().click();
    await this.page.waitForTimeout(3000);
    await this.streamlitHelper.waitForAppReady();
  }

  /**
   * Get sentiment score value
   */
  async getSentimentScore(): Promise<number> {
    const gaugeText = await this.getSentimentGauge().textContent();
    const score = parseFloat(gaugeText || '0');
    return score;
  }

  /**
   * Get trending themes
   */
  async getTrendingThemes(): Promise<string[]> {
    const themeElements = await this.getThemeCards().locator('[data-testid="theme-name"], .theme-title').allTextContents();
    return themeElements.map(t => t.trim());
  }

  /**
   * Get sector sentiment data
   */
  async getSectorSentimentData(): Promise<Record<string, number>> {
    const rows = await this.getSectorHeatmap().locator('tr, [data-testid="sector-row"]').all();
    const result: Record<string, number> = {};

    for (const row of rows) {
      const sector = await row.locator('td:first-child, [data-testid="sector-name"]').textContent();
      const score = await row.locator('td:last-child, [data-testid="sector-score"]').textContent();
      if (sector && score) {
        result[sector.trim()] = parseFloat(score) || 0;
      }
    }

    return result;
  }

  /**
   * Verify sentiment dashboard is loaded
   */
  async verifyDashboardLoaded(): Promise<void> {
    await expect(this.getSentimentGauge()).toBeVisible();
    await expect(this.getSectorHeatmap()).toBeVisible();
    await expect(this.getNewsArticles()).toBeVisible();
  }

  /**
   * Filter by sector
   */
  async filterBySector(sector: string): Promise<void> {
    const sectorFilter = this.page.locator('select[name="sector"], [data-testid="sector-filter"]').first();
    if (await sectorFilter.count() > 0) {
      await sectorFilter.selectOption({ label: sector });
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Get news article count
   */
  async getNewsArticleCount(): Promise<number> {
    return this.getNewsArticles().locator('[data-testid="article-item"], .news-item').count();
  }
}
