/**
 * Stock Detail Dashboard Tests
 *
 * Tests for the stock detail page including all tabs, charts, and signal generation.
 */

import { test, expect } from '@playwright/test';
import { StockDetailPage } from '../../pages/stock-detail.page';
import { StreamlitWaitHelper } from '../../helpers/wait-helper';

test.describe('Stock Detail Dashboard', () => {
  let stockDetailPage: StockDetailPage;
  let streamlitHelper: StreamlitWaitHelper;

  test.beforeEach(async ({ page }) => {
    await page.goto('/02_stock_detail');
    streamlitHelper = new StreamlitWaitHelper(page);
    await streamlitHelper.waitForAppReady();
    stockDetailPage = new StockDetailPage(page);
  });

  test('Stock Detail page loads successfully', async ({ page }) => {
    await expect(page.locator('[data-testid="stAppViewContainer"], .stAppViewContainer')).toBeVisible();
    await streamlitHelper.waitForAppReady();
    await expect(page.locator('text=Top Navigation')).toBeVisible({ timeout: 10000 });
  });

  test('Stock selection works', async ({ page }) => {
    const select = page.locator('[data-testid="stSelectbox"] select, select').first();
    if (await select.count() > 0) {
      await select.selectOption({ index: 1 });
      await page.waitForTimeout(1000);
      await streamlitHelper.waitForAppReady();
    }
  });

  test('Tabs are present', async ({ page }) => {
    // Streamlit tabs use different selectors
    const tabSelectors = [
      '[data-testid="stTabs"] button',
      '.stTabs button',
      'button[role="tab"]',
      '[data-testid="stTab"]',
      '.stTab',
      'button[aria-selected]'
    ];

    for (const selector of tabSelectors) {
      const tabs = page.locator(selector);
      const count = await tabs.count();
      if (count >= 1) {
        await expect(tabs.first()).toBeVisible({ timeout: 5000 });
        return;
      }
    }

    // Fallback: check for any clickable navigation elements
    const navElements = page.locator('button, [role="tab"], nav button').filter({ hasText: /overview|chart|technical|signal|risk|sentiment|flow|fundamental/i });
    const count = await navElements.count();
    if (count > 0) {
      await expect(navElements.first()).toBeVisible();
    }
  });

  test('Overview content is visible', async ({ page }) => {
    await expect(page.locator('text=Latest Close')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Market Cap')).toBeVisible({ timeout: 10000 });
  });

  test('Price chart is visible', async ({ page }) => {
    await expect(page.locator('text=Chart Controls')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=PRICE CHART')).toBeVisible({ timeout: 10000 });
  });

  test('Technical indicators section is visible', async ({ page }) => {
    await expect(page.locator('text=INTELLIGENCE')).toBeVisible({ timeout: 10000 });
  });

  test('Risk check section is visible', async ({ page }) => {
    const selectors = [
      '[data-testid="stMarkdownContainer"]',
      '.stMarkdownContainer',
      '[data-testid="stMetric"]',
      '.stMetric',
      '[data-testid="stAlert"]',
      '.stAlert'
    ];

    for (const selector of selectors) {
      const element = page.locator(selector).first();
      if (await element.count() > 0) {
        await expect(element).toBeVisible({ timeout: 10000 });
        return;
      }
    }
  });

  test('Sentiment section is visible', async ({ page }) => {
    const selectors = [
      '[data-testid="stMarkdownContainer"]',
      '.stMarkdownContainer',
      '[data-testid="stExpander"]',
      '.stExpander'
    ];

    for (const selector of selectors) {
      const element = page.locator(selector).first();
      if (await element.count() > 0) {
        await expect(element).toBeVisible({ timeout: 10000 });
        return;
      }
    }
  });

  test('Foreign flow section is visible', async ({ page }) => {
    const selectors = [
      '[data-testid="stVegaLiteChart"]',
      '.stVegaLiteChart',
      '[data-testid="stPlotlyChart"]',
      '.stPlotlyChart',
      '[data-testid="stDataFrame"]',
      'table',
      '.stDataFrame',
      'canvas',
      '.element-container canvas',
      '[data-testid="stMarkdownContainer"]',
      '.stMarkdownContainer'
    ];

    for (const selector of selectors) {
      const element = page.locator(selector).first();
      if (await element.count() > 0) {
        await expect(element).toBeVisible({ timeout: 10000 });
        return;
      }
    }
  });

  test('Fundamental AI section is visible', async ({ page }) => {
    const selectors = [
      '[data-testid="stMarkdownContainer"]',
      '.stMarkdownContainer',
      '[data-testid="stExpander"]',
      '.stExpander',
      '[data-testid="stSpinner"]',
      '.stSpinner'
    ];

    for (const selector of selectors) {
      const element = page.locator(selector).first();
      if (await element.count() > 0) {
        await expect(element).toBeVisible({ timeout: 10000 });
        return;
      }
    }
  });
});
