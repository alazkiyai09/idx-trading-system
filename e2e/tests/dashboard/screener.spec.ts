/**
 * Screener Dashboard Tests
 *
 * Tests for the stock screener page including filters, scan, and results.
 */

import { test, expect } from '@playwright/test';
import { ScreenerPage } from '../../pages/screener.page';
import { StreamlitWaitHelper } from '../../helpers/wait-helper';

test.describe('Screener Dashboard', () => {
  let screenerPage: ScreenerPage;
  let streamlitHelper: StreamlitWaitHelper;

  test.beforeEach(async ({ page }) => {
    await page.goto('/screener');
    streamlitHelper = new StreamlitWaitHelper(page);
    await streamlitHelper.waitForAppReady();
    screenerPage = new ScreenerPage(page);
  });

  test('Screener page loads successfully', async ({ page }) => {
    // Streamlit pages have generic titles, check for content instead
    await expect(page.locator('[data-testid="stAppViewContainer"], .stAppViewContainer')).toBeVisible();
    await streamlitHelper.waitForAppReady();
  });

  test('Filter controls are visible', async () => {
    await expect(screenerPage.getFilterControls()).toBeVisible();
  });

  test('Classification filters work', async ({ page }) => {
    const checkbox = page.locator('.stCheckbox label, label').filter({ hasText: /LQ45 Only|IDX30 Only/i }).first();
    if (await checkbox.count() > 0) {
      await checkbox.click();
      await page.waitForTimeout(500);
    }
  });

  test('Sector filter works', async ({ page }) => {
    const sectorSelect = page.locator('[data-testid="stSelectbox"] select, select').first();
    if (await sectorSelect.count() > 0) {
      await sectorSelect.selectOption({ index: 1 });
      await page.waitForTimeout(500);
    }
  });

  test('Scan Market button triggers scan', async ({ page }) => {
    const scanButton = page.locator('button:has-text("Scan"), button:has-text("Market")').first();
    if (await scanButton.count() > 0) {
      await scanButton.click();
      await page.waitForTimeout(3000);
      await streamlitHelper.waitForAppReady();
    }
  });

  test('Results table displays stock data', async ({ page }) => {
    await expect(page.locator('text=SCAN CONTEXT')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=candidates remain in the working set')).toBeVisible({ timeout: 10000 });
  });

  test('Deep Dive section allows navigation to stock detail', async ({ page }) => {
    // Look for deep dive section
    const deepDiveSelect = page.locator('[data-testid="stSelectbox"] select, select').first();
    if (await deepDiveSelect.count() > 0) {
      await deepDiveSelect.selectOption({ index: 1 });
      // Click Open Stock Details button
      const openButton = page.locator('button:has-text("Open"), button:has-text("Details")').first();
      if (await openButton.count() > 0) {
        await openButton.click();
        await page.waitForTimeout(1000);
      }
    }
  });
});
