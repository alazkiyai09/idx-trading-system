/**
 * ML Prediction Dashboard Tests
 *
 * Tests the ML workbench page, tab navigation, and primary actions.
 */

import { test, expect } from '@playwright/test';
import { StreamlitWaitHelper } from '../../helpers/wait-helper';

test.describe('ML Prediction Dashboard', () => {
  let streamlitHelper: StreamlitWaitHelper;

  test.beforeEach(async ({ page }) => {
    await page.goto('/ml_prediction');
    streamlitHelper = new StreamlitWaitHelper(page);
    await streamlitHelper.waitForAppReady();
  });

  test('ML prediction page loads workbench content', async ({ page }) => {
    await expect(page.locator('text=ML Prediction & Analysis')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('text=ANALYSIS COCKPIT')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('text=workbench')).toBeVisible({ timeout: 15000 });
  });

  test('primary analysis actions are visible', async ({ page }) => {
    await expect(page.locator('button:has-text("Analyze All")')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('button:has-text("Run Technical Analysis")')).toBeVisible({ timeout: 15000 });

    const mlTab = page.locator('button[role="tab"]:has-text("ML Prediction"), [data-baseweb="tab"]:has-text("ML Prediction")').first();
    if (await mlTab.count() > 0) {
      await mlTab.click();
      await page.waitForTimeout(300);
    }
    await expect(page.locator('button:has-text("Get Prediction")')).toBeVisible();
  });

  test('tab navigation works across the workbench', async ({ page }) => {
    const tabLabels = ['Technical Analysis', 'ML Prediction', 'Monte Carlo', 'Comparison'];

    for (const label of tabLabels) {
      const tab = page.locator(`button[role="tab"]:has-text("${label}"), [data-baseweb="tab"]:has-text("${label}")`).first();
      if (await tab.count() > 0) {
        await tab.click();
        await page.waitForTimeout(300);
        await expect(tab).toBeVisible();
      }
    }
  });

  test('analyze all workflow completes without page failure', async ({ page }) => {
    const analyzeButton = page.locator('button:has-text("Analyze All")').first();
    await expect(analyzeButton).toBeVisible({ timeout: 15000 });
    await analyzeButton.click();
    await page.waitForTimeout(2000);
    await streamlitHelper.waitForAppReady(60000);

    await expect(page.locator('[data-testid="stException"]')).toHaveCount(0);
    await expect(page.getByText(/Traceback|Exception/i)).toHaveCount(0);
  });
});
