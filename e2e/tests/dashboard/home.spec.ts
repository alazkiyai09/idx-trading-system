/**
 * Home Dashboard Tests
 *
 * Tests for the landing page command center, desk actions, and top navigation.
 */

import { test, expect } from '@playwright/test';
import { StreamlitWaitHelper } from '../../helpers/wait-helper';

test.describe('Home Dashboard', () => {
  let streamlitHelper: StreamlitWaitHelper;

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    streamlitHelper = new StreamlitWaitHelper(page);
    await streamlitHelper.waitForAppReady();
  });

  test('home page loads command center content', async ({ page }) => {
    await expect(page.locator('text=IDX Trading Dashboard')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-top-nav-link[href="/market_overview"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('text=COMMAND CENTER')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('text=MARKET PULSE')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('text=TRADE DESK PRIORITIES')).toBeVisible({ timeout: 15000 });
  });

  test('desk actions and status strip are available', async ({ page }) => {
    await expect(page.locator('text=Desk Actions')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-link-pill[href="/screener"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-link-pill[href="/virtual_trading"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-link-pill[href="/ml_prediction"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-link-pill[href="/settings"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('text=System')).toBeVisible({ timeout: 15000 });
  });

  test('module cards and operating notes are visible', async ({ page }) => {
    await expect(page.getByText('Modules', { exact: true })).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-module-link[href="/market_overview"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-module-link[href="/screener"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-module-link[href="/sentiment"]')).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('Scan', { exact: true })).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('Validate', { exact: true })).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('Simulate', { exact: true })).toBeVisible({ timeout: 15000 });
  });

  test('top navigation renders core destinations', async ({ page }) => {
    await expect(page.locator('[data-testid="idx-top-nav-active"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-top-nav-link[href="/market_overview"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-top-nav-link[href="/screener"]')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.idx-top-nav-link[href="/virtual_trading"]')).toBeVisible({ timeout: 15000 });
  });
});
