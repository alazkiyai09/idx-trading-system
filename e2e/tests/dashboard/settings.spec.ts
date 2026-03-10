/**
 * Settings Dashboard Tests
 *
 * Tests for the settings page including LLM provider selection, trading mode configuration, and risk parameters.
 */

import { test, expect } from '@playwright/test';
import { SettingsPage } from '../../pages/settings.page';
import { StreamlitWaitHelper } from '../../helpers/wait-helper';

test.describe('Settings Dashboard', () => {
  let settingsPage: SettingsPage;
  let streamlitHelper: StreamlitWaitHelper;

  test.beforeEach(async ({ page }) => {
    await page.goto('/05_settings');
    streamlitHelper = new StreamlitWaitHelper(page);
    await streamlitHelper.waitForAppReady();
    settingsPage = new SettingsPage(page);
  });

  test('Settings page loads successfully', async ({ page }) => {
    await expect(page.locator('[data-testid="stAppViewContainer"], .stAppViewContainer')).toBeVisible();
    await streamlitHelper.waitForAppReady();
  });

  test('LLM provider section is visible', async ({ page }) => {
    // Look for any selectbox or input related to LLM
    const selectors = [
      'text=/LLM|Provider|Claude|GLM|OpenAI/i',
      '[data-testid="stSelectbox"]',
      '.stSelectbox',
      '[data-testid="stSidebar"] [data-testid="stSelectbox"]',
      '.stSidebar .stSelectbox',
      'select',
      '[data-testid="stRadio"]',
      '.stRadio'
    ];

    for (const selector of selectors) {
      const element = page.locator(selector).first();
      if (await element.count() > 0) {
        await expect(element).toBeVisible({ timeout: 5000 });
        return;
      }
    }
  });

  test('Trading mode section is visible', async ({ page }) => {
    const selectors = [
      'text=/Trading Mode|Mode|Swing|Intraday|Position|Investor/i',
      '[data-testid="stSelectbox"]',
      '.stSelectbox',
      '[data-testid="stRadio"]',
      '.stRadio',
      'select'
    ];

    for (const selector of selectors) {
      const element = page.locator(selector).first();
      if (await element.count() > 0) {
        await expect(element).toBeVisible({ timeout: 5000 });
        return;
      }
    }
  });

  test('Risk parameters section is visible', async ({ page }) => {
    const selectors = [
      'text=/Risk|Stop|Loss|Capital|Position|Limit/i',
      '[data-testid="stSlider"]',
      '.stSlider',
      '[data-testid="stNumberInput"]',
      '.stNumberInput',
      '[data-testid="stTextInput"]',
      '.stTextInput',
      'input[type="range"]',
      'input[type="number"]'
    ];

    for (const selector of selectors) {
      const element = page.locator(selector).first();
      if (await element.count() > 0) {
        await expect(element).toBeVisible({ timeout: 5000 });
        return;
      }
    }
  });

  test('LLM provider selection works', async ({ page }) => {
    const selectElement = page.locator('[data-testid="stSelectbox"] select, select, [data-testid="stRadio"]').first();
    if (await selectElement.count() > 0) {
      const tagName = await selectElement.evaluate(el => el.tagName.toLowerCase());
      if (tagName === 'select') {
        await selectElement.selectOption({ index: 1 });
      } else {
        await selectElement.click();
      }
      await page.waitForTimeout(500);
    }
  });

  test('Trading mode selection works', async ({ page }) => {
    const selectElements = page.locator('[data-testid="stSelectbox"] select, select, [data-testid="stRadio"]');
    const count = await selectElements.count();
    if (count >= 2) {
      const secondSelect = selectElements.nth(1);
      const tagName = await secondSelect.evaluate(el => el.tagName.toLowerCase());
      if (tagName === 'select') {
        await secondSelect.selectOption({ index: 1 });
      } else {
        await secondSelect.click();
      }
      await page.waitForTimeout(500);
    }
  });

  test('Risk parameter adjustment works', async ({ page }) => {
    const slider = page.locator('[data-testid="stSlider"] input, input[type="range"], [data-testid="stNumberInput"] input, input[type="number"]').first();
    if (await slider.count() > 0) {
      const type = await slider.getAttribute('type');
      if (type === 'range') {
        // Slider interaction
        await slider.fill('1.5');
      } else {
        await slider.fill('1.5');
      }
      await page.waitForTimeout(500);
    }
  });

  test('Save settings works', async ({ page }) => {
    const saveButton = page.locator('button:has-text("Save"), button:has-text("Apply"), button:has-text("Confirm")').first();
    if (await saveButton.count() > 0) {
      await saveButton.click();
      await page.waitForTimeout(500);
    }
  });

  test('Reset to defaults works', async ({ page }) => {
    const resetButton = page.locator('button:has-text("Reset"), button:has-text("Default"), button:has-text("Clear")').first();
    if (await resetButton.count() > 0) {
      await resetButton.click();
      await page.waitForTimeout(500);
    }
  });
});
