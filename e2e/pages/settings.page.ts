/**
 * Settings Page Object
 *
 * Handles interactions with the Settings dashboard page.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';

export class SettingsPage extends BasePage {
  readonly pageUrl = '/05_settings';

  constructor(page: Page) {
    super(page);
  }

  async goto(): Promise<void> {
    await this.page.goto(this.pageUrl);
    await this.waitForLoad();
  }

  /**
   * Get LLM provider section
   */
  getLLMProviderSection(): Locator {
    return this.page.locator('[data-testid="stSelectbox"], .stSelectbox').first();
  }

  /**
   * Get trading mode section
   */
  getTradingModeSection(): Locator {
    return this.page.locator('[data-testid="stSelectbox"], .stSelectbox').nth(1);
  }

  /**
   * Get risk parameters section
   */
  getRiskParametersSection(): Locator {
    return this.page.locator('[data-testid="stSlider"], .stSlider, [data-testid="stNumberInput"], .stNumberInput').first();
  }

  /**
   * Get LLM provider selector
   */
  getLLMProviderSelect(): Locator {
    return this.page.locator('select[name="llm_provider"], [data-testid="llm-provider"]').first();
  }

  /**
   * Get trading mode selector
   */
  getTradingModeSelect(): Locator {
    return this.page.locator('select[name="trading_mode"], [data-testid="trading-mode"]').first();
  }

  /**
   * Get risk per trade input
   */
  getRiskPerTradeInput(): Locator {
    return this.page.locator('input[name="risk_per_trade"], [data-testid="risk-per-trade"]').first();
  }

  /**
   * Get max position input
   */
  getMaxPositionInput(): Locator {
    return this.page.locator('input[name="max_position"], [data-testid="max-position"]').first();
  }

  /**
   * Get save settings button
   */
  getSaveButton(): Locator {
    return this.page.locator('button:has-text("Save"), button:has-text("Apply")');
  }

  /**
   * Select LLM provider
   */
  async selectLLMProvider(provider: 'claude' | 'glm' | 'openai'): Promise<void> {
    const select = this.getLLMProviderSelect();
    if (await select.count() > 0) {
      await select.selectOption({ label: provider });
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Select trading mode
   */
  async selectTradingMode(mode: 'intraday' | 'swing' | 'position' | 'investor'): Promise<void> {
    const select = this.getTradingModeSelect();
    if (await select.count() > 0) {
      await select.selectOption({ label: mode });
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Set risk per trade
   */
  async setRiskPerTrade(riskPct: number): Promise<void> {
    const input = this.getRiskPerTradeInput();
    if (await input.count() > 0) {
      await input.fill(riskPct.toString());
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Set max position percentage
   */
  async setMaxPosition(pct: number): Promise<void> {
    const input = this.getMaxPositionInput();
    if (await input.count() > 0) {
      await input.fill(pct.toString());
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Save settings
   */
  async saveSettings(): Promise<void> {
    await this.getSaveButton().click();
    await this.page.waitForTimeout(500);
    await this.streamlitHelper.waitForAppReady();
  }

  /**
   * Verify settings are saved
   */
  async verifySettingsSaved(): Promise<void> {
    await this.streamlitHelper.waitForToast('success');
  }

  /**
   * Get API key input for provider
   */
  getApiKeyInput(provider: string): Locator {
    return this.page.locator(`input[name="${provider}_api_key"], [data-testid="${provider}-api-key"]`).first();
  }

  /**
   * Set API key for provider
   */
  async setApiKey(provider: string, apiKey: string): Promise<void> {
    const input = this.getApiKeyInput(provider);
    if (await input.count() > 0) {
      await input.fill(apiKey);
    }
  }

  /**
   * Get current settings values
   */
  async getCurrentSettings(): Promise<{
    llmProvider: string;
    tradingMode: string;
    riskPerTrade: number;
    maxPosition: number;
  }> {
    const llmProvider = await this.getLLMProviderSelect().inputValue() || '';
    const tradingMode = await this.getTradingModeSelect().inputValue() || '';
    const riskPerTrade = parseFloat(await this.getRiskPerTradeInput().inputValue() || '0');
    const maxPosition = parseFloat(await this.getMaxPositionInput().inputValue() || '0');

    return {
      llmProvider,
      tradingMode,
      riskPerTrade,
      maxPosition
    };
  }
}
