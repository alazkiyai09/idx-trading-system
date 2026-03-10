/**
 * Base Page Object
 *
 * Provides common functionality for all Streamlit dashboard pages.
 */

import { Page, Locator, expect } from '@playwright/test';
import { StreamlitWaitHelper, WaitHelper } from '../helpers/wait-helper';

export abstract class BasePage {
  readonly page: Page;
  readonly streamlitHelper: StreamlitWaitHelper;
  readonly waitHelper: WaitHelper;

  constructor(page: Page) {
    this.page = page;
    this.streamlitHelper = new StreamlitWaitHelper(page);
    this.waitHelper = new WaitHelper(page);
  }

  /**
   * Navigate to this page
   */
  abstract goto(): Promise<void>;

  /**
   * Wait for page to be fully loaded
   */
  async waitForLoad(): Promise<void> {
    await this.streamlitHelper.waitForAppReady();
  }

  /**
   * Take a screenshot for debugging
   */
  async takeScreenshot(name: string): Promise<void> {
    await this.page.screenshot({ path: `test-results/${name}.png`, fullPage: true });
  }

  /**
   * Get page title
   */
  async getTitle(): Promise<string> {
    return this.page.title();
  }

  /**
   * Check if element is visible
   */
  async isElementVisible(selector: string): Promise<boolean> {
    const element = this.page.locator(selector);
    return element.isVisible();
  }

  /**
   * Click element with retry
   */
  async clickWithRetry(selector: string, retries = 3): Promise<void> {
    for (let i = 0; i < retries; i++) {
      try {
        await this.page.click(selector, { timeout: 5000 });
        return;
      } catch (error) {
        if (i === retries - 1) throw error;
        await this.page.waitForTimeout(500);
      }
    }
  }

  /**
   * Fill input field
   */
  async fillInput(selector: string, value: string): Promise<void> {
    await this.page.fill(selector, value);
  }

  /**
   * Select option from dropdown
   */
  async selectOption(selector: string, value: string): Promise<void> {
    await this.page.locator(selector).selectOption({ label: value });
  }

  /**
   * Get text content
   */
  async getTextContent(selector: string): Promise<string> {
    return (await this.page.locator(selector).textContent()) || '';
  }

  /**
   * Wait for element to contain text
   */
  async waitForTextInElement(selector: string, text: string): Promise<void> {
    await this.page.locator(selector).waitFor({ state: 'visible' });
    await expect(this.page.locator(selector)).toContainText(text);
  }
}
