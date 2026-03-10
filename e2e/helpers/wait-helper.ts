/**
 * Wait Helpers
 *
 * Utilities for waiting on specific conditions, especially for Streamlit apps.
 */

import { Page, Locator } from '@playwright/test';

/**
 * Streamlit-specific wait utilities
 */
export class StreamlitWaitHelper {
  constructor(private page: Page) {}

  /**
   * Wait for Streamlit app to fully load
   * Streamlit apps show a loading indicator that we need to wait for
   */
  async waitForAppReady(timeout = 30000): Promise<void> {
    // Wait for the main Streamlit container
    await this.page.waitForSelector('[data-testid="stApp"]', { timeout });
    await this.page.waitForSelector('[data-testid="stMainBlockContainer"]', { timeout });

    // Wait for any loading spinners to disappear
    await this.page.waitForFunction(
      () => {
        const loadingElements = document.querySelectorAll(
          '[data-testid="stSpinner"], .stSpinner, [class*="loading"]'
        );
        const mainContainer = document.querySelector('[data-testid="stMainBlockContainer"]');
        const hasMainText = (mainContainer?.textContent || '').trim().length > 0;
        const hasException = document.querySelector('[data-testid="stException"]') !== null;
        return loadingElements.length === 0 && hasMainText && !hasException;
      },
      { timeout }
    );

    // Additional wait for animations to settle
    await this.page.waitForTimeout(500);
  }

  /**
   * Wait for a Plotly chart to render
   */
  async waitForChartRender(selector: string, timeout = 15000): Promise<void> {
    // Wait for the chart container
    await this.page.waitForSelector(selector, { timeout });

    // Wait for Plotly to inject the chart
    await this.page.waitForFunction(
      (sel) => {
        const container = document.querySelector(sel);
        if (!container) return false;
        // Plotly adds a .plotly-graph-div class when chart is rendered
        const plotlyDiv = container.querySelector('.plotly-graph-div, .js-plotly-plot');
        return plotlyDiv !== null;
      },
      selector,
      { timeout }
    );
  }

  /**
   * Wait for a data table to load
   */
  async waitForTableLoad(selector: string, minRows = 1, timeout = 15000): Promise<void> {
    await this.page.waitForSelector(selector, { timeout });

    await this.page.waitForFunction(
      ({ sel, min }) => {
        const table = document.querySelector(sel);
        if (!table) return false;
        // Check for rows in either a standard table or ag-grid
        const rows = table.querySelectorAll('tr, .ag-row');
        return rows.length >= min;
      },
      { sel: selector, min: minRows },
      { timeout }
    );
  }

  /**
   * Wait for Streamlit's st.success/st.error/st.warning to appear
   */
  async waitForToast(
    type: 'success' | 'error' | 'warning' | 'info',
    timeout = 5000
  ): Promise<Locator> {
    const selectors = {
      success: '[data-testid="stSuccess"]',
      error: '[data-testid="stError"]',
      warning: '[data-testid="stWarning"]',
      info: '[data-testid="stInfo"]',
    };

    return this.page.locator(selectors[type]).first();
  }

  /**
   * Wait for sidebar to be visible
   */
  async waitForSidebar(timeout = 10000): Promise<void> {
    await this.page.waitForSelector('[data-testid="stSidebar"]', { timeout });
  }

  /**
   * Wait for a specific text to appear anywhere on the page
   */
  async waitForText(text: string, timeout = 10000): Promise<void> {
    await this.page.waitForSelector(`text="${text}"`, { timeout });
  }

  /**
   * Wait for element to be stable (not moving)
   */
  async waitForElementStable(selector: string, timeout = 5000): Promise<void> {
    await this.page.waitForSelector(selector, { state: 'visible', timeout });

    let lastPosition = { x: 0, y: 0 };
    let stableCount = 0;

    await this.page.waitForFunction(
      ({ sel, lastPos, stable, requiredStable }) => {
        const element = document.querySelector(sel);
        if (!element) return false;

        const rect = element.getBoundingClientRect();
        const currentPos = { x: rect.x, y: rect.y };

        if (currentPos.x === lastPos.x && currentPos.y === lastPos.y) {
          stable++;
        } else {
          stable = 0;
        }

        return stable >= requiredStable;
      },
      { sel: selector, lastPos: lastPosition, stable: stableCount, requiredStable: 3 },
      { timeout }
    );
  }
}

/**
 * General wait utilities
 */
export class WaitHelper {
  constructor(private page: Page) {}

  /**
   * Wait for API response matching a URL pattern
   */
  async waitForApiResponse(urlPattern: string | RegExp, timeout = 30000): Promise<any> {
    const response = await this.page.waitForResponse(
      (response) => {
        const url = response.url();
        if (typeof urlPattern === 'string') {
          return url.includes(urlPattern);
        }
        return urlPattern.test(url);
      },
      { timeout }
    );
    return response.json();
  }

  /**
   * Wait for multiple API responses
   */
  async waitForApiResponses(
    patterns: Array<string | RegExp>,
    timeout = 30000
  ): Promise<any[]> {
    const promises = patterns.map((pattern) =>
      this.page.waitForResponse(
        (response) => {
          const url = response.url();
          if (typeof pattern === 'string') {
            return url.includes(pattern);
          }
          return pattern.test(url);
        },
        { timeout }
      ).then((r) => r.json())
    );

    return Promise.all(promises);
  }

  /**
   * Wait for navigation to complete
   */
  async waitForNavigation(timeout = 10000): Promise<void> {
    await this.page.waitForLoadState('networkidle', { timeout });
  }
}

/**
 * Create wait helpers for a page
 */
export function createWaitHelpers(page: Page) {
  return {
    streamlit: new StreamlitWaitHelper(page),
    general: new WaitHelper(page),
  };
}
