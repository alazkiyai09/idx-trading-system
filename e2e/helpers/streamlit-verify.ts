/**
 * Streamlit Verification Utility
 *
 * Provides helper functions for verifying Streamlit-specific UI elements.
 */

import { Page, Locator } from '@playwright/test';

export interface StreamlitElementInfo {
  selector: string;
  count: number;
  isVisible: boolean;
  tagName?: string;
  text?: string;
}

export class StreamlitVerify {
  constructor(private page: Page) {}

  /**
   * Common Streamlit element selectors
   */
  static readonly SELECTORS = {
    // Containers
    APP_CONTAINER: '[data-testid="stAppViewContainer"], .stAppViewContainer',
    MAIN_CONTENT: '[data-testid="stMainBlock"], .stMainBlock',
    SIDEBAR: '[data-testid="stSidebar"], .stSidebar',
    HEADER: '[data-testid="stHeader"], .stHeader',

    // Data display
    METRIC: '[data-testid="stMetric"], .stMetric',
    DATA_FRAME: '[data-testid="stDataFrame"], table, .stDataFrame',
    DATA_EDITOR: '[data-testid="stDataEditor"], .stDataEditor',
    MARKDOWN: '[data-testid="stMarkdownContainer"], .stMarkdownContainer',
    CAPTION: '[data-testid="stCaption"], .stCaption',

    // Charts
    VEGA_CHART: '[data-testid="stVegaLiteChart"], .stVegaLiteChart',
    PLOTLY_CHART: '[data-testid="stPlotlyChart"], .stPlotlyChart',
    ALTAR_CHART: '[data-testid="stAltairChart"], .stAltairChart',
    PYLOT_CHART: '[data-testid="stPyplotChart"], .stPyplotChart',
    GRAPHVIZ: '[data-testid="stGraphvizChart"], .stGraphvizChart',
    CANVAS: 'canvas',

    // Inputs
    SELECTBOX: '[data-testid="stSelectbox"], .stSelectbox',
    MULTISELECT: '[data-testid="stMultiSelect"], .stMultiSelect',
    NUMBER_INPUT: '[data-testid="stNumberInput"], .stNumberInput',
    TEXT_INPUT: '[data-testid="stTextInput"], .stTextInput',
    SLIDER: '[data-testid="stSlider"], .stSlider',
    CHECKBOX: '[data-testid="stCheckbox"], .stCheckbox',
    RADIO: '[data-testid="stRadio"], .stRadio',
    DATE_INPUT: '[data-testid="stDateInput"], .stDateInput',
    FILE_UPLOADER: '[data-testid="stFileUploader"], .stFileUploader',

    // Buttons
    BUTTON: '[data-testid="stBaseButton-secondary"], button, .stButton',
    FORM_SUBMIT: '[data-testid="stFormSubmitButton"], .stFormSubmitButton',

    // Layout
    COLUMNS: '[data-testid="stHorizontalBlock"], .stHorizontalBlock',
    EXPANDER: '[data-testid="stExpander"], .stExpander',
    TABS: '[data-testid="stTabs"], .stTabs',
    FORM: '[data-testid="stForm"], form',

    // Feedback
    ERROR: '[data-testid="stError"], .stError',
    WARNING: '[data-testid="stWarning"], .stWarning',
    SUCCESS: '[data-testid="stSuccess"], .stSuccess',
    INFO: '[data-testid="stInfo"], .stInfo',
    SPINNER: '[data-testid="stSpinner"], .stSpinner',
    TOAST: '[data-testid="stToast"], .stToast'
  };

  /**
   * Find any visible element from a list of selectors
   */
  async findFirstVisible(selectors: string[]): Promise<Locator | null> {
    for (const selector of selectors) {
      const element = this.page.locator(selector).first();
      if (await element.count() > 0) {
        return element;
      }
    }
    return null;
  }

  /**
   * Verify at least one element from selectors is visible
   */
  async verifyAnyVisible(selectors: string[], timeout = 5000): Promise<boolean> {
    const element = await this.findFirstVisible(selectors);
    if (element) {
      await element.waitFor({ state: 'visible', timeout });
      return true;
    }
    return false;
  }

  /**
   * Get info about elements matching selectors
   */
  async getElementInfo(selectors: string[]): Promise<StreamlitElementInfo[]> {
    const results: StreamlitElementInfo[] = [];

    for (const selector of selectors) {
      const elements = this.page.locator(selector);
      const count = await elements.count();

      if (count > 0) {
        const first = elements.first();
        results.push({
          selector,
          count,
          isVisible: await first.isVisible(),
          tagName: await first.evaluate(el => el.tagName),
          text: (await first.textContent()) ?? undefined
        });
      }
    }

    return results;
  }

  /**
   * Wait for Streamlit app to be ready
   */
  async waitForStreamlitReady(timeout = 30000): Promise<void> {
    // Wait for the main content area to be present
    await this.page.waitForSelector(
      StreamlitVerify.SELECTORS.APP_CONTAINER,
      { state: 'visible', timeout }
    );

    // Wait for any spinners to disappear
    const spinner = this.page.locator(StreamlitVerify.SELECTORS.SPINNER);
    const spinnerCount = await spinner.count();
    if (spinnerCount > 0) {
      await spinner.last().waitFor({ state: 'hidden', timeout }).catch(() => {});
    }
  }

  /**
   * Get all visible metrics on the page
   */
  async getMetrics(): Promise<Locator[]> {
    const metrics = this.page.locator(StreamlitVerify.SELECTORS.METRIC);
    const count = await metrics.count();
    const result: Locator[] = [];
    for (let i = 0; i < count; i++) {
      result.push(metrics.nth(i));
    }
    return result;
  }

  /**
   * Get all charts on the page
   */
  async getCharts(): Promise<Locator[]> {
    const chartSelectors = [
      StreamlitVerify.SELECTORS.VEGA_CHART,
      StreamlitVerify.SELECTORS.PLOTLY_CHART,
      StreamlitVerify.SELECTORS.ALTAR_CHART,
      StreamlitVerify.SELECTORS.CANVAS
    ];

    const charts: Locator[] = [];
    for (const selector of chartSelectors) {
      const elements = this.page.locator(selector);
      const count = await elements.count();
      for (let i = 0; i < count; i++) {
        charts.push(elements.nth(i));
      }
    }
    return charts;
  }

  /**
   * Check if page has any error messages
   */
  async hasErrors(): Promise<boolean> {
    const error = this.page.locator(StreamlitVerify.SELECTORS.ERROR);
    return await error.count() > 0;
  }

  /**
   * Get error messages on the page
   */
  async getErrors(): Promise<string[]> {
    const errors = this.page.locator(StreamlitVerify.SELECTORS.ERROR);
    const count = await errors.count();
    const messages: string[] = [];
    for (let i = 0; i < count; i++) {
      messages.push(await errors.nth(i).textContent() || '');
    }
    return messages;
  }

  /**
   * Take a screenshot with a descriptive name
   */
  async takeScreenshot(name: string): Promise<Buffer> {
    return await this.page.screenshot({ path: `test-results/${name}.png`, fullPage: true });
  }

  /**
   * Debug: Print all visible elements
   */
  async debugPrintVisibleElements(): Promise<void> {
    console.log('\n=== Visible Streamlit Elements ===');

    for (const [name, selector] of Object.entries(StreamlitVerify.SELECTORS)) {
      const count = await this.page.locator(selector).count();
      if (count > 0) {
        console.log(`${name}: ${count} elements`);
      }
    }

    console.log('================================\n');
  }
}
