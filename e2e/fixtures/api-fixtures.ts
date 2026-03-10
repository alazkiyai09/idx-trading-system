/**
 * API Testing Fixtures
 *
 * Provides fixtures for testing the FastAPI backend endpoints.
 */

import { test as base, APIRequestContext } from '@playwright/test';

// API-specific fixtures
type ApiFixtures = {
  apiContext: APIRequestContext;
  apiBaseUrl: string;
};

export const test = base.extend<ApiFixtures>({
  // API request context with base URL configured
  apiContext: async ({ playwright }, use) => {
    const baseUrl = process.env.API_URL || 'http://localhost:8000';
    const context = await playwright.request.newContext({
      baseURL: baseUrl,
      extraHTTPHeaders: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });
    await use(context);
    await context.dispose();
  },

  // Base URL for reference
  apiBaseUrl: async ({}, use) => {
    await use(process.env.API_URL || 'http://localhost:8000');
  },
});

export { expect } from '@playwright/test';

/**
 * Helper to check if API is healthy before running tests
 */
export async function waitForApiHealth(
  apiContext: APIRequestContext,
  maxRetries = 10,
  delayMs = 1000
): Promise<boolean> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await apiContext.get('/health');
      if (response.ok()) {
        return true;
      }
    } catch {
      // Ignore errors and retry
    }
    await new Promise(resolve => setTimeout(resolve, delayMs));
  }
  return false;
}

/**
 * Test constants
 */
export const API_CONSTANTS = {
  DEFAULT_TIMEOUT: 30000,
  TRADING_MODES: ['intraday', 'swing', 'position', 'investr'] as const,
  LOT_SIZE: 100,
  MAX_PRICE_CHANGE_PCT: 7, // IDX daily limit
} as const;
