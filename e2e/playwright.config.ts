import { defineConfig, devices } from '@playwright/test';
import path from 'path';

const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';
const DASHBOARD_BASE_URL = process.env.DASHBOARD_URL || 'http://localhost:8501';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['list'],
  ],
  use: {
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'on-first-retry',
  },

  projects: [
    // API Tests
    {
      name: 'api',
      testMatch: /api\/.*\.spec\.ts/,
      use: {
        baseURL: API_BASE_URL,
      },
    },
    // Dashboard Tests - Chromium
    {
      name: 'dashboard-chromium',
      testMatch: /dashboard\/.*\.spec\.ts/,
      use: {
        baseURL: DASHBOARD_BASE_URL,
        ...devices['Desktop Chrome'],
      },
    },
    // Dashboard Tests - Firefox
    {
      name: 'dashboard-firefox',
      testMatch: /dashboard\/.*\.spec\.ts/,
      use: {
        baseURL: DASHBOARD_BASE_URL,
        ...devices['Desktop Firefox'],
      },
    },
  ],

  // Optional: Auto-start servers for local development
  // webServer: [
  //   {
  //     command: 'uvicorn api.main:app --port 8000',
  //     url: 'http://localhost:8000/health',
  //     reuseExistingServer: !process.env.CI,
  //     timeout: 120 * 1000,
  //   },
  //   {
  //     command: 'streamlit run dashboard/app.py --server.port 8501',
  //     url: 'http://localhost:8501',
  //     reuseExistingServer: !process.env.CI,
  //     timeout: 120 * 1000,
  //   },
  // ],
});
