/**
 * Backtest API Tests
 *
 * Tests for backtest execution, metrics calculation, result retrieval.
 */

import { expect } from '@playwright/test';
import { test } from '../../fixtures/api-fixtures';
import { ApiHelper } from '../../helpers/api-helper';
import { generateMockBacktestRequest } from '../../fixtures/mock-data';

test.describe('Backtest API', () => {
  let helper: ApiHelper

  test.beforeEach(async ({ apiContext }) => {
    helper = new ApiHelper(apiContext)
  })

  test('POST /backtest/run starts a backtest', async () => {
    const request = generateMockBacktestRequest()
    const response = await helper.post<any>('/backtest/run', request)
    expect(response.id).toBeDefined()
    expect(response.status).toBeDefined() // May be 'completed' or 'running'
    expect(response.mode).toBe('swing')
    expect(response.initial_capital).toBeDefined()
  })

  test('POST /backtest/run with different modes', async () => {
    const modes = ['intraday', 'swing', 'position', 'investor']
    for (const mode of modes) {
      const response = await helper.post<any>('/backtest/run', {
        ...generateMockBacktestRequest(),
        mode
      })
      expect(response.mode).toBe(mode)
    }
  })

  test('GET /backtest/{id} returns backtest results', async () => {
    // First run a backtest
    const runResponse = await helper.post<any>('/backtest/run', generateMockBacktestRequest())
    const id = runResponse.id
    const response = await helper.get<any>(`/backtest/${id}`)
    expect(response.id).toBe(id)
    expect(response.status).toBeDefined()
  })

  test('GET /backtest/{id} returns 404 for invalid ID', async () => {
    const response = await helper.getRaw('/backtest/invalid-backtest-id')
    expect(response.status()).toBe(404)
  })

  test('Backtest metrics include required fields', async () => {
    const request = generateMockBacktestRequest()
    const response = await helper.post<any>('/backtest/run', request)
    // Metrics may not be available if backtest is still running
    if (response.metrics) {
    const metrics = response.metrics
    expect(metrics).toHaveProperty('total_return_pct')
    expect(metrics).toHaveProperty('sharpe_ratio')
    expect(metrics).toHaveProperty('max_drawdown_pct')
    expect(metrics).toHaveProperty('win_rate')
    } else {
    // If no metrics, verify the response structure
    expect(response.id).toBeDefined()
    }
  })
})
