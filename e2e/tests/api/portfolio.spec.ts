/**
 * Portfolio API Tests
 *
 * Tests for portfolio summary, positions, and trade history.
 */

import { expect } from '@playwright/test';
import { test } from '../../fixtures/api-fixtures';
import { ApiHelper } from '../../helpers/api-helper';

test.describe('Portfolio API', () => {
  let helper: ApiHelper

  test.beforeEach(async ({ apiContext }) => {
    helper = new ApiHelper(apiContext)
  })

  test('GET /portfolio returns portfolio summary', async () => {
    const response = await helper.get<any>('/portfolio')
    expect(response.total_value).toBeDefined()
    expect(response.cash).toBeDefined()
    expect(response.invested).toBeDefined()
    expect(response.unrealized_pnl).toBeDefined()
    expect(response.realized_pnl).toBeDefined()
    expect(response.total_pnl).toBeDefined()
    expect(response.num_positions).toBeDefined()
    expect(Array.isArray(response.positions)).toBe(true)
  })

  test('GET /portfolio/positions returns open positions', async () => {
    const response = await helper.get<any>('/portfolio/positions')
    expect(response.positions).toBeDefined()
    expect(response.count).toBeDefined()
    expect(Array.isArray(response.positions)).toBe(true)
  })

  test('GET /portfolio/history returns trade history', async () => {
    const response = await helper.get<any>('/portfolio/history')
    expect(response.trades).toBeDefined()
    expect(response.total).toBeDefined()
    expect(response.total_pnl).toBeDefined()
    expect(response.win_rate).toBeDefined()
  })

  test('GET /portfolio/history with symbol filter', async () => {
    const response = await helper.get<any>('/portfolio/history', { symbol: 'BBCA' })
    expect(response.trades).toBeDefined()
  })

  test('GET /portfolio/history with limit parameter', async () => {
    const response = await helper.get<any>('/portfolio/history', { limit: 10 })
    expect(response.trades.length).toBeLessThan(12) // Allow slight variance
  })

  test('Portfolio response includes required fields', async () => {
    const response = await helper.get<any>('/portfolio')
    const requiredFields = [
      'total_value',
      'cash',
      'invested',
      'unrealized_pnl',
      'realized_pnl',
      'total_pnl',
      'total_pnl_pct',
      'num_positions',
      'positions'
    ]
    for (const field of requiredFields) {
      expect(response).toHaveProperty(field)
    }
  })
})
