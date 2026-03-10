/**
 * Stocks API Tests
 *
 * Tests for stock listing, details, chart data, and filtering.
 */

import { expect } from '@playwright/test';
import { test } from '../../fixtures/api-fixtures';
import { ApiHelper } from '../../helpers/api-helper';

test.describe('Stocks API', () => {
  let helper: ApiHelper

  test.beforeEach(async ({ apiContext }) => {
    helper = new ApiHelper(apiContext)
  })

  test('GET /stocks lists all stocks', async () => {
    const response = await helper.get<any>('/stocks')
    expect(Array.isArray(response)).toBe(true)
    expect(response.length).toBeGreaterThan(0)
  })

  test('GET /stocks with sector filter', async () => {
    const response = await helper.get<any>('/stocks', { sector: 'Financials' })
    expect(Array.isArray(response)).toBe(true)
    response.forEach((stock: any) => {
      expect(stock.sector).toBe('Financials')
    })
  })

  test('GET /stocks with LQ45 filter', async () => {
    const response = await helper.get<any>('/stocks', { is_lq45: 'true' })
    expect(Array.isArray(response)).toBe(true)
    response.forEach((stock: any) => {
      expect(stock.is_lq45).toBe(true)
    })
  })

  test('GET /stocks with market cap filter', async () => {
    const response = await helper.get<any>('/stocks', { min_market_cap: 100000000000 })
    expect(Array.isArray(response)).toBe(true)
    response.forEach((stock: any) => {
      expect(stock.market_cap).toBeGreaterThan(100000000000)
    })
  })

  test('GET /stocks/{symbol} returns stock details', async () => {
    const response = await helper.get<any>('/stocks/BBCA')
    expect(response.symbol).toBe('BBCA')
    expect(response.name).toBeDefined()
    expect(response.sector).toBeDefined()
  })

  test('GET /stocks/{symbol} returns 404 for invalid symbol', async () => {
    const response = await helper.getRaw('/stocks/INVALID_SYMBOL')
    expect(response.status()).toBe(404)
  })

  test('GET /stocks/{symbol}/chart returns OHLCV data', async () => {
    const response = await helper.get<any>('/stocks/BBCA/chart', { days: 200 })
    expect(Array.isArray(response)).toBe(true)
    expect(response.length).toBeGreaterThan(0)
    expect(response[0]).toHaveProperty('date')
    expect(response[0]).toHaveProperty('open')
    expect(response[0]).toHaveProperty('high')
    expect(response[0]).toHaveProperty('low')
    expect(response[0]).toHaveProperty('close')
    expect(response[0]).toHaveProperty('volume')
  })

  test('GET /stocks/{symbol}/chart respects days parameter', async () => {
    const response = await helper.get<any>('/stocks/BBCA/chart', { days: 30 })
    expect(response.length).toBeLessThanOrEqual(200)
  })

  test('Stock data includes required fields', async () => {
    const response = await helper.get<any>('/stocks')
    const requiredFields = ['symbol', 'name', 'sector', 'sub_sector', 'market_cap', 'is_lq45', 'change_pct']
    for (const stock of response) {
      for (const field of requiredFields) {
        expect(stock).toHaveProperty(field)
      }
    }
  })
})
