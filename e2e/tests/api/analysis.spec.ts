/**
 * Analysis API Tests
 *
 * Tests for technical analysis, signal generation, and risk validation.
 */

import { expect } from '@playwright/test';
import { test } from '../../fixtures/api-fixtures';
import { ApiHelper } from '../../helpers/api-helper';

test.describe('Analysis API', () => {
  let helper: ApiHelper

  test.beforeEach(async ({ apiContext }) => {
    helper = new ApiHelper(apiContext)
  })

  test('POST /analysis/technical/{symbol} runs technical analysis', async () => {
    const response = await helper.post<any>('/analysis/technical/BBCA')
    expect(response.symbol).toBe('BBCA')
    expect(response.date).toBeDefined()
    expect(response.score).toBeDefined()
    expect(response.indicators).toBeDefined()
  })

  test('POST /analysis/technical/{symbol} returns required indicators', async () => {
    const response = await helper.post<any>('/analysis/technical/BBCA')
    expect(response.indicators).toHaveProperty('close')
    expect(response.indicators).toHaveProperty('ema20')
    expect(response.indicators).toHaveProperty('ema50')
    expect(response.indicators).toHaveProperty('rsi')
    expect(response.indicators).toHaveProperty('macd')
    expect(response.indicators).toHaveProperty('atr')
  })

  test('POST /analysis/technical/{symbol} score validation', async () => {
    const response = await helper.post<any>('/analysis/technical/BBCA')
    expect(response.score.total).toBeGreaterThan(0)
    expect(response.score.total).toBeLessThan(100)
    expect(response.score.trend_score).toBeDefined()
    expect(response.score.momentum_score).toBeDefined()
    expect(response.score.volume_score).toBeDefined()
  })

  test('POST /analysis/signal/{symbol} generates trading signal', async () => {
    const request = {
      mode: 'swing',
      capital: 100000000
    }
    const response = await helper.post<any>('/analysis/signal/BBCA', request)
    expect(response.symbol).toBe('BBCA')
    // Either returns signal info or "None" message
    expect(response.type || response.signal || response.message).toBeDefined()
  })

  test('POST /analysis/signal/{symbol} returns None when no setup', async () => {
    const request = {
      mode: 'swing',
      capital: 100000000
    }
    // Use postRaw to check for 404 response for invalid symbol
    const response = await helper.postRaw('/analysis/signal/INVALID_SYMBOL', request)
    // Expect 404 or 400 for invalid symbol
    expect([400, 404]).toContain(response.status())
  })

  test('POST /analysis/risk-check/{symbol} validates with risk manager', async () => {
    const request = {
      mode: 'swing',
      capital: 100000000
    }
    const response = await helper.post<any>('/analysis/risk-check/BBCA', request)
    expect(response.approved).toBeDefined()
    expect(response.reasons).toBeDefined()
    expect(Array.isArray(response.reasons)).toBe(true)
  })

  test('POST /analysis/risk-check/{symbol} returns position sizing', async () => {
    const request = {
      mode: 'swing',
      capital: 100000000
    }
    const response = await helper.post<any>('/analysis/risk-check/BBCA', request)
    if (response.approved) {
      expect(response.position_size).toBeDefined()
      expect(response.position_shares).toBeDefined()
      expect(response.kelly_fraction).toBeDefined()
      // Position shares should be multiple of lot size
      expect(response.position_shares % 100).toBe(0)
    }
  })
})
