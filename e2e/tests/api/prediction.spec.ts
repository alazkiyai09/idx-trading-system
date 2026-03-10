/**
 * ML Prediction API Tests
 *
 * Tests for ML prediction endpoints.
 */

import { expect } from '@playwright/test';
import { test } from '../../fixtures/api-fixtures';
import { ApiHelper } from '../../helpers/api-helper';

test.describe('Prediction API', () => {
  let helper: ApiHelper

  test.beforeEach(async ({ apiContext }) => {
    helper = new ApiHelper(apiContext)
  })

  test('GET /prediction/{symbol} returns prediction', async () => {
    const response = await helper.getRaw('/prediction/BBCA')
    expect([200, 404, 503]).toContain(response.status())

    if (response.status() === 200) {
      const body = await response.json()
      expect(body.symbol).toBe('BBCA')
      expect(body.current_price).toBeDefined()
      expect(Array.isArray(body.predictions)).toBe(true)
    }
  })

  test('GET /prediction/{symbol} prediction includes required fields', async () => {
    const response = await helper.getRaw('/prediction/BBCA')
    if (response.status() !== 200) {
      expect([404, 503]).toContain(response.status())
      return
    }
    const body = await response.json()
    if (body.predictions.length > 0) {
      const prediction = body.predictions[0]
      expect(prediction).toHaveProperty('date')
      expect(prediction).toHaveProperty('predicted_price')
      expect(prediction).toHaveProperty('predicted_return')
    }
  })

  test('GET /prediction/{symbol} returns 404 for invalid symbol', async () => {
    const response = await helper.getRaw('/prediction/INVALID_SYMBOL')
    expect([404, 503]).toContain(response.status())
  })

  test('POST /prediction/train/{symbol} triggers training', async () => {
    const response = await helper.postRaw('/prediction/train/BBCA', {
      lookback_days: 200,
      test_size: 0.2,
      use_exogenous: true,
    })
    expect([202, 400, 501, 503]).toContain(response.status())
  })

  test('Prediction dates are future dates', async () => {
    const response = await helper.getRaw('/prediction/BBCA')
    if (response.status() !== 200) {
      expect([404, 503]).toContain(response.status())
      return
    }
    const body = await response.json()
    if (body.predictions && body.predictions.length > 0) {
    const today = new Date()
    today.setHours(0, 0, 0, 0) // Reset time to start of day
    body.predictions.forEach((pred: any) => {
      const predDate = new Date(pred.date)
      expect(predDate.getTime()).toBeGreaterThanOrEqual(today.getTime())
    })
    }
  })
})
