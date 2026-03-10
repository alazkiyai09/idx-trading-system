/**
 * Sentiment API Tests
 *
 * Tests for sentiment endpoints including latest, sector aggregation, and cleanup.
 */

import { expect } from '@playwright/test';
import { test } from '../../fixtures/api-fixtures';
import { ApiHelper } from '../../helpers/api-helper';

test.describe('Sentiment API', () => {
  let helper: ApiHelper

  test.beforeEach(async ({ apiContext }) => {
    helper = new ApiHelper(apiContext)
  })

  test('GET /sentiment/latest returns latest sentiment', async () => {
    const response = await helper.get<any>('/sentiment/latest')
    expect(response.articles).toBeDefined()
    expect(Array.isArray(response.articles)).toBe(true)
  })

  test('GET /sentiment/latest with symbol filter', async () => {
    const response = await helper.get<any>('/sentiment/latest', { symbol: 'BBCA' })
    expect(response.articles).toBeDefined()
    response.articles.forEach((article: any) => {
      expect(article.symbol).toBe('BBCA')
    })
  })

  test('GET /sentiment/sector returns sector aggregation', async () => {
    const response = await helper.get<any>('/sentiment/sector')
    expect(Array.isArray(response)).toBe(true)
    response.forEach((sector: any) => {
      expect(sector.sector).toBeDefined()
      expect(sector.avg_score).toBeDefined()
    })
  })

  test('GET /sentiment/themes returns trending themes', async () => {
    const response = await helper.get<any>('/sentiment/themes')
    expect(Array.isArray(response)).toBe(true)
    response.forEach((theme: any) => {
      expect(theme.theme).toBeDefined()
      expect(theme.sector).toBeDefined()
      expect(theme.impact_direction).toBeDefined()
    })
  })

  test('POST /sentiment/fetch/{symbol} triggers sentiment fetch', async () => {
    const response = await helper.post<any>('/sentiment/fetch/BBCA')
    expect(response.status).toBe('accepted')
  })

  test('DELETE /sentiment/cleanup removes old data', async () => {
    const response = await helper.delete<any>('/sentiment/cleanup', { days: '30' })
    expect(response.status).toBeDefined()
  })

  test('Sentiment article includes required fields', async () => {
    const response = await helper.get<any>('/sentiment/latest')
    if (response.articles.length > 0) {
      const article = response.articles[0]
      expect(article).toHaveProperty('symbol')
      expect(article).toHaveProperty('article_title')
      expect(article).toHaveProperty('sentiment_score')
      expect(article).toHaveProperty('confidence')
      expect(article.sentiment_score).toBeGreaterThanOrEqual(-1)
      expect(article.sentiment_score).toBeLessThanOrEqual(1)
      expect(article.confidence).toBeGreaterThan(0)
      expect(article.confidence).toBeLessThanOrEqual(1)
    }
  })
})
