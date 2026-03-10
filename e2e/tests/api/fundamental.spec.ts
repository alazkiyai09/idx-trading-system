/**
 * Fundamental Analysis API Tests
 *
 * Tests for multi-agent fundamental analysis.
 */

import { expect } from '@playwright/test';
import { test } from '../../fixtures/api-fixtures';
import { ApiHelper } from '../../helpers/api-helper';

test.describe('Fundamental API', () => {
  let helper: ApiHelper

  test.beforeEach(async ({ apiContext }) => {
    helper = new ApiHelper(apiContext)
  })

  test('POST /fundamental/analyze runs analysis', async () => {
    const request = {
      symbol: 'BBCA'
    }
    const response = await helper.post<any>('/fundamental/analyze', request)
    expect(response.symbol).toBe('BBCA')
    expect(response.overall_score).toBeDefined()
    expect(response.recommendation).toBeDefined()
    expect(response.confidence).toBeDefined()
  })

  test('POST /fundamental/analyze with PDF path', async () => {
    const request = {
      symbol: 'BBRI',
      pdf_path: '/path/to/report.pdf'
    }
    const response = await helper.post<any>('/fundamental/analyze', request)
    expect(response.symbol).toBeDefined()
  })

  test('Fundamental response includes required fields', async () => {
    const request = {
      symbol: 'TLKM'
    }
    const response = await helper.post<any>('/fundamental/analyze', request)
    expect(response).toHaveProperty('symbol')
    expect(response).toHaveProperty('overall_score')
    expect(response).toHaveProperty('recommendation')
    expect(response).toHaveProperty('confidence')
    expect(response).toHaveProperty('agent_reports')
    expect(response).toHaveProperty('veto_triggered')
  })
})
