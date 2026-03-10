/**
 * Health API Tests
 *
 * Tests for health, data freshness, and detailed health endpoints.
 */

import { expect } from '@playwright/test';
import { test } from '../../fixtures/api-fixtures';
import { ApiHelper } from '../../helpers/api-helper';

test.describe('Health API', () => {
  let helper: ApiHelper

  test.beforeEach(async ({ apiContext }) => {
    helper = new ApiHelper(apiContext)
  })

  test('GET / returns API info', async () => {
    const response = await helper.get<any>('/')
    expect(response.name).toBe('IDX Trading System API')
    expect(response.version).toBe('3.0.0')
    expect(response.docs).toBe('/docs')
    expect(response.health).toBe('/health')
  })

  test('GET /health returns status', async () => {
    const response = await helper.get<any>('/health')
    expect(response.status).toBe('ok')
    expect(response.version).toBe('3.0.0')
    expect(response.timestamp).toBeDefined()
  })

  test('GET /health/data returns data freshness', async () => {
    const response = await helper.get<any>('/health/data')
    expect(response.status).toBeDefined()
    expect(response.warnings).toBeDefined()
    expect(Array.isArray(response.warnings)).toBe(true)
  })

  test('GET /health/detailed returns component status', async () => {
    const response = await helper.get<any>('/health/detailed')
    expect(response.status).toBeDefined()
    expect(response.timestamp).toBeDefined()
    expect(response.components).toBeDefined()
    expect(response.components.api).toBeDefined()
    expect(response.components.database).toBeDefined()
    // Check that required components are ok
    const requiredComponents = ['api', 'database']
    for (const component of requiredComponents) {
      expect(response.components[component].status).toBe('ok')
    }
  })

  test('API should respond to proper content type on errors', async () => {
    const response = await helper.getRaw('/nonexistent-endpoint')
    expect(response.status()).toBe(404)
  })
})
