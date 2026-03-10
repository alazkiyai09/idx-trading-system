/**
 * API Request Helpers
 *
 * Utility functions for making API requests in tests.
 */

import { APIRequestContext, APIResponse } from '@playwright/test';

/**
 * API Helper class for common API operations
 */
export class ApiHelper {
  constructor(private request: APIRequestContext) {}

  /**
   * Make a GET request and return parsed JSON
   */
  async get<T = any>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
    const response = await this.request.get(path, { params });
    return this.handleResponse<T>(response);
  }

  /**
   * Make a POST request with JSON body and return parsed JSON
   */
  async post<T = any>(path: string, body?: any): Promise<T> {
    const response = await this.request.post(path, { data: body });
    return this.handleResponse<T>(response);
  }

  /**
   * Make a PUT request with JSON body
   */
  async put<T = any>(path: string, body?: any): Promise<T> {
    const response = await this.request.put(path, { data: body });
    return this.handleResponse<T>(response);
  }

  /**
   * Make a DELETE request
   */
  async delete<T = any>(path: string, params?: Record<string, string>): Promise<T> {
    const response = await this.request.delete(path, { params });
    return this.handleResponse<T>(response);
  }

  /**
   * Handle API response and throw descriptive errors
   */
  private async handleResponse<T>(response: APIResponse): Promise<T> {
    if (!response.ok()) {
      let errorDetail: string;
      try {
        const errorBody = await response.json();
        errorDetail = errorBody.detail || errorBody.error || JSON.stringify(errorBody);
      } catch {
        errorDetail = await response.text();
      }
      throw new Error(`API Error ${response.status()}: ${errorDetail}`);
    }
    return response.json();
  }

  /**
   * Get raw response for status code checking
   */
  async getRaw(path: string, params?: Record<string, string | number | boolean>): Promise<APIResponse> {
    return this.request.get(path, { params });
  }

  /**
   * Post raw response for status code checking
   */
  async postRaw(path: string, body?: any): Promise<APIResponse> {
    return this.request.post(path, { data: body });
  }
}

/**
 * Convenience function to create API helper
 */
export function createApiHelper(request: APIRequestContext): ApiHelper {
  return new ApiHelper(request);
}

/**
 * Wait for a condition to be true with polling
 */
export async function pollUntil<T>(
  fn: () => Promise<T>,
  predicate: (result: T) => boolean,
  options: { timeout?: number; interval?: number } = {}
): Promise<T> {
  const { timeout = 30000, interval = 500 } = options;
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const result = await fn();
    if (predicate(result)) {
      return result;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }

  throw new Error(`Polling timeout after ${timeout}ms`);
}
