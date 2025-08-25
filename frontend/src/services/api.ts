// SEP Trading System - API Service
// Simplified API client for core backend communications

const API_BASE_URL =
  process.env.REACT_APP_API_URL ||
  (window as any)?._env_?.REACT_APP_API_URL;
if (!API_BASE_URL) {
  throw new Error('REACT_APP_API_URL is not configured');
}

class APIClient {
  baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint: string, options: RequestInit = {}) {
    const url = `${this.baseURL}${endpoint}`;

    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers as Record<string, string> | undefined),
      },
      ...options,
    };

    const response = await fetch(url, config);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // OANDA Candle Data
  async fetchCandleData(instruments?: string[]) {
    return this.request('/api/candles/fetch', {
      method: 'POST',
      body: JSON.stringify({ instruments: instruments || [] }),
    });
  }

  async getStoredCandles(instrument: string) {
    return this.request(`/api/candles/${instrument}`);
  }

  // Configuration
  async getConfig() {
    return this.request('/api/config/get');
  }

  async updateConfig(config: any) {
    return this.request('/api/config/set', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }
}

// Create singleton instance
const apiClient = new APIClient();

// Export individual methods for convenience
export const {
  fetchCandleData,
  getStoredCandles,
  getConfig,
  updateConfig,
} = apiClient as any;

export { apiClient };
export default apiClient;
