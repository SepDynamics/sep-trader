// SEP Trading System - API Service
// Centralized API client for all backend communications

const API_BASE_URL = process.env.REACT_APP_API_URL;
if (!API_BASE_URL) {
  throw new Error('REACT_APP_API_URL is not configured');
}

class APIClient {
  baseURL: string;
  token: string | null;

  constructor() {
    this.baseURL = API_BASE_URL;
    this.token = localStorage.getItem('auth_token');
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

    if (this.token && config.headers && typeof config.headers === 'object' && !Array.isArray(config.headers)) {
      (config.headers as Record<string, string>).Authorization = `Bearer ${this.token}`;
    }

    const response = await fetch(url, config);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Authentication
  async login(credentials: any) {
    return this.request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  }

  async logout() {
    return this.request('/api/auth/logout', { method: 'POST' });
  }

  // Market Data
  async getMarketData(instrument: string, from: number, to: number) {
    const formattedInstrument = instrument.replace('/', '_');
    const params = new URLSearchParams({
      instrument: formattedInstrument,
      from: from.toString(),
      to: to.toString(),
    });
    // This assumes your backend serves from /api/market-data
    // Your nginx.conf already proxies /api/ to the backend.
    return this.request(`/api/market-data?${params.toString()}`);
  }

  async getLiveMetrics() {
    return this.request('/api/metrics/live');
  }

  // Signals
  async getSignals() {
    return this.request('/api/signals');
  }

  async getSignalHistory(params = {}) {
    const queryString = new URLSearchParams(params as Record<string, string>).toString();
    return this.request(`/api/signals/history${queryString ? '?' + queryString : ''}`);
  }

  // Valkey/Redis Integration
  async getValkeyMetrics() {
    return this.request('/api/valkey/metrics');
  }

  async getRedisMetrics() {
    return this.request('/api/metrics/redis');
  }

  async getValkeyStatus() {
    return this.request('/api/valkey/status');
  }

  async getLivePatterns() {
    return this.request('/api/patterns/live');
  }

  async getQuantumSignals() {
    return this.request('/api/quantum/signals');
  }

  // OANDA Candle Data
  async getCandleData(instrument: string, granularity?: string, count?: number) {
    const params = new URLSearchParams();
    if (granularity) params.append('granularity', granularity);
    if (count) params.append('count', count.toString());
    
    const queryString = params.toString() ? `?${params.toString()}` : '';
    return this.request(`/api/candles/${instrument}${queryString}`);
  }

  async fetchCandleData(instruments?: string[]) {
    return this.request('/api/candles/fetch', {
      method: 'POST',
      body: JSON.stringify({ instruments: instruments || [] }),
    });
  }

  async getStoredCandles(instrument: string) {
    return this.request(`/api/candles/${instrument}`);
  }

  // Trading Operations
  async submitOrder(order: any) {
    return this.request('/api/place-order', {
      method: 'POST',
      body: JSON.stringify(order),
    });
  }

  async getPositions() {
    return this.request('/api/positions');
  }

  async getTradingSignals() {
    return this.request('/api/trading-signals');
  }

  // Quick Actions
  async startTrading() {
    return this.request('/api/trading/start', { method: 'POST' });
  }

  async stopTrading() {
    return this.request('/api/trading/stop', { method: 'POST' });
  }

  async pauseSystem() {
    return this.request('/api/system/pause', { method: 'POST' });
  }

  async uploadTrainingData(payload = {}) {
    return this.request('/api/training/upload', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async startModelTraining() {
    return this.request('/api/training/start', { method: 'POST' });
  }

  async generateReport() {
    return this.request('/api/reports/generate', { method: 'POST' });
  }

  // System Status
  async getSystemStatus() {
    return this.request('/api/system-status');
  }

  async getSystemStatusConfig() {
    return this.request('/api/system-status/config');
  }

  async getHealth() {
    return this.request('/api/health');
  }

  // Performance Metrics
  async getPerformanceCurrent() {
    return this.request('/api/performance/current');
  }

  async getPerformanceHistory(params = {}) {
    const queryString = new URLSearchParams(params as Record<string, string>).toString();
    return this.request(`/api/performance/history${queryString ? '?' + queryString : ''}`);
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

  // CLI Operations
  async executeCLICommand(command: string, args: string[] = []) {
    return this.request('/api/cli/execute', {
      method: 'POST',
      body: JSON.stringify({ command, args }),
    });
  }

  async getCLIStatus() {
    return this.request('/api/cli/status');
  }

  // Utility Methods
  setAuthToken(token: string) {
    this.token = token;
    localStorage.setItem('auth_token', token);
  }

  clearAuthToken() {
    this.token = null;
    localStorage.removeItem('auth_token');
  }

  isAuthenticated() {
    return !!this.token;
  }
}

// Create singleton instance
const apiClient = new APIClient();

// Export individual methods for convenience
export const {
  login,
  logout,
  getMarketData,
  getLiveMetrics,
  getSignals,
  getSignalHistory,
  getValkeyMetrics,
  getValkeyStatus,
  getLivePatterns,
  getQuantumSignals,
  getCandleData,
  fetchCandleData,
  getStoredCandles,
  submitOrder,
  getPositions,
  getTradingSignals,
  getSystemStatus,
  getSystemStatusConfig,
  getHealth,
  startTrading,
  stopTrading,
  pauseSystem,
  uploadTrainingData,
  startModelTraining,
  generateReport,
  getPerformanceCurrent,
  getPerformanceHistory,
  getConfig,
  updateConfig,
  executeCLICommand,
  getCLIStatus,
  setAuthToken,
  clearAuthToken,
  isAuthenticated,
} = apiClient as any;

// Add alias for getConfiguration to maintain compatibility
export const getConfiguration = apiClient.getConfig.bind(apiClient);

export { apiClient };
export default apiClient;

