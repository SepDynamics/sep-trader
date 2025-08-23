// SEP Trading System - API Service
// Centralized API client for all backend communications

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

class APIClient {
  baseURL: string;
  token: string | null;

  constructor() {
    this.baseURL = API_BASE_URL;
    this.token = localStorage.getItem('auth_token');
  }

  async request(endpoint: string, options: RequestInit = {}) {
    const url = `${this.baseURL}${endpoint}`;

    const config: RequestInit & { headers: Record<string, string> } = {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers as Record<string, string> | undefined),
      },
      ...options,
    };

    if (this.token) {
      config.headers.Authorization = `Bearer ${this.token}`;
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
  async getMarketData() {
    return this.request('/api/market-data');
  }

  // Trading Operations
  async placeOrder(order: any) {
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

  // System Status
  async getSystemStatus() {
    return this.request('/api/system-status');
  }

  async getHealth() {
    return this.request('/api/health');
  }

  // Performance Metrics
  async getPerformanceMetrics() {
    return this.request('/api/performance/current');
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
  placeOrder,
  getPositions,
  getTradingSignals,
  getSystemStatus,
  getHealth,
  getPerformanceMetrics,
  getConfig,
  updateConfig,
  executeCLICommand,
  getCLIStatus,
  setAuthToken,
  clearAuthToken,
  isAuthenticated,
} = apiClient as any;

export { apiClient };
export default apiClient;

