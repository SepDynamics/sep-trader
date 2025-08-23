// SEP Trading System - API Service
// Centralized API client for all backend communications

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

class APIClient {
  constructor() {
    this.baseURL = API_BASE_URL;
    this.token = localStorage.getItem('auth_token');
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    if (this.token) {
      config.headers.Authorization = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Authentication
  async login(credentials) {
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
  async placeOrder(order) {
    return this.request('/api/place-order', {
      method: 'POST',
      body: JSON.stringify(order),
    });
  }

  async submitOrder(order) {
    return this.request('/api/orders', {
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
  async getConfiguration() {
    return this.request('/api/config/get');
  }

  async setConfiguration(config) {
    return this.request('/api/config/set', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // CLI Operations
  async executeCLICommand(command, args = []) {
    return this.request('/api/cli/execute', {
      method: 'POST',
      body: JSON.stringify({ command, args }),
    });
  }

  async getCLIStatus() {
    return this.request('/api/cli/status');
  }

  // Utility Methods
  setAuthToken(token) {
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
  submitOrder,
  getPositions,
  getTradingSignals,
  getSystemStatus,
  getHealth,
  getPerformanceMetrics,
  getConfiguration,
  setConfiguration,
  executeCLICommand,
  getCLIStatus,
  setAuthToken,
  clearAuthToken,
  isAuthenticated,
} = apiClient;

export { apiClient };
export default apiClient;
