// SEP Trading System - Complete API Client
// This connects to your actual backend services

const API_BASE_URL = window._env_?.REACT_APP_API_URL || 'http://localhost:5000';

class SEPApiClient {
  constructor() {
    this.baseURL = API_BASE_URL;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...this.headers,
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      
      return await response.text();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // System endpoints
  async getHealth() {
    return this.request('/api/health');
  }

  async getStatus() {
    return this.request('/api/status');
  }

  async getSystemInfo() {
    return this.request('/api/system/info');
  }

  // Trading endpoints
  async getTradingStatus() {
    return this.request('/api/trading/status');
  }

  async startTrading(config = {}) {
    return this.request('/api/trading/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async stopTrading() {
    return this.request('/api/trading/stop', {
      method: 'POST',
    });
  }

  async getPositions() {
    return this.request('/api/trading/positions');
  }

  async closePosition(positionId) {
    return this.request('/api/trading/position/close', {
      method: 'POST',
      body: JSON.stringify({ position_id: positionId }),
    });
  }

  // Pairs management
  async getPairs() {
    return this.request('/api/pairs');
  }

  async enablePair(pair) {
    return this.request(`/api/pairs/${pair}/enable`, {
      method: 'POST',
    });
  }

  async disablePair(pair) {
    return this.request(`/api/pairs/${pair}/disable`, {
      method: 'POST',
    });
  }

  // Performance endpoints
  async getPerformanceMetrics() {
    return this.request('/api/performance/metrics');
  }

  async getPerformanceCurrent() {
    return this.request('/api/performance/current');
  }

  async getPerformanceHistory(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    return this.request(`/api/performance/history${queryString ? '?' + queryString : ''}`);
  }

  // Metrics endpoints
  async getLiveMetrics() {
    return this.request('/api/metrics/live');
  }

  // Configuration endpoints
  async getConfig(key = null) {
    const params = key ? `?key=${key}` : '';
    return this.request(`/api/config/get${params}`);
  }

  async setConfig(key, value) {
    return this.request('/api/config/set', {
      method: 'POST',
      body: JSON.stringify({ key, value }),
    });
  }

  async getConfigSchema() {
    return this.request('/api/config/schema');
  }

  // Command execution
  async executeCommand(command) {
    return this.request('/api/commands/execute', {
      method: 'POST',
      body: JSON.stringify({ command }),
    });
  }

  // Data reload
  async reloadData() {
    return this.request('/api/data/reload', {
      method: 'POST',
    });
  }
}

// Create singleton instance
const apiClient = new SEPApiClient();

export default apiClient;
export { apiClient, SEPApiClient };