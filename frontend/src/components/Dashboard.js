import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { apiClient } from '../services/api';
import '../styles/Dashboard.css';

const Dashboard = () => {
  const { connected, systemStatus, marketData, performanceData, tradingSignals } = useWebSocket();
  const [systemInfo, setSystemInfo] = useState({});
  const [performance, setPerformance] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load system status and performance data
      const [statusResponse, perfResponse] = await Promise.all([
        apiClient.getSystemStatus(),
        apiClient.getPerformanceCurrent()
      ]);

      setSystemInfo(statusResponse.data);
      setPerformance(perfResponse.data);
      setError(null);
    } catch (err) {
      console.error('Failed to load initial dashboard data:', err);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    if (value === null || value === undefined) return '--';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const formatPercentage = (value) => {
    if (value === null || value === undefined) return '--';
    const formatted = (value * 100).toFixed(2);
    const sign = value >= 0 ? '+' : '';
    return `${sign}${formatted}%`;
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'online':
      case 'active':
      case 'healthy':
        return 'success';
      case 'warning':
      case 'degraded':
        return 'warning';
      case 'offline':
      case 'error':
      case 'critical':
        return 'danger';
      default:
        return 'secondary';
    }
  };

  if (loading) {
    return (
      <div className="dashboard">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard">
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <h3>Dashboard Error</h3>
          <p>{error}</p>
          <button onClick={loadInitialData} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  const currentPerf = performanceData || performance;
  const currentStatus = systemStatus || systemInfo;
  const latestSignals = tradingSignals.slice(0, 5);

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>SEP Trading System Dashboard</h1>
        <div className="connection-status">
          <span className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '🟢' : '🔴'}
          </span>
          <span className="status-text">
            {connected ? 'Real-time Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* System Status Cards */}
        <div className="status-cards">
          <div className="status-card">
            <div className="card-header">
              <h3>System Status</h3>
              <span className={`status-badge ${getStatusColor(currentStatus.status)}`}>
                {currentStatus.status || 'Unknown'}
              </span>
            </div>
            <div className="card-content">
              <div className="status-item">
                <label>Engine:</label>
                <span className={getStatusColor(currentStatus.engine_status)}>
                  {currentStatus.engine_status || '--'}
                </span>
              </div>
              <div className="status-item">
                <label>Memory Tiers:</label>
                <span className={getStatusColor(currentStatus.memory_status)}>
                  {currentStatus.memory_status || '--'}
                </span>
              </div>
              <div className="status-item">
                <label>Trading:</label>
                <span className={getStatusColor(currentStatus.trading_status)}>
                  {currentStatus.trading_status || '--'}
                </span>
              </div>
              <div className="status-item">
                <label>Last Update:</label>
                <span>{currentStatus.last_update ? new Date(currentStatus.last_update).toLocaleTimeString() : '--'}</span>
              </div>
            </div>
          </div>

          <div className="status-card">
            <div className="card-header">
              <h3>Performance Summary</h3>
            </div>
            <div className="card-content">
              <div className="perf-metric">
                <label>Total P&L:</label>
                <span className={`value ${(currentPerf.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
                  {formatCurrency(currentPerf.total_pnl)}
                </span>
              </div>
              <div className="perf-metric">
                <label>Daily P&L:</label>
                <span className={`value ${(currentPerf.daily_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
                  {formatCurrency(currentPerf.daily_pnl)}
                </span>
              </div>
              <div className="perf-metric">
                <label>Win Rate:</label>
                <span className="value">
                  {formatPercentage(currentPerf.win_rate)}
                </span>
              </div>
              <div className="perf-metric">
                <label>Total Trades:</label>
                <span className="value">{currentPerf.total_trades || 0}</span>
              </div>
            </div>
          </div>

          <div className="status-card">
            <div className="card-header">
              <h3>Market Overview</h3>
            </div>
            <div className="card-content">
              {Object.keys(marketData).length > 0 ? (
                Object.entries(marketData).slice(0, 4).map(([symbol, data]) => (
                  <div key={symbol} className="market-item">
                    <label>{symbol}:</label>
                    <span className={`value ${(data.change || 0) >= 0 ? 'positive' : 'negative'}`}>
                      ${data.price?.toFixed(2) || '--'} ({formatPercentage(data.change / 100)})
                    </span>
                  </div>
                ))
              ) : (
                <div className="no-data">No market data available</div>
              )}
            </div>
          </div>
        </div>

        {/* Recent Signals */}
        <div className="recent-signals">
          <div className="card">
            <div className="card-header">
              <h3>Recent Trading Signals</h3>
              <button className="view-all-btn">View All</button>
            </div>
            <div className="card-content">
              {latestSignals.length > 0 ? (
                <div className="signals-list">
                  {latestSignals.map((signal, index) => (
                    <div key={index} className="signal-item">
                      <div className="signal-header">
                        <span className={`signal-type ${signal.type?.toLowerCase()}`}>
                          {signal.type}
                        </span>
                        <span className="signal-time">
                          {signal.timestamp ? new Date(signal.timestamp).toLocaleTimeString() : '--'}
                        </span>
                      </div>
                      <div className="signal-details">
                        <span className="signal-symbol">{signal.symbol || '--'}</span>
                        <span className="signal-price">${signal.price?.toFixed(2) || '--'}</span>
                        <span className={`signal-confidence confidence-${Math.floor((signal.confidence || 0) * 10)}`}>
                          {((signal.confidence || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-data">No recent signals</div>
              )}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="quick-actions">
          <div className="card">
            <div className="card-header">
              <h3>Quick Actions</h3>
            </div>
            <div className="card-content">
              <div className="action-buttons">
                <button className="action-btn primary">
                  <span className="btn-icon">▶️</span>
                  Start Trading
                </button>
                <button className="action-btn secondary">
                  <span className="btn-icon">⏸️</span>
                  Pause System
                </button>
                <button className="action-btn info">
                  <span className="btn-icon">📊</span>
                  View Reports
                </button>
                <button className="action-btn warning">
                  <span className="btn-icon">⚙️</span>
                  Settings
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;