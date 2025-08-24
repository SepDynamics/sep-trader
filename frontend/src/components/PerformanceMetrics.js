import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { apiClient } from '../services/api';

const PerformanceMetrics = () => {
  const { connected, performanceData } = useWebSocket();
  const [metrics, setMetrics] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadPerformanceData();
  }, []);

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getPerformanceCurrent();
      setMetrics(response);
      setError(null);
    } catch (err) {
      console.error('Failed to load performance data:', err);
      setError('Failed to load performance data');
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

  const currentMetrics = performanceData || metrics;

  if (loading && !currentMetrics.total_pnl) {
    return <div className="loading-container">Loading performance metrics...</div>;
  }

  if (error && !currentMetrics.total_pnl) {
    return <div className="error-container">{error}</div>;
  }

  return (
    <div className="performance-metrics">
      <div className="metrics-header">
        <h1>Performance Metrics</h1>
        <div className="metrics-controls">
          <div className="connection-status">
            <span className={connected ? 'connected' : 'disconnected'}>
              {connected ? '🟢 Real-time' : '🔴 Offline'}
            </span>
          </div>
          <button onClick={loadPerformanceData} className="refresh-btn">Refresh</button>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Total P&L</h3>
          <div className={`metric-value ${(currentMetrics.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
            {formatCurrency(currentMetrics.total_pnl)}
          </div>
        </div>

        <div className="metric-card">
          <h3>Daily P&L</h3>
          <div className={`metric-value ${(currentMetrics.daily_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
            {formatCurrency(currentMetrics.daily_pnl)}
          </div>
        </div>

        <div className="metric-card">
          <h3>Win Rate</h3>
          <div className="metric-value">{formatPercentage(currentMetrics.win_rate)}</div>
        </div>

        <div className="metric-card">
          <h3>Total Trades</h3>
          <div className="metric-value">{currentMetrics.total_trades || 0}</div>
        </div>

        <div className="metric-card">
          <h3>Sharpe Ratio</h3>
          <div className="metric-value">{(currentMetrics.sharpe_ratio || 0).toFixed(2)}</div>
        </div>

        <div className="metric-card">
          <h3>Max Drawdown</h3>
          <div className="metric-value negative">{formatPercentage(currentMetrics.max_drawdown)}</div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMetrics;