import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { apiClient } from '../services/api';
import '../styles/SystemStatus.css';

const SystemStatus = () => {
  const { connected, systemStatus } = useWebSocket();
  const [systemInfo, setSystemInfo] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(new Date());

  useEffect(() => {
    loadSystemStatus();
    
    // Refresh system status every 30 seconds
    const interval = setInterval(() => {
      loadSystemStatus();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const loadSystemStatus = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getSystemStatus();
      setSystemInfo(response.data);
      setLastRefresh(new Date());
      setError(null);
    } catch (err) {
      console.error('Failed to load system status:', err);
      setError('Failed to load system status');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    if (!status) return 'unknown';
    
    switch (status.toLowerCase()) {
      case 'online':
      case 'active':
      case 'healthy':
      case 'running':
        return 'success';
      case 'warning':
      case 'degraded':
      case 'slow':
        return 'warning';
      case 'offline':
      case 'error':
      case 'critical':
      case 'failed':
        return 'danger';
      default:
        return 'secondary';
    }
  };

  const getStatusIcon = (status) => {
    const color = getStatusColor(status);
    switch (color) {
      case 'success': return '✅';
      case 'warning': return '⚠️';
      case 'danger': return '❌';
      default: return '❓';
    }
  };

  const formatUptime = (seconds) => {
    if (!seconds) return 'Unknown';
    
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  };

  const currentStatus = systemStatus || systemInfo;

  if (loading && !currentStatus.status) {
    return (
      <div className="system-status">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading system status...</p>
        </div>
      </div>
    );
  }

  if (error && !currentStatus.status) {
    return (
      <div className="system-status">
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <h3>System Status Error</h3>
          <p>{error}</p>
          <button onClick={loadSystemStatus} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="system-status">
      <div className="status-header">
        <h1>System Status</h1>
        <div className="status-controls">
          <div className="last-refresh">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </div>
          <button 
            onClick={loadSystemStatus} 
            className="refresh-btn"
            disabled={loading}
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      <div className="status-grid">
        {/* Overall Status */}
        <div className="status-card overall-status">
          <div className="card-header">
            <h3>Overall System Status</h3>
            <div className={`status-badge ${getStatusColor(currentStatus.status)}`}>
              {getStatusIcon(currentStatus.status)} {currentStatus.status || 'Unknown'}
            </div>
          </div>
          <div className="card-content">
            <div className="status-summary">
              <div className="connection-info">
                <span className={`connection-dot ${connected ? 'connected' : 'disconnected'}`}></span>
                <span>Real-time: {connected ? 'Connected' : 'Disconnected'}</span>
              </div>
              <div className="uptime-info">
                <label>System Uptime:</label>
                <span>{formatUptime(currentStatus.uptime)}</span>
              </div>
              <div className="version-info">
                <label>Version:</label>
                <span>{currentStatus.version || 'Unknown'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Component Status */}
        <div className="status-card">
          <div className="card-header">
            <h3>Core Components</h3>
          </div>
          <div className="card-content">
            <div className="component-list">
              <div className="component-item">
                <div className="component-info">
                  <span className="component-name">SEP Engine</span>
                  <span className={`component-status ${getStatusColor(currentStatus.engine_status)}`}>
                    {getStatusIcon(currentStatus.engine_status)} {currentStatus.engine_status || 'Unknown'}
                  </span>
                </div>
              </div>
              
              <div className="component-item">
                <div className="component-info">
                  <span className="component-name">Memory Tiers</span>
                  <span className={`component-status ${getStatusColor(currentStatus.memory_status)}`}>
                    {getStatusIcon(currentStatus.memory_status)} {currentStatus.memory_status || 'Unknown'}
                  </span>
                </div>
              </div>
              
              <div className="component-item">
                <div className="component-info">
                  <span className="component-name">Trading System</span>
                  <span className={`component-status ${getStatusColor(currentStatus.trading_status)}`}>
                    {getStatusIcon(currentStatus.trading_status)} {currentStatus.trading_status || 'Unknown'}
                  </span>
                </div>
              </div>
              
              <div className="component-item">
                <div className="component-info">
                  <span className="component-name">WebSocket Service</span>
                  <span className={`component-status ${getStatusColor(connected ? 'online' : 'offline')}`}>
                    {getStatusIcon(connected ? 'online' : 'offline')} {connected ? 'Online' : 'Offline'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Resource Usage */}
        <div className="status-card">
          <div className="card-header">
            <h3>Resource Usage</h3>
          </div>
          <div className="card-content">
            <div className="resource-metrics">
              <div className="metric-item">
                <div className="metric-header">
                  <label>CPU Usage</label>
                  <span className="metric-value">{(currentStatus.cpu_usage * 100 || 0).toFixed(1)}%</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${(currentStatus.cpu_usage * 100 || 0)}%` }}
                  ></div>
                </div>
              </div>

              <div className="metric-item">
                <div className="metric-header">
                  <label>Memory Usage</label>
                  <span className="metric-value">
                    {formatBytes(currentStatus.memory_used)} / {formatBytes(currentStatus.memory_total)}
                  </span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ 
                      width: `${((currentStatus.memory_used / currentStatus.memory_total) * 100 || 0)}%` 
                    }}
                  ></div>
                </div>
              </div>

              <div className="metric-item">
                <div className="metric-header">
                  <label>Active Connections</label>
                  <span className="metric-value">{currentStatus.active_connections || 0}</span>
                </div>
              </div>

              <div className="metric-item">
                <div className="metric-header">
                  <label>Processed Patterns</label>
                  <span className="metric-value">{currentStatus.patterns_processed || 0}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Events */}
        <div className="status-card events-card">
          <div className="card-header">
            <h3>Recent System Events</h3>
          </div>
          <div className="card-content">
            {currentStatus.recent_events && currentStatus.recent_events.length > 0 ? (
              <div className="events-list">
                {currentStatus.recent_events.map((event, index) => (
                  <div key={index} className={`event-item ${event.type}`}>
                    <div className="event-time">
                      {new Date(event.timestamp).toLocaleString()}
                    </div>
                    <div className="event-message">
                      {getStatusIcon(event.type)} {event.message}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-events">No recent events</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemStatus;