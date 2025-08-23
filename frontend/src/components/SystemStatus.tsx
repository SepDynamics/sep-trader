// @ts-nocheck
import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { apiClient } from '../services/api';
import usePolling from '../hooks/usePolling';
import '../styles/SystemStatus.css';

interface ComponentConfig {
  name: string;
  key: string;
}

const SystemStatus: React.FC = () => {
  const { connected, systemStatus, connectionStatus } = useWebSocket();
  const [config, setConfig] = useState<{ poll_interval: number; components: ComponentConfig[] }>({ poll_interval: 30000, components: [] });
  const [lastRefresh, setLastRefresh] = useState(new Date());

  useEffect(() => {
    apiClient.getSystemStatusConfig()
      .then(setConfig)
      .catch(() => {});
  }, []);

  const fetchStatus = async () => {
    const data = await apiClient.getSystemStatus();
    setLastRefresh(new Date());
    return data;
  };

  const { data: systemInfo, loading, error, refresh } = usePolling(fetchStatus, config.poll_interval);

  const getStatusColor = (status?: string) => {
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

  const getStatusIcon = (status?: string) => {
    const color = getStatusColor(status);
    switch (color) {
      case 'success': return '✅';
      case 'warning': return '⚠️';
      case 'danger': return '❌';
      default: return '❓';
    }
  };

  const formatUptime = (seconds?: number) => {
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

  const formatBytes = (bytes?: number) => {
    if (!bytes) return '0 B';
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  };

  const currentStatus = systemStatus || systemInfo || {};

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
          <p>{error.message}</p>
          <button onClick={refresh} className="retry-button">
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
            onClick={refresh}
            className="refresh-btn"
            disabled={loading}
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      <div className="status-grid">
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
                <span className="connection-status-label">State: {connectionStatus}</span>
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

        <div className="status-card">
          <div className="card-header">
            <h3>Core Components</h3>
          </div>
          <div className="card-content">
            <div className="component-list">
              {config.components.map((comp) => {
                const status = comp.key === 'websocket'
                  ? (connected ? 'online' : 'offline')
                  : currentStatus[comp.key];
                return (
                  <div key={comp.key} className="component-item">
                    <div className="component-info">
                      <span className="component-name">{comp.name}</span>
                      <span className={`component-status ${getStatusColor(status)}`}>
                        {getStatusIcon(status)} {status || 'Unknown'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

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
                    style={{ width: `${((currentStatus.memory_used / currentStatus.memory_total) * 100 || 0)}%` }}
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
