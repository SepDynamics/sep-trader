import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import '../styles/SystemMonitor.css';

const SystemMonitor = () => {
  const { connectionStatus, systemMetrics, valkeyMetrics } = useWebSocket();
  const [systemStats, setSystemStats] = useState({
    cpu: 0,
    memory: 0,
    gpu: 0,
    network: 0
  });
  const [pipelineStatus, setPipelineStatus] = useState('OFFLINE');
  const [quantumEngine, setQuantumEngine] = useState('ACTIVE');

  useEffect(() => {
    if (systemMetrics) {
      setSystemStats({
        cpu: systemMetrics.cpu_usage || 0,
        memory: systemMetrics.memory_usage || 0,
        gpu: systemMetrics.gpu_usage || 0,
        network: systemMetrics.network_io || 0
      });
    }
    
    if (valkeyMetrics && valkeyMetrics.pipeline_status) {
      setPipelineStatus(valkeyMetrics.pipeline_status);
    }
  }, [systemMetrics, valkeyMetrics]);

  const getStatusColor = (value) => {
    if (value < 50) return 'status-green';
    if (value < 80) return 'status-yellow';
    return 'status-red';
  };

  const renderProgressBar = (value, label) => {
    return (
      <div className="progress-container">
        <div className="progress-label">
          <span>{label}</span>
          <span>{value}%</span>
        </div>
        <div className="progress-bar">
          <div 
            className={`progress-fill ${getStatusColor(value)}`}
            style={{ width: `${value}%` }}
          ></div>
        </div>
      </div>
    );
  };

  return (
    <div className="system-monitor">
      <div className="monitor-header">
        <h1>System Health & Performance</h1>
        <p>Real-time monitoring of SEP Engine components</p>
      </div>

      <div className="status-grid">
        <div className="status-card">
          <h3>WebSocket Connection</h3>
          <div className={`status-indicator ${connectionStatus === 'connected' ? 'connected' : 'disconnected'}`}>
            {connectionStatus === 'connected' ? 'CONNECTED' : 'DISCONNECTED'}
          </div>
        </div>
        
        <div className="status-card">
          <h3>OANDA Pipeline</h3>
          <div className={`status-indicator ${pipelineStatus === 'ONLINE' ? 'connected' : 'disconnected'}`}>
            {pipelineStatus}
          </div>
        </div>
        
        <div className="status-card">
          <h3>Quantum Engine</h3>
          <div className={`status-indicator ${quantumEngine === 'ACTIVE' ? 'connected' : 'disconnected'}`}>
            {quantumEngine}
          </div>
        </div>
      </div>

      <div className="performance-section">
        <h2>System Performance</h2>
        <div className="performance-grid">
          {renderProgressBar(systemStats.cpu, 'CPU Usage')}
          {renderProgressBar(systemStats.memory, 'Memory Usage')}
          {renderProgressBar(systemStats.gpu, 'GPU Usage')}
          {renderProgressBar(systemStats.network, 'Network I/O')}
        </div>
      </div>

      <div className="components-section">
        <h2>Engine Components</h2>
        <div className="components-grid">
          <div className="component-card">
            <h3>Manifold Processor</h3>
            <div className="component-status active">ACTIVE</div>
            <p>Processing timestamped market identities</p>
          </div>
          
          <div className="component-card">
            <h3>Entropy Calculator</h3>
            <div className="component-status active">ACTIVE</div>
            <p>Computing market state disorder</p>
          </div>
          
          <div className="component-card">
            <h3>Pattern Matcher</h3>
            <div className="component-status active">ACTIVE</div>
            <p>Identifying quantum patterns</p>
          </div>
          
          <div className="component-card">
            <h3>Backwards Integrator</h3>
            <div className="component-status active">ACTIVE</div>
            <p>Enabling temporal computation</p>
          </div>
        </div>
      </div>

      <div className="data-flow-section">
        <h2>Data Flow Status</h2>
        <div className="data-flow-diagram">
          <div className="flow-step">
            <div className="step-icon">📡</div>
            <div className="step-label">OANDA API</div>
            <div className={`step-status ${pipelineStatus === 'ONLINE' ? 'active' : 'inactive'}`}></div>
          </div>
          
          <div className="flow-arrow">→</div>
          
          <div className="flow-step">
            <div className="step-icon">🌊</div>
            <div className="step-label">Valkey Pipeline</div>
            <div className={`step-status ${pipelineStatus === 'ONLINE' ? 'active' : 'inactive'}`}></div>
          </div>
          
          <div className="flow-arrow">→</div>
          
          <div className="flow-step">
            <div className="step-icon">🧠</div>
            <div className="step-label">Quantum Engine</div>
            <div className="step-status active"></div>
          </div>
          
          <div className="flow-arrow">→</div>
          
          <div className="flow-step">
            <div className="step-icon">🖥️</div>
            <div className="step-label">Frontend</div>
            <div className={`step-status ${connectionStatus === 'connected' ? 'active' : 'inactive'}`}></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemMonitor;