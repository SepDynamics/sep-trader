import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import ConnectionStatusIndicator from './ConnectionStatusIndicator';
import '../styles/AppHeader.css';

const AppHeader = () => {
  const { connected, valkeyMetrics } = useWebSocket();
  const [pipelineStatus, setPipelineStatus] = useState('offline');
  const [oandaFeedRate, setOandaFeedRate] = useState(0);

  // Determine pipeline status based on WebSocket data
  useEffect(() => {
    const updatePipelineStatus = () => {
      // Get actual feed rate from valkeyMetrics if available
      const feedRate = valkeyMetrics?.feedRate || 0;
      
      // Update pipeline status based on connection and feed rate
      // This matches the logic used in ValkeyPipelineManager
      if (connected && feedRate > 1) {
        setPipelineStatus('active');
      } else if (connected) {
        setPipelineStatus('waiting');
      } else {
        setPipelineStatus('offline');
      }
      
      setOandaFeedRate(feedRate);
    };

    updatePipelineStatus();
    const interval = setInterval(updatePipelineStatus, 2000);
    
    return () => clearInterval(interval);
  }, [connected, valkeyMetrics]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'waiting': return 'text-yellow-400';
      case 'offline': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <header className="header">
      <div className="logo">
        <div className="logo-icon">⚡</div>
        <span>SEP Engine - Quantum Pattern Analysis System</span>
      </div>
      <div className="header-controls">
        <ConnectionStatusIndicator />
        <div className={`pipeline-status ${getStatusColor(pipelineStatus)}`}>
          <span className={`status-indicator ${pipelineStatus}`}></span>
          <span>OANDA Pipeline: {pipelineStatus.toUpperCase()} ({oandaFeedRate.toFixed(1)}/s)</span>
        </div>
      </div>
    </header>
  );
};

export default AppHeader;