import React from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { useSymbol } from '../context/SymbolContext';

const ConnectionStatusIndicator = () => {
  const { connectionStatus, connected, quantumSignals, valkeyMetrics } = useWebSocket();
  const { selectedSymbol } = useSymbol();

  const getStatusIcon = () => {
    switch(connectionStatus) {
      case 'connected': return '🟢';
      case 'connecting': return '🟡';
      case 'disconnected': return '🔴';
      case 'error': return '⚠️';
      default: return '⚪';
    }
  };

  const getStatusText = () => {
    switch(connectionStatus) {
      case 'connected': return 'Live';
      case 'connecting': return 'Connecting';
      case 'disconnected': return 'Offline';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  };

  const signalCount = Object.keys(quantumSignals).length;
  const valkeyConnected = valkeyMetrics && Object.keys(valkeyMetrics).length > 0;

  return (
    <div className="connection-status-indicator flex items-center gap-4 text-sm">
      {/* WebSocket Status */}
      <div className={`flex items-center gap-1 px-2 py-1 rounded ${connected ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'}`}>
        <span>{getStatusIcon()}</span>
        <span className="font-medium">{getStatusText()}</span>
      </div>

      {/* Current Symbol */}
      <div className="flex items-center gap-1 px-2 py-1 rounded bg-blue-900 text-blue-400">
        <span>📈</span>
        <span className="font-mono font-medium">{selectedSymbol}</span>
      </div>

      {/* Valkey Manifold Status */}
      {connected && (
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <div className={`flex items-center gap-1 px-2 py-1 rounded ${valkeyConnected ? 'bg-purple-900 text-purple-400' : 'bg-gray-800 text-gray-500'}`}>
            <span>⚡</span>
            <span>Valkey</span>
          </div>
          {signalCount > 0 && (
            <div className="flex items-center gap-1 px-2 py-1 rounded bg-cyan-900 text-cyan-400">
              <span>🧠</span>
              <span>{signalCount} signals</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ConnectionStatusIndicator;
