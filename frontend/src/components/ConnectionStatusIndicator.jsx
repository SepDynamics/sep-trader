import React from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { useSymbol } from '../context/SymbolContext';

const ConnectionStatusIndicator = () => {
  const { connectionStatus, connected } = useWebSocket();
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
    </div>
  );
};

export default ConnectionStatusIndicator;
