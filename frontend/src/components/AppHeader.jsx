import React from 'react';
import { useWebSocket } from '../context/WebSocketContext';

const StatusIndicator = ({ status, label, data }) => {
  const statusConfig = {
    active: { color: 'bg-green-500', text: 'Active' },
    waiting: { color: 'bg-yellow-500', text: 'Waiting' },
    offline: { color: 'bg-red-500', text: 'Offline' },
  };

  const { color, text } = statusConfig[status] || { color: 'bg-gray-500', text: 'Unknown' };

  return (
    <div className="flex items-center space-x-2">
      <span className={`h-3 w-3 rounded-full ${color} animate-pulse`}></span>
      <span className="text-sm font-medium text-gray-300">
        {label}: <span className="font-semibold text-white">{text}</span> {data && `(${data})`}
      </span>
    </div>
  );
};

const AppHeader = () => {
  const { connected, valkeyMetrics } = useWebSocket();

  const pipelineStatus = connected && valkeyMetrics?.feedRate > 1 ? 'active' : connected ? 'waiting' : 'offline';
  const feedRate = valkeyMetrics?.feedRate?.toFixed(1) || '0.0';

  return (
    <header className="bg-gray-900/80 backdrop-blur-lg border-b border-gray-700/50 shadow-lg sticky top-0 z-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <div className="flex-shrink-0">
              <svg className="h-8 w-8 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h1 className="text-xl font-bold text-gray-100 tracking-wider">SEP Engine</h1>
          </div>
          <div className="hidden md:flex items-center space-x-6">
            <StatusIndicator status={connected ? 'active' : 'offline'} label="WebSocket" />
            <StatusIndicator status={pipelineStatus} label="OANDA Pipeline" data={`${feedRate}/s`} />
          </div>
        </div>
      </div>
    </header>
  );
};

export default AppHeader;
