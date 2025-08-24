import React from 'react';
import { useWebSocket } from '../context/WebSocketContext';

const ManifoldVisualizer = () => {
  const { connected, quantumSignals, livePatterns } = useWebSocket();

  if (!connected) {
    return <div className="text-center text-sm text-gray-500 py-4">Disconnected</div>;
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-4">Manifold Visualizer</h1>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-2">Quantum Signals</h3>
          {Object.entries(quantumSignals || {}).map(([key, sig]) => (
            <div key={key} className="text-sm text-gray-300">
              {key}: {(((sig || {}).coherence || 0) * 100).toFixed(1)}%
            </div>
          ))}
          {Object.keys(quantumSignals || {}).length === 0 && (
            <div className="text-sm text-gray-500">No signals available</div>
          )}
        </div>
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-2">Live Patterns</h3>
          {Object.entries(livePatterns || {}).map(([key]) => (
            <div key={key} className="text-sm text-gray-300">
              {key}
            </div>
          ))}
          {Object.keys(livePatterns || {}).length === 0 && (
            <div className="text-sm text-gray-500">No patterns available</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ManifoldVisualizer;
