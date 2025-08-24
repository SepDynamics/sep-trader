import React from 'react';
import { useWebSocket } from '../context/WebSocketContext';

const ManifoldKernel = () => {
  const { connected, quantumSignals } = useWebSocket();

  if (!connected) {
    return <div className="text-center text-sm text-gray-500 py-4">Disconnected</div>;
  }

  const entries = Object.entries(quantumSignals || {});

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-4">Manifold Kernel</h1>
      {entries.length === 0 ? (
        <div className="text-gray-500">No signals available</div>
      ) : (
        <div className="space-y-2">
          {entries.map(([key, signal]) => (
            <div key={key} className="bg-gray-900 rounded p-2 text-sm">
              <div className="text-gray-400">{key}</div>
              <div>Coherence: {(((signal || {}).coherence || 0) * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ManifoldKernel;
