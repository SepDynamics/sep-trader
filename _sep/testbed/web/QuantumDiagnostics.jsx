import React, { useState } from 'react';
import useWebSocket from './useWebSocket';

export default function QuantumDiagnostics() {
  const [metrics, setMetrics] = useState([]);

  useWebSocket((metric) => {
    setMetrics((prev) => [...prev, metric]);
  });

  const latest = metrics[metrics.length - 1];

  return (
    <div>
      <h3>Quantum Diagnostics</h3>
      {latest ? <pre>{JSON.stringify(latest, null, 2)}</pre> : <p>No metrics received.</p>}
    </div>
  );
}
