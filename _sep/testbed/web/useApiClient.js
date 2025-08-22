import { useCallback } from 'react';

const API_URL = process.env.REACT_APP_API_URL || '';

export default function useApiClient() {
  const fetchJson = useCallback(async (path, options = {}) => {
    const response = await fetch(`${API_URL}${path}`, options);
    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }
    return response.json();
  }, []);

  const getStatus = useCallback(() => fetchJson('/api/status'), [fetchJson]);
  const getPairs = useCallback(() => fetchJson('/api/pairs'), [fetchJson]);
  const getPerformanceHistory = useCallback(() => fetchJson('/api/performance/history'), [fetchJson]);
  const executeCommand = useCallback(
    (command) =>
      fetchJson('/api/commands/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command }),
      }),
    [fetchJson]
  );

  return {
    getStatus,
    getPairs,
    getPerformanceHistory,
    executeCommand,
  };
}
