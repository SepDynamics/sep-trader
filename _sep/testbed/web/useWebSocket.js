import { useEffect, useRef } from 'react';

export default function useWebSocket(onMetric) {
  const socketRef = useRef(null);

  useEffect(() => {
    const base = process.env.REACT_APP_API_URL || '';
    const wsUrl = `${base}`.replace(/^http/, 'ws') + '/ws';
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMetric && onMetric(data);
      } catch (_) {
        // ignore parse errors
      }
    };

    return () => {
      socket.close();
    };
  }, [onMetric]);

  return socketRef;
}
