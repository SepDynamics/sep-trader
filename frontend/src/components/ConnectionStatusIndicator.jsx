import React from 'react';
import { useWebSocket } from '../context/WebSocketContext';

const ConnectionStatusIndicator = () => {
  const { connectionStatus } = useWebSocket();
  return (
    <span className={`ws-status ws-${connectionStatus} text-sm ml-2`}>
      {connectionStatus}
    </span>
  );
};

export default ConnectionStatusIndicator;
