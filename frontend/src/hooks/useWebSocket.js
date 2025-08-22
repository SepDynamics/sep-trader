import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const WebSocketContext = createContext();

const WS_URL = window._env_?.REACT_APP_WS_URL || 'ws://localhost:8765';

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [marketData, setMarketData] = useState({});
  const [systemStatus, setSystemStatus] = useState({});
  const [tradingSignals, setTradingSignals] = useState([]);
  const [performanceData, setPerformanceData] = useState({});

  const connect = () => {
    try {
      const newSocket = io(WS_URL);

      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setConnected(true);
        newSocket.emit('subscribe', { channels: ['market', 'system', 'signals', 'performance'] });
      });

      newSocket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        setConnected(false);
      });

      newSocket.on('message', (message) => {
        handleWebSocketMessage(message);
      });

      setSocket(newSocket);
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
    }
  };

  const handleWebSocketMessage = (message) => {
    const { type, channel, data } = message;

    switch (channel) {
      case 'market':
        if (type === 'market_update') {
          setMarketData(prev => ({
            ...prev,
            [data.symbol]: data
          }));
        }
        break;

      case 'system':
        if (type === 'system_status') {
          setSystemStatus(data);
        }
        break;

      case 'signals':
        if (type === 'trading_signal') {
          setTradingSignals(prev => [data, ...prev.slice(0, 49)]); // Keep last 50
        }
        break;

      case 'performance':
        if (type === 'performance_update') {
          setPerformanceData(data);
        }
        break;

      default:
        console.log('Received message:', message);
    }
  };

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (socket) {
      socket.close();
    }
  };

  const subscribeToChannels = (channels) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: 'subscribe',
        channels: channels
      }));
    }
  };

  const unsubscribeFromChannels = (channels) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: 'unsubscribe',
        channels: channels
      }));
    }
  };

  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, []);

  const value = {
    socket,
    connected,
    marketData,
    systemStatus,
    tradingSignals,
    performanceData,
    connect,
    disconnect,
    subscribeToChannels,
    unsubscribeFromChannels,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export default useWebSocket;