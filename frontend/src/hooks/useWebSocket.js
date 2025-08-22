import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const WebSocketContext = createContext();

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8765';

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [marketData, setMarketData] = useState({});
  const [systemStatus, setSystemStatus] = useState({});
  const [tradingSignals, setTradingSignals] = useState([]);
  const [performanceData, setPerformanceData] = useState({});

  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = () => {
    try {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
        reconnectAttemptsRef.current = 0;
        
        // Send connection message and subscribe to channels
        ws.send(JSON.stringify({
          type: 'subscribe',
          channels: ['market', 'system', 'signals', 'performance']
        }));
        
        // Start heartbeat
        const heartbeatInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'heartbeat' }));
          } else {
            clearInterval(heartbeatInterval);
          }
        }, 30000); // 30 seconds
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setConnected(false);
        setSocket(null);
        
        // Attempt to reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          console.log(`Reconnecting in ${delay}ms... (attempt ${reconnectAttemptsRef.current})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else {
          console.error('Max reconnection attempts reached');
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      setSocket(ws);
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