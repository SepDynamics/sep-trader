// SEP Trading System - WebSocket Context
// Real-time data connection to your WebSocket service

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';

const WebSocketContext = createContext(null);

const WS_URL = window._env_?.REACT_APP_WS_URL || 'ws://localhost:8765';

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  
  // Data states
  const [marketData, setMarketData] = useState({});
  const [systemStatus, setSystemStatus] = useState({});
  const [tradingSignals, setTradingSignals] = useState([]);
  const [performanceData, setPerformanceData] = useState({});
  const [systemMetrics, setSystemMetrics] = useState({});
  
  // Connection management
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const heartbeatIntervalRef = useRef(null);
  const maxReconnectAttempts = 10;
  const reconnectDelay = 3000;

  // Message handlers for different channels
  const messageHandlers = {
    market: (data) => {
      if (data.symbol) {
        setMarketData(prev => ({
          ...prev,
          [data.symbol]: {
            ...prev[data.symbol],
            ...data,
            lastUpdate: new Date().toISOString()
          }
        }));
      }
    },
    
    system: (data) => {
      setSystemStatus(prev => ({
        ...prev,
        ...data,
        lastUpdate: new Date().toISOString()
      }));
      
      // Update metrics if included
      if (data.metrics) {
        setSystemMetrics(data.metrics);
      }
    },
    
    signals: (data) => {
      setTradingSignals(prev => {
        const newSignals = [data, ...prev];
        // Keep only last 100 signals
        return newSignals.slice(0, 100);
      });
    },
    
    performance: (data) => {
      setPerformanceData(prev => ({
        ...prev,
        ...data,
        lastUpdate: new Date().toISOString()
      }));
    },
    
    trades: (data) => {
      // Handle trade execution updates
      console.log('Trade update:', data);
    }
  };

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (socket?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    console.log(`Connecting to WebSocket at ${WS_URL}...`);
    setConnectionStatus('connecting');
    
    try {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        console.log('WebSocket connected successfully');
        setConnected(true);
        setConnectionStatus('connected');
        setSocket(ws);
        reconnectAttemptsRef.current = 0;
        
        // Subscribe to all channels
        ws.send(JSON.stringify({
          type: 'subscribe',
          channels: ['market', 'system', 'signals', 'performance', 'trades']
        }));
        
        // Start heartbeat
        startHeartbeat(ws);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log(`WebSocket disconnected: ${event.code} - ${event.reason}`);
        setConnected(false);
        setConnectionStatus('disconnected');
        setSocket(null);
        stopHeartbeat();
        
        // Attempt reconnection
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          const attempt = reconnectAttemptsRef.current + 1;
          reconnectAttemptsRef.current = attempt;
          const delay = Math.min(reconnectDelay * attempt, 30000);
          
          console.log(`Reconnecting in ${delay}ms (attempt ${attempt}/${maxReconnectAttempts})...`);
          setConnectionStatus('reconnecting');
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else {
          console.error('Max reconnection attempts reached');
          setConnectionStatus('failed');
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionStatus('error');
      setConnected(false);
    }
  }, []);

  // Handle incoming messages
  const handleMessage = (message) => {
    const { type, channel, data } = message;
    
    // Handle different message types
    switch (type) {
      case 'connection':
        console.log('Connection acknowledged:', data);
        break;
        
      case 'subscription':
        console.log('Subscription confirmed:', data);
        break;
        
      case 'heartbeat':
        // Heartbeat response received
        break;
        
      default:
        // Route to appropriate handler based on channel
        if (channel && messageHandlers[channel]) {
          messageHandlers[channel](data);
        } else if (data) {
          // Try to infer channel from message structure
          if (data.symbol && data.price) {
            messageHandlers.market(data);
          } else if (data.signal_type && data.confidence) {
            messageHandlers.signals(data);
          } else if (data.total_pnl !== undefined) {
            messageHandlers.performance(data);
          }
        }
    }
  };

  // Heartbeat mechanism
  const startHeartbeat = (ws) => {
    stopHeartbeat();
    heartbeatIntervalRef.current = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'heartbeat' }));
      }
    }, 30000); // Send heartbeat every 30 seconds
  };

  const stopHeartbeat = () => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  };

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    console.log('Disconnecting WebSocket...');
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    stopHeartbeat();
    
    if (socket) {
      socket.close();
      setSocket(null);
    }
    
    setConnected(false);
    setConnectionStatus('disconnected');
  }, [socket]);

  // Subscribe to specific channels
  const subscribe = useCallback((channels) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: 'subscribe',
        channels: Array.isArray(channels) ? channels : [channels]
      }));
    }
  }, [socket]);

  // Unsubscribe from specific channels
  const unsubscribe = useCallback((channels) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: 'unsubscribe',
        channels: Array.isArray(channels) ? channels : [channels]
      }));
    }
  }, [socket]);

  // Send custom message
  const sendMessage = useCallback((message) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('Cannot send message: WebSocket not connected');
    }
  }, [socket]);

  // Initialize connection on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, []);

  // Context value
  const value = {
    // Connection state
    socket,
    connected,
    connectionStatus,
    
    // Data
    marketData,
    systemStatus,
    tradingSignals,
    performanceData,
    systemMetrics,
    
    // Methods
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    sendMessage,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Custom hook to use WebSocket context
export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

// Export specific data hooks for convenience
export const useMarketData = () => {
  const { marketData } = useWebSocket();
  return marketData;
};

export const useTradingSignals = () => {
  const { tradingSignals } = useWebSocket();
  return tradingSignals;
};

export const useSystemStatus = () => {
  const { systemStatus, systemMetrics } = useWebSocket();
  return { systemStatus, systemMetrics };
};

export const usePerformanceData = () => {
  const { performanceData } = useWebSocket();
  return performanceData;
};

export default WebSocketContext;