// SEP Trading System - WebSocket Context
// Real-time data connection to your WebSocket service

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';

const WebSocketContext = createContext(null);

const WS_URL =
  process.env.REACT_APP_WS_URL ||
  window._env_?.REACT_APP_WS_URL;

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  
  // Data states for SEP system
  const [marketData, setMarketData] = useState({});
  const [systemStatus, setSystemStatus] = useState({});
  const [tradingSignals, setTradingSignals] = useState([]);
  const [performanceData, setPerformanceData] = useState({});
  const [systemMetrics, setSystemMetrics] = useState({});
  
  // New data states for Valkey/Redis integration
  const [quantumSignals, setQuantumSignals] = useState({});
  const [signalHistory, setSignalHistory] = useState([]);
  const [valkeyMetrics, setValkeyMetrics] = useState({});
  const [livePatterns, setLivePatterns] = useState({});
  
  // Connection management
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const heartbeatIntervalRef = useRef(null);
  const maxReconnectAttempts = 10;
  const reconnectDelay = 3000;

  // Enhanced message handlers for Valkey/Redis integration
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
    },
    
    // New handlers for Valkey/Redis data streams
    quantum_signals: (data) => {
      if (data.instrument && data.timestamp) {
        const signalKey = `${data.instrument}:${data.timestamp}`;
        setQuantumSignals(prev => ({
          ...prev,
          [signalKey]: {
            ...data,
            received_at: new Date().toISOString()
          }
        }));
        
        // Add to history for analysis
        setSignalHistory(prev => {
          const newHistory = [data, ...prev];
          return newHistory.slice(0, 500); // Keep last 500 signals
        });
      }
    },
    
    valkey_metrics: (data) => {
      setValkeyMetrics(prev => ({
        ...prev,
        ...data,
        lastUpdate: new Date().toISOString()
      }));
    },
    
    live_patterns: (data) => {
      if (data.pattern_id) {
        setLivePatterns(prev => ({
          ...prev,
          [data.pattern_id]: {
            ...data,
            lastUpdate: new Date().toISOString()
          }
        }));
      }
    },
    
    signal_updates: (data) => {
      // Handle real-time signal state updates (entropy, stability, coherence)
      if (data.signal_key) {
        setQuantumSignals(prev => ({
          ...prev,
          [data.signal_key]: {
            ...prev[data.signal_key],
            ...data.updates,
            lastUpdate: new Date().toISOString()
          }
        }));
      }
    },
    
    // Enhanced handlers for backwards computation manifold
    manifold_update: (data) => {
      // Handle timestamped identity updates from Valkey - time-intrinsic compression
      const { timestamp, instrument, valkey_key, manifold_data } = data;
      
      setManifoldStream(prev => ({
        ...prev,
        [timestamp]: {
          instrument,
          valkey_key, // Time IS the key identifier
          manifold_data,
          received_at: Date.now(),
          time_intrinsic: true // Key identifier intrinsically contains the time
        }
      }));
    },
    
    pin_state_change: (data) => {
      // Handle pin state transitions (flux → stabilizing → converged)
      const { valkey_key, pin_state, metrics, transition_data } = data;
      
      setPinStates(prev => {
        const newStates = new Map(prev);
        newStates.set(valkey_key, {
          state: pin_state, // 'flux', 'stabilizing', 'converged'
          metrics: {
            entropy: metrics.entropy,
            stability: metrics.stability,
            coherence: metrics.coherence
          },
          transition_data,
          updated_at: Date.now(),
          backwards_derivable: pin_state === 'converged',
          unique_path: transition_data?.unique_path || false // Only ONE path can produce these metrics
        });
        return newStates;
      });
    },
    
    signal_evolution: (data) => {
      // Handle real-time signal metric evolution with backwards integration capability
      const { valkey_key, evolution_step, previous_metrics, current_metrics, derivative_strength } = data;
      
      setSignalEvolution(prev => {
        const newEvolution = new Map(prev);
        const signalHistory = newEvolution.get(valkey_key) || [];
        
        signalHistory.push({
          step: evolution_step,
          timestamp: Date.now(),
          previous: previous_metrics,
          current: current_metrics,
          derivative_strength, // How strongly backwards derivable
          entropy_delta: current_metrics.entropy - previous_metrics.entropy,
          stability_delta: current_metrics.stability - previous_metrics.stability,
          coherence_delta: current_metrics.coherence - previous_metrics.coherence,
          integration_ready: derivative_strength > 0.8 // Ready for backwards integration
        });
        
        // Keep only last 100 evolution steps per signal
        if (signalHistory.length > 100) {
          signalHistory.shift();
        }
        
        newEvolution.set(valkey_key, signalHistory);
        return newEvolution;
      });
    },
    
    backwards_computation: (data) => {
      // Handle backwards integration results - derive previous state from current + metrics
      const { valkey_key, current_state, derived_previous_state, confidence, computation_proof } = data;
      
      setBackwardsDerivations(prev => {
        const newDerivations = new Map(prev);
        newDerivations.set(valkey_key, {
          current_state,
          derived_previous_state,
          confidence,
          computation_proof, // Mathematical proof that only ONE path is possible
          computed_at: Date.now(),
          verified: confidence > 0.95,
          unique_solution: computation_proof?.unique_solution || false
        });
        return newDerivations;
      });
      
      // Update signal evolution with backwards integration results
      setSignalEvolution(prev => {
        const newEvolution = new Map(prev);
        const signalHistory = newEvolution.get(valkey_key) || [];
        
        // Add backwards computation verification to latest step
        const lastStep = signalHistory[signalHistory.length - 1];
        if (lastStep) {
          lastStep.backwards_verified = true;
          lastStep.backwards_confidence = confidence;
          lastStep.derived_previous = derived_previous_state;
          lastStep.mathematical_proof = computation_proof;
        }
        
        newEvolution.set(valkey_key, signalHistory);
        return newEvolution;
      });
    }
  };

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!WS_URL) {
      console.error('WebSocket URL not configured');
      setConnectionStatus('configuration-error');
      return;
    }
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
        
        // Subscribe to all channels including backwards computation manifold streams
        ws.send(JSON.stringify({
          type: 'subscribe',
          channels: [
            'market',
            'system',
            'signals',
            'performance',
            'trades',
            'quantum_signals',
            'valkey_metrics',
            'live_patterns',
            'signal_updates',
            'manifold_update',
            'pin_state_change',
            'signal_evolution',
            'backwards_computation'
          ]
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

  // Notify listeners of connection status changes
  useEffect(() => {
    console.log(`WebSocket status: ${connectionStatus}`);
    window.dispatchEvent(new CustomEvent('ws-status', { detail: connectionStatus }));
  }, [connectionStatus]);

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
          // Enhanced message routing for Valkey/Redis data
          if (data.symbol && data.price) {
            messageHandlers.market(data);
          } else if (data.signal_type && data.confidence) {
            messageHandlers.signals(data);
          } else if (data.total_pnl !== undefined) {
            messageHandlers.performance(data);
          } else if (data.instrument && data.entropy !== undefined) {
            // Quantum signal from Valkey
            messageHandlers.quantum_signals(data);
          } else if (data.pattern_id && data.coherence !== undefined) {
            // Live pattern data
            messageHandlers.live_patterns(data);
          } else if (data.signal_key && data.updates) {
            // Signal state updates
            messageHandlers.signal_updates(data);
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

  // Context value with Valkey/Redis data
  const value = {
    // Connection state
    socket,
    connected,
    connectionStatus,
    
    // Traditional trading data
    marketData,
    systemStatus,
    tradingSignals,
    performanceData,
    systemMetrics,
    
    // Valkey/Redis quantum data
    quantumSignals,
    signalHistory,
    valkeyMetrics,
    livePatterns,
    
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

export { WebSocketContext };
export default WebSocketContext;