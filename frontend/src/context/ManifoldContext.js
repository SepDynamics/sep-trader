// SEP Trading System - Manifold Context
// Enhanced context for quantum manifold data and backwards computation

import React, { createContext, useContext, useEffect, useState, useMemo } from 'react';
import { useWebSocket } from './WebSocketContext';

const ManifoldContext = createContext(null);

export const ManifoldProvider = ({ children }) => {
  const {
    quantumSignals,
    signalHistory,
    livePatterns
  } = useWebSocket();

  // Enhanced manifold state management
  const [manifestoData, setManifestoData] = useState(new Map());
  const [identityTimelines, setIdentityTimelines] = useState(new Map());
  const [ruptureEvents, setRuptureEvents] = useState([]);
  const [patternEvolution, setPatternEvolution] = useState([]);
  const [selectedIdentity, setSelectedIdentity] = useState(null);

  // Process quantum signals into identity timelines
  const processedIdentities = useMemo(() => {
    const identities = new Map();
    
    // Group signals by instrument and create timelines
    Object.entries(quantumSignals).forEach(([key, signal]) => {
      const { instrument, timestamp } = signal;
      
      if (!identities.has(instrument)) {
        identities.set(instrument, []);
      }
      
      identities.get(instrument).push({
        timestamp: new Date(timestamp).getTime(),
        entropy: signal.entropy || 0.5,
        stability: signal.stability || 0.5,
        coherence: signal.coherence || 0.5,
        state: signal.state || 'flux',
        valkey_key: key,
        price: signal.price,
        volume: signal.volume,
        raw_signal: signal
      });
    });

    // Sort timelines by timestamp
    identities.forEach((timeline, instrument) => {
      timeline.sort((a, b) => a.timestamp - b.timestamp);
    });

    return identities;
  }, [quantumSignals]);

  // Generate manifold bands based on entropy and time
  const manifoldBands = useMemo(() => {
    const now = Date.now();
    const bands = {
      hot: {
        timeRange: [now - 300000, now], // Last 5 minutes - high entropy
        entropyRange: [0.7, 1.0],
        identities: [],
        color: '#ff4444',
        description: 'Fresh signals - high entropy, most die immediately'
      },
      warm: {
        timeRange: [now - 1800000, now - 300000], // 5-30 minutes - consolidation
        entropyRange: [0.3, 0.7], 
        identities: [],
        color: '#ffaa44',
        description: 'Consolidation zone - pattern formation'
      },
      cold: {
        timeRange: [now - 7200000, now - 1800000], // 30min-2h - settled
        entropyRange: [0.0, 0.3],
        identities: [],
        color: '#44ff44', 
        description: 'Settled patterns - stable, long-tail'
      }
    };

    // Classify identities into bands
    processedIdentities.forEach((timeline, instrument) => {
      timeline.forEach(point => {
        Object.entries(bands).forEach(([bandName, band]) => {
          const inTimeRange = point.timestamp >= band.timeRange[0] && point.timestamp <= band.timeRange[1];
          const inEntropyRange = point.entropy >= band.entropyRange[0] && point.entropy <= band.entropyRange[1];
          
          if (inTimeRange && inEntropyRange) {
            band.identities.push({
              instrument,
              ...point
            });
          }
        });
      });
    });

    return bands;
  }, [processedIdentities]);

  // Detect rupture events (high entropy spikes)
  useEffect(() => {
    const newRuptures = [];
    
    processedIdentities.forEach((timeline, instrument) => {
      for (let i = 1; i < timeline.length; i++) {
        const prev = timeline[i - 1];
        const curr = timeline[i];
        
        const entropyDelta = curr.entropy - prev.entropy;
        const timeDelta = curr.timestamp - prev.timestamp;
        
        // Detect rupture: entropy spike > 0.3 within 60 seconds
        if (entropyDelta > 0.3 && timeDelta < 60000) {
          newRuptures.push({
            instrument,
            timestamp: curr.timestamp,
            entropy_delta: entropyDelta,
            time_delta: timeDelta,
            rupture_strength: entropyDelta / (timeDelta / 1000), // entropy/sec
            before_state: prev.state,
            after_state: curr.state,
            valkey_key: curr.valkey_key
          });
        }
      }
    });

    if (newRuptures.length > 0) {
      setRuptureEvents(prev => [...prev, ...newRuptures].slice(-100)); // Keep last 100
    }
  }, [processedIdentities]);

  // Identity inspector data
  const getIdentityHistory = (instrument) => {
    return processedIdentities.get(instrument) || [];
  };

  const getIdentityByKey = (valkeyKey) => {
    for (const [instrument, timeline] of processedIdentities) {
      const identity = timeline.find(point => point.valkey_key === valkeyKey);
      if (identity) {
        return { instrument, ...identity };
      }
    }
    return null;
  };

  // Backwards computation helpers
  const canDeriveBackwards = (valkeyKey) => {
    const identity = getIdentityByKey(valkeyKey);
    return identity && identity.state === 'converged' && identity.entropy < 0.2;
  };

  const getDeterministicPath = (instrument) => {
    const timeline = getIdentityHistory(instrument);
    return timeline.filter(point => point.state === 'converged');
  };

  // Context value
  const value = {
    // Core data
    processedIdentities,
    manifoldBands,
    ruptureEvents,
    patternEvolution,
    
    // Selection state
    selectedIdentity,
    setSelectedIdentity,
    
    // Helper functions
    getIdentityHistory,
    getIdentityByKey,
    canDeriveBackwards,
    getDeterministicPath,
    
    // Computed metrics
    totalIdentities: processedIdentities.size,
    totalDataPoints: Array.from(processedIdentities.values()).reduce((sum, timeline) => sum + timeline.length, 0),
    ruptureCount: ruptureEvents.length,
    convergenceRate: Array.from(processedIdentities.values()).flat().filter(p => p.state === 'converged').length,
    
    // Band statistics
    bandStats: Object.fromEntries(
      Object.entries(manifoldBands).map(([name, band]) => [
        name,
        {
          count: band.identities.length,
          avgEntropy: band.identities.reduce((sum, i) => sum + i.entropy, 0) / (band.identities.length || 1),
          avgStability: band.identities.reduce((sum, i) => sum + i.stability, 0) / (band.identities.length || 1),
          avgCoherence: band.identities.reduce((sum, i) => sum + i.coherence, 0) / (band.identities.length || 1)
        }
      ])
    )
  };

  return (
    <ManifoldContext.Provider value={value}>
      {children}
    </ManifoldContext.Provider>
  );
};

export const useManifold = () => {
  const context = useContext(ManifoldContext);
  if (!context) {
    throw new Error('useManifold must be used within a ManifoldProvider');
  }
  return context;
};

export default ManifoldContext;