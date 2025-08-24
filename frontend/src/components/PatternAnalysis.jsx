import React, { useState, useEffect, useContext } from 'react';
import { WebSocketContext } from '../context/WebSocketContext';
import { SymbolContext } from '../context/SymbolContext';
import '../styles/PatternAnalysis.css';

const PatternAnalysis = () => {
  const { latestData } = useContext(WebSocketContext);
  const { currentSymbol } = useContext(SymbolContext);
  const [activePatterns, setActivePatterns] = useState([]);
  const [patternHistory, setPatternHistory] = useState([]);

  useEffect(() => {
    if (latestData && latestData.quantumPatterns) {
      setActivePatterns(latestData.quantumPatterns.active || []);
      setPatternHistory(latestData.quantumPatterns.history || []);
    }
  }, [latestData]);

  const renderPatternCard = (pattern, index) => {
    return (
      <div key={index} className="pattern-card">
        <div className="pattern-header">
          <span className="pattern-id">Pattern #{pattern.id}</span>
          <span className={`pattern-status ${pattern.stability > 0.7 ? 'stable' : 'unstable'}`}>
            {pattern.stability > 0.7 ? 'STABLE' : 'VOLATILE'}
          </span>
        </div>
        <div className="pattern-metrics">
          <div className="metric">
            <span className="label">Entropy:</span>
            <span className="value">{pattern.entropy.toFixed(3)}</span>
          </div>
          <div className="metric">
            <span className="label">Coherence:</span>
            <span className="value">{pattern.coherence.toFixed(3)}</span>
          </div>
          <div className="metric">
            <span className="label">Stability:</span>
            <span className="value">{pattern.stability.toFixed(3)}</span>
          </div>
        </div>
        <div className="pattern-timestamp">
          Detected: {new Date(pattern.timestamp).toLocaleTimeString()}
        </div>
      </div>
    );
  };

  return (
    <div className="pattern-analysis">
      <div className="analysis-header">
        <h1>Live Quantum Pattern Analysis</h1>
        <div className="current-symbol">
          Monitoring: <span className="symbol-highlight">{currentSymbol}</span>
        </div>
      </div>

      <div className="patterns-container">
        <div className="active-patterns">
          <h2>Active Patterns ({activePatterns.length})</h2>
          <div className="patterns-grid">
            {activePatterns.map((pattern, index) => renderPatternCard(pattern, index))}
          </div>
        </div>

        <div className="pattern-history">
          <h2>Pattern History</h2>
          <div className="history-list">
            {patternHistory.slice(0, 10).map((pattern, index) => (
              <div key={index} className="history-item">
                <span>Pattern #{pattern.id}</span>
                <span className="history-metrics">
                  E:{pattern.entropy.toFixed(2)} | C:{pattern.coherence.toFixed(2)} | S:{pattern.stability.toFixed(2)}
                </span>
                <span className="history-time">
                  {new Date(pattern.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatternAnalysis;