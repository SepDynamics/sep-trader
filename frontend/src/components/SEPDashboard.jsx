import React, { useState, useEffect, useContext } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { SymbolContext } from '../context/SymbolContext';
import EntropyBandAnalysis from './EntropyBandAnalysis';
import PathHistoryMatching from './PathHistoryMatching';
import ContextualRelevanceScoring from './ContextualRelevanceScoring';
import '../styles/Dashboard.css';

const SEPDashboard = () => {
  const { quantumSignals, livePatterns, valkeyMetrics } = useWebSocket();
  const { currentSymbol } = useContext(SymbolContext);
  const [systemMetrics, setSystemMetrics] = useState({
    patternsDetected: 0,
    entropyLevel: 0,
    coherenceScore: 0,
    stabilityIndex: 0
  });

  useEffect(() => {
    // Calculate metrics from quantum signals and live patterns
    const totalPatterns = Object.keys(livePatterns).length;
    const patternValues = Object.values(livePatterns);
    
    const avgEntropy = patternValues.length > 0
      ? patternValues.reduce((sum, pattern) => sum + (pattern.entropy || 0), 0) / patternValues.length
      : 0;
      
    const avgCoherence = patternValues.length > 0
      ? patternValues.reduce((sum, pattern) => sum + (pattern.coherence || 0), 0) / patternValues.length
      : 0;
      
    const avgStability = patternValues.length > 0
      ? patternValues.reduce((sum, pattern) => sum + (pattern.stability || 0), 0) / patternValues.length
      : 0;
      
    setSystemMetrics({
      patternsDetected: totalPatterns,
      entropyLevel: avgEntropy,
      coherenceScore: avgCoherence,
      stabilityIndex: avgStability
    });
  }, [quantumSignals, livePatterns, valkeyMetrics]);

  return (
    <div className="sep-dashboard">
      <div className="dashboard-header">
        <h1>SEP Engine - Quantum Pattern Analysis</h1>
        <div className="current-symbol">
          Analyzing: <span className="symbol-highlight">{currentSymbol}</span>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Patterns Detected</h3>
          <div className="metric-value">{systemMetrics.patternsDetected}</div>
          <div className="metric-description">Active quantum patterns in manifold</div>
        </div>
        
        <div className="metric-card">
          <h3>Entropy Level</h3>
          <div className="metric-value">{systemMetrics.entropyLevel.toFixed(3)}</div>
          <div className="metric-description">Market state disorder measurement</div>
        </div>
        
        <div className="metric-card">
          <h3>Coherence Score</h3>
          <div className="metric-value">{systemMetrics.coherenceScore.toFixed(3)}</div>
          <div className="metric-description">Pattern stability consistency</div>
        </div>
        
        <div className="metric-card">
          <h3>Stability Index</h3>
          <div className="metric-value">{systemMetrics.stabilityIndex.toFixed(3)}</div>
          <div className="metric-description">Temporal pattern persistence</div>
        </div>
      </div>

      <div className="analysis-sections">
        <div className="section-card">
          <h2>Entropy Band Distribution</h2>
          <EntropyBandAnalysis />
        </div>
        
        <div className="section-card">
          <h2>Path History Matching</h2>
          <PathHistoryMatching />
        </div>
        
        <div className="section-card">
          <h2>Contextual Relevance</h2>
          <ContextualRelevanceScoring />
        </div>
      </div>
    </div>
  );
};

export default SEPDashboard;