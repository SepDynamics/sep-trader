import React, { useState, useEffect, useContext } from 'react';
import { WebSocketContext } from '../context/WebSocketContext';
import { SymbolContext } from '../context/SymbolContext';
import EntropyBandAnalysis from './EntropyBandAnalysis';
import PathHistoryMatching from './PathHistoryMatching';
import ContextualRelevanceScoring from './ContextualRelevanceScoring';
import '../styles/Dashboard.css';

const SEPDashboard = () => {
  const { latestData } = useContext(WebSocketContext);
  const { currentSymbol } = useContext(SymbolContext);
  const [systemMetrics, setSystemMetrics] = useState({
    patternsDetected: 0,
    entropyLevel: 0,
    coherenceScore: 0,
    stabilityIndex: 0
  });

  useEffect(() => {
    if (latestData && latestData.patternMetrics) {
      setSystemMetrics({
        patternsDetected: latestData.patternMetrics.totalPatterns || 0,
        entropyLevel: latestData.patternMetrics.averageEntropy || 0,
        coherenceScore: latestData.patternMetrics.averageCoherence || 0,
        stabilityIndex: latestData.patternMetrics.averageStability || 0
      });
    }
  }, [latestData]);

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