import React, { useState, useEffect, useMemo } from 'react';
import { Brain, TrendingUp, Clock, Layers, Star, Filter } from 'lucide-react';
import { useWebSocket } from '../context/WebSocketContext';

const ContextualRelevanceScoring = () => {
  const {
    manifoldStream,
    pinStates,
    signalEvolution,
    backwardsDerivations,
    quantumSignals,
    getSignalState,
    getSignalHistory,
    getBackwardsDerivation
  } = useWebSocket();

  const [timeWindow, setTimeWindow] = useState('15m'); // 15m, 1h, 4h
  const [relevanceMode, setRelevanceMode] = useState('temporal'); // 'temporal', 'quantum', 'contextual'
  const [minRelevanceScore, setMinRelevanceScore] = useState(0.3);
  const [selectedKey, setSelectedKey] = useState(null);

  // Calculate contextual relevance scores for timestamped keys
  const relevanceScores = useMemo(() => {
    const scores = new Map();
    const now = Date.now();
    const windowMs = {
      '15m': 900000,
      '1h': 3600000,
      '4h': 14400000
    }[timeWindow];

    // Get all active keys from manifold stream
    const activeKeys = Object.keys(manifoldStream)
      .filter(timestamp => now - parseInt(timestamp) <= windowMs)
      .sort((a, b) => b - a); // Most recent first

    activeKeys.forEach((timestamp, index) => {
      const manifoldData = manifoldStream[timestamp];
      const valkeyKey = manifoldData.valkey_key;
      const pinState = pinStates.get(valkeyKey);
      const signalHistory = signalEvolution.get(valkeyKey) || [];
      const backwardsDeriv = backwardsDerivations.get(valkeyKey);

      // Calculate multi-dimensional relevance
      const score = calculateRelevanceScore({
        timestamp: parseInt(timestamp),
        index,
        totalKeys: activeKeys.length,
        manifoldData,
        pinState,
        signalHistory,
        backwardsDeriv,
        now,
        windowMs,
        mode: relevanceMode
      });

      scores.set(valkeyKey, {
        timestamp: parseInt(timestamp),
        valkeyKey,
        instrument: manifoldData.instrument,
        score: score.total,
        components: score.components,
        rank: 0, // Will be set after sorting
        contextual_factors: score.contextual_factors,
        time_intrinsic_value: score.time_intrinsic_value
      });
    });

    // Rank scores and assign ranks
    const sortedScores = Array.from(scores.values())
      .sort((a, b) => b.score - a.score);
    
    sortedScores.forEach((scoreData, index) => {
      scoreData.rank = index + 1;
      scores.set(scoreData.valkeyKey, scoreData);
    });

    return scores;
  }, [manifoldStream, pinStates, signalEvolution, backwardsDerivations, timeWindow, relevanceMode]);

  // Helper function to calculate comprehensive relevance score
  const calculateRelevanceScore = ({
    timestamp, index, totalKeys, manifoldData, pinState, signalHistory, backwardsDeriv, now, windowMs, mode
  }) => {
    const components = {};
    
    // 1. Temporal Relevance - how recent and time-intrinsically valuable
    const timeAge = now - timestamp;
    const timeRelevance = Math.exp(-timeAge / (windowMs * 0.3)); // Exponential decay
    const timeIntrinsicValue = 1 - (index / totalKeys); // Newer keys are more intrinsically valuable
    components.temporal = timeRelevance * 0.7 + timeIntrinsicValue * 0.3;

    // 2. Quantum State Relevance - entropy, stability, coherence dynamics
    let quantumRelevance = 0.5; // Default if no pin state
    if (pinState && pinState.metrics) {
      const entropy = pinState.metrics.entropy;
      const stability = pinState.metrics.stability;
      const coherence = pinState.metrics.coherence;
      
      // High relevance for signals in transition zones
      const entropyFactor = entropy > 0.3 && entropy < 0.8 ? 1.0 : 0.6;
      const stabilityFactor = stability > 0.5 ? Math.min(stability * 1.2, 1.0) : stability;
      const coherenceFactor = coherence > 0.6 ? Math.min(coherence * 1.1, 1.0) : coherence;
      
      quantumRelevance = (entropyFactor + stabilityFactor + coherenceFactor) / 3;
    }
    components.quantum = quantumRelevance;

    // 3. Evolution Relevance - signal development and backwards computation potential
    let evolutionRelevance = 0.3; // Default
    if (signalHistory.length > 0) {
      const recentEvolution = signalHistory.slice(-5); // Last 5 steps
      const evolutionVelocity = recentEvolution.reduce((sum, step) => {
        return sum + Math.abs(step.entropy_delta) + Math.abs(step.stability_delta) + Math.abs(step.coherence_delta);
      }, 0) / recentEvolution.length;
      
      const integrationReadiness = recentEvolution.filter(step => step.integration_ready).length / recentEvolution.length;
      evolutionRelevance = Math.min(evolutionVelocity * 2 + integrationReadiness, 1.0);
    }
    components.evolution = evolutionRelevance;

    // 4. Backwards Computation Relevance - mathematical uniqueness
    let backwardsRelevance = 0.2; // Default
    if (backwardsDeriv && backwardsDeriv.verified) {
      backwardsRelevance = backwardsDeriv.confidence * 0.8 + 
                          (backwardsDeriv.unique_solution ? 0.2 : 0);
    }
    components.backwards = backwardsRelevance;

    // 5. Contextual Market Relevance - instrument importance, market conditions
    const instrumentWeight = getInstrumentWeight(manifoldData.instrument);
    const marketVolatility = calculateMarketVolatility(manifoldData.instrument, timestamp);
    const contextualRelevance = instrumentWeight * 0.6 + marketVolatility * 0.4;
    components.contextual = contextualRelevance;

    // Calculate weighted total based on mode
    let total;
    switch (mode) {
      case 'temporal':
        total = components.temporal * 0.4 + components.quantum * 0.25 + 
                components.evolution * 0.2 + components.backwards * 0.1 + components.contextual * 0.05;
        break;
      case 'quantum':
        total = components.quantum * 0.4 + components.evolution * 0.3 + 
                components.backwards * 0.2 + components.temporal * 0.05 + components.contextual * 0.05;
        break;
      case 'contextual':
        total = components.contextual * 0.35 + components.temporal * 0.25 + 
                components.quantum * 0.2 + components.evolution * 0.15 + components.backwards * 0.05;
        break;
      default:
        total = Object.values(components).reduce((sum, val) => sum + val, 0) / Object.keys(components).length;
    }

    return {
      total: Math.min(total, 1.0),
      components,
      contextual_factors: {
        instrument_weight: instrumentWeight,
        market_volatility: marketVolatility,
        time_distance: timeAge / windowMs
      },
      time_intrinsic_value: timeIntrinsicValue
    };
  };

  // Helper function to get instrument trading weight
  const getInstrumentWeight = (instrument) => {
    const weights = {
      'EUR_USD': 0.95,
      'GBP_USD': 0.90,
      'USD_JPY': 0.88,
      'USD_CHF': 0.82,
      'AUD_USD': 0.78,
      'USD_CAD': 0.75,
      'NZD_USD': 0.68
    };
    return weights[instrument] || 0.5;
  };

  // Helper function to calculate market volatility
  const calculateMarketVolatility = (instrument, timestamp) => {
    // Mock calculation - in real implementation, would use recent price movements
    const baseVolatility = Math.sin(timestamp / 3600000) * 0.3 + 0.5; // Simulate market cycles
    return Math.max(0.2, Math.min(baseVolatility, 0.9));
  };

  // Filter scores by minimum relevance
  const filteredScores = useMemo(() => {
    return Array.from(relevanceScores.values())
      .filter(score => score.score >= minRelevanceScore)
      .sort((a, b) => b.score - a.score);
  }, [relevanceScores, minRelevanceScore]);

  const getRelevanceColor = (score) => {
    if (score >= 0.8) return 'from-green-500 to-green-400';
    if (score >= 0.6) return 'from-yellow-500 to-yellow-400';
    if (score >= 0.4) return 'from-orange-500 to-orange-400';
    return 'from-red-500 to-red-400';
  };

  const getRelevanceLabel = (score) => {
    if (score >= 0.8) return 'Critical';
    if (score >= 0.6) return 'High';
    if (score >= 0.4) return 'Medium';
    return 'Low';
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Brain className="w-6 h-6 text-cyan-400" />
          Contextual Relevance Scoring
        </h2>
        
        <div className="flex gap-4 items-center">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Min Score:</label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.1"
              value={minRelevanceScore}
              onChange={(e) => setMinRelevanceScore(parseFloat(e.target.value))}
              className="w-16"
            />
            <span className="text-sm text-white">{minRelevanceScore}</span>
          </div>

          <select
            value={timeWindow}
            onChange={(e) => setTimeWindow(e.target.value)}
            className="bg-gray-700 text-white px-3 py-1 rounded text-sm"
          >
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
          </select>

          <select
            value={relevanceMode}
            onChange={(e) => setRelevanceMode(e.target.value)}
            className="bg-gray-700 text-white px-3 py-1 rounded text-sm"
          >
            <option value="temporal">Temporal Focus</option>
            <option value="quantum">Quantum Focus</option>
            <option value="contextual">Market Context</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Relevance Overview */}
        <div className="lg:col-span-1">
          <h3 className="text-lg font-semibold text-white mb-4">Relevance Overview</h3>
          
          <div className="space-y-3 mb-6">
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400">Total Keys</span>
                <span className="text-white font-bold">{relevanceScores.size}</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400">Filtered</span>
                <span className="text-cyan-400 font-bold">{filteredScores.length}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Time Window</span>
                <span className="text-blue-400">{timeWindow}</span>
              </div>
            </div>

            {/* Top Scoring Categories */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h4 className="font-medium text-white mb-3">Score Distribution</h4>
              <div className="space-y-2">
                {['Critical', 'High', 'Medium', 'Low'].map(label => {
                  const count = filteredScores.filter(s => getRelevanceLabel(s.score) === label).length;
                  const percentage = filteredScores.length > 0 ? (count / filteredScores.length) * 100 : 0;
                  
                  return (
                    <div key={label} className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded bg-gradient-to-r ${
                          label === 'Critical' ? 'from-green-500 to-green-400' :
                          label === 'High' ? 'from-yellow-500 to-yellow-400' :
                          label === 'Medium' ? 'from-orange-500 to-orange-400' :
                          'from-red-500 to-red-400'
                        }`}></div>
                        <span className="text-gray-300">{label}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-white">{count}</span>
                        <span className="text-gray-400">({percentage.toFixed(1)}%)</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Time-Intrinsic Value Explanation */}
          <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-600/30">
            <h4 className="font-medium text-purple-300 mb-2 flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Time-Intrinsic Principle
            </h4>
            <div className="text-sm text-purple-200 space-y-1">
              <div>• Key identifier = timestamp</div>
              <div>• No external naming required</div>
              <div>• Relevance inherent in time position</div>
              <div>• Backwards computation preserves causality</div>
            </div>
          </div>
        </div>

        {/* Scored Keys List */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Relevance-Scored Keys</h3>
            <div className="text-sm text-gray-400">
              Showing {filteredScores.length} keys (mode: {relevanceMode})
            </div>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {filteredScores.map((scoreData, index) => (
              <div
                key={scoreData.valkeyKey}
                onClick={() => setSelectedKey(scoreData)}
                className={`p-4 rounded cursor-pointer transition-all ${
                  selectedKey?.valkeyKey === scoreData.valkeyKey
                    ? 'bg-cyan-600 text-white transform scale-105'
                    : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                      <Star className={`w-4 h-4 ${
                        scoreData.rank <= 3 ? 'text-yellow-400 fill-current' : 'text-gray-500'
                      }`} />
                      <span className="text-sm font-mono">#{scoreData.rank}</span>
                    </div>
                    <div>
                      <div className="font-medium">{scoreData.instrument}</div>
                      <div className="text-xs opacity-75">
                        {new Date(scoreData.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className={`text-lg font-bold bg-gradient-to-r ${getRelevanceColor(scoreData.score)} bg-clip-text text-transparent`}>
                      {(scoreData.score * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs opacity-75">
                      {getRelevanceLabel(scoreData.score)}
                    </div>
                  </div>
                </div>

                {/* Component Breakdown */}
                <div className="grid grid-cols-5 gap-1 text-xs">
                  {Object.entries(scoreData.components).map(([component, value]) => (
                    <div key={component} className="text-center">
                      <div className="capitalize text-gray-400 mb-1">{component.substring(0, 4)}</div>
                      <div className="h-1 bg-gray-700 rounded overflow-hidden">
                        <div
                          className={`h-full bg-gradient-to-r ${getRelevanceColor(value)}`}
                          style={{ width: `${value * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Time-Intrinsic Value Indicator */}
                <div className="mt-2 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <Clock className="w-3 h-3 text-purple-400" />
                    <span className="text-purple-300">
                      Intrinsic: {(scoreData.time_intrinsic_value * 100).toFixed(0)}%
                    </span>
                  </div>
                  {scoreData.contextual_factors.market_volatility > 0.7 && (
                    <div className="text-orange-400">High Volatility</div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {filteredScores.length === 0 && (
            <div className="text-center py-8 text-gray-400">
              <Filter className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <div>No keys meet relevance criteria</div>
              <div className="text-sm">Try lowering the minimum score threshold</div>
            </div>
          )}
        </div>
      </div>

      {/* Detailed Analysis Panel */}
      {selectedKey && (
        <div className="mt-6 bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">
              Detailed Analysis: {selectedKey.instrument}
            </h3>
            <button
              onClick={() => setSelectedKey(null)}
              className="text-gray-400 hover:text-white"
            >
              ×
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Relevance Components */}
            <div>
              <h4 className="font-medium text-white mb-3">Relevance Components</h4>
              <div className="space-y-3">
                {Object.entries(selectedKey.components).map(([component, value]) => (
                  <div key={component}>
                    <div className="flex justify-between mb-1">
                      <span className="text-gray-400 capitalize">{component}</span>
                      <span className="text-white">{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded overflow-hidden">
                      <div
                        className={`h-full bg-gradient-to-r ${getRelevanceColor(value)}`}
                        style={{ width: `${value * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Contextual Factors */}
            <div>
              <h4 className="font-medium text-white mb-3">Contextual Factors</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Instrument Weight:</span>
                  <span className="text-cyan-400">
                    {(selectedKey.contextual_factors.instrument_weight * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Market Volatility:</span>
                  <span className="text-orange-400">
                    {(selectedKey.contextual_factors.market_volatility * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Time Distance:</span>
                  <span className="text-blue-400">
                    {(selectedKey.contextual_factors.time_distance * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Global Rank:</span>
                  <span className="text-yellow-400">#{selectedKey.rank}</span>
                </div>
              </div>
            </div>

            {/* Time-Intrinsic Analysis */}
            <div>
              <h4 className="font-medium text-white mb-3">Time-Intrinsic Analysis</h4>
              <div className="space-y-2 text-sm">
                <div className="p-3 bg-purple-900/20 rounded border border-purple-600/20">
                  <div className="text-purple-300 font-medium mb-2">Key Properties</div>
                  <div className="space-y-1 text-purple-200">
                    <div>Timestamp: {selectedKey.timestamp}</div>
                    <div>Valkey Key: {selectedKey.valkeyKey}</div>
                    <div>Intrinsic Value: {(selectedKey.time_intrinsic_value * 100).toFixed(1)}%</div>
                  </div>
                </div>
                
                <div className="text-xs text-gray-400 italic mt-2">
                  Time-intrinsic keys require no external naming - 
                  the temporal position contains all identity information needed
                  for backwards computation and pattern matching.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ContextualRelevanceScoring;