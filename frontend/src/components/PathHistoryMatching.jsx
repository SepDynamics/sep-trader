import React, { useState, useEffect, useMemo } from 'react';
import { GitBranch, Target, TrendingUp, Clock, Zap, Search } from 'lucide-react';
import { useWebSocket } from '../context/WebSocketContext';

const PathHistoryMatching = () => {
  const {
    signalEvolution,
    backwardsDerivations,
    signalHistory,
    manifoldStream,
    getSignalHistory,
    getBackwardsDerivation
  } = useWebSocket();

  const [selectedPattern, setSelectedPattern] = useState(null);
  const [matchThreshold, setMatchThreshold] = useState(0.85);
  const [timeWindow, setTimeWindow] = useState('1h'); // 1h, 4h, 1d
  const [viewMode, setViewMode] = useState('patterns'); // 'patterns', 'matches', 'tree'

  // Generate pattern signatures from signal evolution
  const patternSignatures = useMemo(() => {
    const signatures = new Map();

    Array.from(signalEvolution.keys()).forEach(signalKey => {
      const history = signalEvolution.get(signalKey);
      if (history && history.length >= 3) {
        // Create pattern signature from evolution sequence
        const signature = {
          signal_key: signalKey,
          instrument: signalKey.split(':')[0],
          timestamp: history[0]?.timestamp,
          sequence: history.map(step => ({
            entropy: step.current.entropy,
            stability: step.current.stability,
            coherence: step.current.coherence,
            entropy_delta: step.entropy_delta,
            stability_delta: step.stability_delta,
            coherence_delta: step.coherence_delta
          })),
          backwards_verified: history.some(step => step.backwards_verified),
          pattern_hash: generatePatternHash(history),
          convergence_point: findConvergencePoint(history),
          outcome_metrics: getOutcomeMetrics(history)
        };

        signatures.set(signalKey, signature);
      }
    });

    return signatures;
  }, [signalEvolution]);

  // Find historical matches for current patterns
  const patternMatches = useMemo(() => {
    const matches = [];
    const currentPatterns = Array.from(patternSignatures.values())
      .filter(sig => isRecentPattern(sig.timestamp, timeWindow));

    currentPatterns.forEach(currentPattern => {
      const historicalMatches = findHistoricalMatches(currentPattern, patternSignatures, matchThreshold);
      if (historicalMatches.length > 0) {
        matches.push({
          current: currentPattern,
          historical: historicalMatches,
          confidence: calculateMatchConfidence(currentPattern, historicalMatches),
          prediction: generatePrediction(currentPattern, historicalMatches)
        });
      }
    });

    return matches.sort((a, b) => b.confidence - a.confidence);
  }, [patternSignatures, matchThreshold, timeWindow]);

  // Generate decision tree based on pattern matching
  const decisionTree = useMemo(() => {
    const tree = {
      root: {
        condition: 'Initial State',
        branches: []
      }
    };

    patternMatches.forEach(match => {
      const branch = {
        pattern_id: match.current.signal_key,
        instrument: match.current.instrument,
        confidence: match.confidence,
        historical_outcomes: match.historical.map(h => h.outcome_metrics),
        prediction: match.prediction,
        backwards_path: getBackwardsPath(match.current),
        decision_points: identifyDecisionPoints(match.current, match.historical)
      };

      tree.root.branches.push(branch);
    });

    return tree;
  }, [patternMatches]);

  // Helper function to generate pattern hash
  const generatePatternHash = (history) => {
    const sequence = history.map(step => 
      `${Math.round(step.entropy_delta * 1000)}-${Math.round(step.stability_delta * 1000)}-${Math.round(step.coherence_delta * 1000)}`
    ).join('|');
    return btoa(sequence).substring(0, 8);
  };

  // Helper function to find convergence point
  const findConvergencePoint = (history) => {
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].current.entropy < 0.3 && history[i].current.stability > 0.7) {
        return {
          step: i,
          timestamp: history[i].timestamp,
          metrics: history[i].current
        };
      }
    }
    return null;
  };

  // Helper function to get outcome metrics
  const getOutcomeMetrics = (history) => {
    const lastStep = history[history.length - 1];
    return {
      final_entropy: lastStep.current.entropy,
      final_stability: lastStep.current.stability,
      final_coherence: lastStep.current.coherence,
      convergence_time: history.length,
      backwards_verified: lastStep.backwards_verified || false
    };
  };

  // Helper function to check if pattern is recent
  const isRecentPattern = (timestamp, window) => {
    const now = Date.now();
    const windowMs = {
      '1h': 3600000,
      '4h': 14400000,
      '1d': 86400000
    }[window];
    return now - timestamp < windowMs;
  };

  // Helper function to find historical matches
  const findHistoricalMatches = (currentPattern, allSignatures, threshold) => {
    const matches = [];
    
    Array.from(allSignatures.values()).forEach(historical => {
      if (historical.signal_key === currentPattern.signal_key) return;
      if (historical.instrument !== currentPattern.instrument) return;
      
      const similarity = calculatePatternSimilarity(currentPattern, historical);
      if (similarity >= threshold) {
        matches.push({
          ...historical,
          similarity,
          time_difference: currentPattern.timestamp - historical.timestamp
        });
      }
    });

    return matches.sort((a, b) => b.similarity - a.similarity);
  };

  // Helper function to calculate pattern similarity
  const calculatePatternSimilarity = (pattern1, pattern2) => {
    if (pattern1.sequence.length !== pattern2.sequence.length) return 0;
    
    let totalSimilarity = 0;
    for (let i = 0; i < pattern1.sequence.length; i++) {
      const s1 = pattern1.sequence[i];
      const s2 = pattern2.sequence[i];
      
      const entropySim = 1 - Math.abs(s1.entropy - s2.entropy);
      const stabilitySim = 1 - Math.abs(s1.stability - s2.stability);
      const coherenceSim = 1 - Math.abs(s1.coherence - s2.coherence);
      
      totalSimilarity += (entropySim + stabilitySim + coherenceSim) / 3;
    }
    
    return totalSimilarity / pattern1.sequence.length;
  };

  // Helper function to calculate match confidence
  const calculateMatchConfidence = (current, matches) => {
    if (matches.length === 0) return 0;
    
    const avgSimilarity = matches.reduce((sum, m) => sum + m.similarity, 0) / matches.length;
    const backwardsBonus = current.backwards_verified ? 0.1 : 0;
    const convergenceBonus = current.convergence_point ? 0.1 : 0;
    
    return Math.min(avgSimilarity + backwardsBonus + convergenceBonus, 1.0);
  };

  // Helper function to generate prediction
  const generatePrediction = (current, matches) => {
    const outcomes = matches.map(m => m.outcome_metrics);
    const avgFinalEntropy = outcomes.reduce((sum, o) => sum + o.final_entropy, 0) / outcomes.length;
    const avgConvergenceTime = outcomes.reduce((sum, o) => sum + o.convergence_time, 0) / outcomes.length;
    
    return {
      predicted_final_entropy: avgFinalEntropy,
      predicted_convergence_time: avgConvergenceTime,
      confidence: calculateMatchConfidence(current, matches),
      suggested_action: avgFinalEntropy < 0.3 ? 'TRADE' : 'WAIT'
    };
  };

  // Helper function to get backwards path
  const getBackwardsPath = (pattern) => {
    const derivation = getBackwardsDerivation(pattern.signal_key);
    if (!derivation) return null;
    
    return {
      current_state: derivation.current_state,
      derived_previous: derivation.derived_previous_state,
      confidence: derivation.confidence,
      unique_solution: derivation.unique_solution
    };
  };

  // Helper function to identify decision points
  const identifyDecisionPoints = (current, historical) => {
    const points = [];
    
    // Find critical entropy thresholds
    if (current.sequence.some(s => s.entropy < 0.5 && s.entropy_delta < 0)) {
      points.push({
        type: 'entropy_threshold',
        description: 'Entropy dropping below 0.5',
        historical_success_rate: historical.filter(h => 
          h.outcome_metrics.final_entropy < 0.3
        ).length / historical.length
      });
    }
    
    // Find stability convergence points
    if (current.sequence.some(s => s.stability > 0.7 && s.stability_delta > 0)) {
      points.push({
        type: 'stability_convergence',
        description: 'Stability rising above 0.7',
        historical_success_rate: historical.filter(h => 
          h.outcome_metrics.final_stability > 0.8
        ).length / historical.length
      });
    }
    
    return points;
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <GitBranch className="w-6 h-6 text-green-400" />
          Path History Matching
        </h2>
        
        <div className="flex gap-4 items-center">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Threshold:</label>
            <input
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={matchThreshold}
              onChange={(e) => setMatchThreshold(parseFloat(e.target.value))}
              className="w-20"
            />
            <span className="text-sm text-white">{matchThreshold}</span>
          </div>
          
          <select
            value={timeWindow}
            onChange={(e) => setTimeWindow(e.target.value)}
            className="bg-gray-700 text-white px-3 py-1 rounded text-sm"
          >
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
          </select>
          
          <div className="flex gap-1">
            {['patterns', 'matches', 'tree'].map(mode => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1 rounded text-sm capitalize ${
                  viewMode === mode 
                    ? 'bg-green-600 text-white' 
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </div>

      {viewMode === 'patterns' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Current Patterns</h3>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {Array.from(patternSignatures.values())
                .filter(sig => isRecentPattern(sig.timestamp, timeWindow))
                .map(pattern => (
                  <div
                    key={pattern.signal_key}
                    onClick={() => setSelectedPattern(pattern)}
                    className={`p-4 rounded cursor-pointer transition-colors ${
                      selectedPattern?.signal_key === pattern.signal_key
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <div className="font-mono text-sm">{pattern.instrument}</div>
                        <div className="text-xs opacity-75">Hash: {pattern.pattern_hash}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs">Steps: {pattern.sequence.length}</div>
                        {pattern.backwards_verified && (
                          <div className="text-xs text-purple-400">✓ Verified</div>
                        )}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <div className="text-gray-400">Entropy</div>
                        <div className="text-red-400">
                          {pattern.outcome_metrics.final_entropy.toFixed(3)}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Stability</div>
                        <div className="text-orange-400">
                          {pattern.outcome_metrics.final_stability.toFixed(3)}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Coherence</div>
                        <div className="text-green-400">
                          {pattern.outcome_metrics.final_coherence.toFixed(3)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Pattern Analysis</h3>
            {selectedPattern ? (
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="mb-4">
                  <h4 className="font-medium text-white mb-2">Evolution Sequence</h4>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {selectedPattern.sequence.map((step, index) => (
                      <div key={index} className="flex justify-between text-sm">
                        <span className="text-gray-400">Step {index + 1}</span>
                        <div className="flex gap-3">
                          <span className="text-red-400">E: {step.entropy.toFixed(3)}</span>
                          <span className="text-orange-400">S: {step.stability.toFixed(3)}</span>
                          <span className="text-green-400">C: {step.coherence.toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {selectedPattern.convergence_point && (
                  <div className="mb-4 p-3 bg-teal-900/30 rounded border border-teal-600/30">
                    <h4 className="font-medium text-teal-300 mb-2">Convergence Point</h4>
                    <div className="text-sm text-teal-200">
                      Step {selectedPattern.convergence_point.step} - 
                      {new Date(selectedPattern.convergence_point.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                )}

                {selectedPattern.backwards_verified && (
                  <div className="p-3 bg-purple-900/30 rounded border border-purple-600/30">
                    <h4 className="font-medium text-purple-300 mb-2">Backwards Verification</h4>
                    <div className="text-sm text-purple-200">
                      Pattern has been mathematically verified through backwards computation
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-gray-800 rounded-lg p-4 text-center text-gray-400">
                <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <div>Select a pattern to view analysis</div>
              </div>
            )}
          </div>
        </div>
      )}

      {viewMode === 'matches' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white mb-4">Pattern Matches</h3>
          
          {patternMatches.map((match, index) => (
            <div key={index} className="bg-gray-800 rounded-lg p-4">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h4 className="font-medium text-white">{match.current.instrument}</h4>
                  <div className="text-sm text-gray-400">
                    Current: {match.current.pattern_hash} | 
                    Matches: {match.historical.length} | 
                    Confidence: {(match.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div className={`px-3 py-1 rounded text-sm font-medium ${
                  match.prediction.suggested_action === 'TRADE' 
                    ? 'bg-green-600 text-white' 
                    : 'bg-yellow-600 text-white'
                }`}>
                  {match.prediction.suggested_action}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h5 className="font-medium text-gray-300 mb-2">Prediction</h5>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Final Entropy:</span>
                      <span className="text-red-400">
                        {match.prediction.predicted_final_entropy.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Convergence Time:</span>
                      <span className="text-orange-400">
                        {match.prediction.predicted_convergence_time.toFixed(0)} steps
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h5 className="font-medium text-gray-300 mb-2">Historical Performance</h5>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Success Rate:</span>
                      <span className="text-green-400">
                        {((match.historical.filter(h => h.outcome_metrics.final_entropy < 0.3).length / match.historical.length) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Avg Similarity:</span>
                      <span className="text-blue-400">
                        {(match.historical.reduce((sum, h) => sum + h.similarity, 0) / match.historical.length).toFixed(3)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}

          {patternMatches.length === 0 && (
            <div className="text-center py-8 text-gray-400">
              <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <div>No pattern matches found</div>
              <div className="text-sm">Try lowering the match threshold</div>
            </div>
          )}
        </div>
      )}

      {viewMode === 'tree' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white mb-4">Decision Tree</h3>
          
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="mb-4">
              <h4 className="font-medium text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-blue-400" />
                Root: Market State Analysis
              </h4>
            </div>

            <div className="space-y-4">
              {decisionTree.root.branches.map((branch, index) => (
                <div key={index} className="ml-6 border-l border-gray-600 pl-4">
                  <div className="flex items-start gap-4">
                    <div className="w-3 h-3 bg-blue-500 rounded-full mt-2"></div>
                    <div className="flex-1">
                      <h5 className="font-medium text-white">{branch.instrument}</h5>
                      <div className="text-sm text-gray-400 mb-2">
                        Confidence: {(branch.confidence * 100).toFixed(1)}% | 
                        Prediction: {branch.prediction.suggested_action}
                      </div>

                      {branch.decision_points.length > 0 && (
                        <div className="space-y-2">
                          {branch.decision_points.map((point, pointIndex) => (
                            <div key={pointIndex} className="ml-6 border-l border-gray-700 pl-4">
                              <div className="flex items-start gap-3">
                                <div className="w-2 h-2 bg-green-400 rounded-full mt-2"></div>
                                <div>
                                  <div className="text-sm font-medium text-green-400">
                                    {point.description}
                                  </div>
                                  <div className="text-xs text-gray-500">
                                    Historical Success: {(point.historical_success_rate * 100).toFixed(1)}%
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      {branch.backwards_path && (
                        <div className="mt-2 p-2 bg-purple-900/20 rounded border border-purple-600/20">
                          <div className="text-xs text-purple-400 font-medium mb-1">
                            🔄 Backwards Path Verified
                          </div>
                          <div className="text-xs text-purple-300">
                            Confidence: {(branch.backwards_path.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {decisionTree.root.branches.length === 0 && (
              <div className="text-center py-8 text-gray-400">
                <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <div>No decision branches available</div>
                <div className="text-sm">Waiting for pattern convergence...</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PathHistoryMatching;