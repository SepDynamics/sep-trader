import React, { useState, useEffect, useMemo } from 'react';
import { TrendingUp, TrendingDown, Zap, Target, Activity } from 'lucide-react';
import { useWebSocket } from '../context/WebSocketContext';

const EntropyBandAnalysis = () => {
  const {
    signalEvolution,
    pinStates,
    backwardsDerivations,
    getSignalState,
    getSignalHistory,
    getActiveSignals,
    getConvergedSignals,
    getDerivableSignals
  } = useWebSocket();

  const [selectedSignal, setSelectedSignal] = useState(null);
  const [viewMode, setViewMode] = useState('bands'); // 'bands', 'evolution', 'decisions'

  // Compute entropy band distribution
  const entropyBands = useMemo(() => {
    const activeSignals = getActiveSignals();
    const convergedSignals = getConvergedSignals();
    
    const bands = {
      flux: [], // High entropy (0.8-1.0)
      stabilizing: [], // Medium entropy (0.3-0.8)
      converged: [], // Low entropy (0.0-0.3)
      backwards_ready: [] // Signals ready for backwards computation
    };

    // Categorize active signals by entropy
    activeSignals.forEach(key => {
      const state = pinStates.get(key);
      if (state && state.metrics) {
        const entropy = state.metrics.entropy;
        if (entropy >= 0.8) {
          bands.flux.push({ key, state, entropy });
        } else if (entropy >= 0.3) {
          bands.stabilizing.push({ key, state, entropy });
        } else {
          bands.converged.push({ key, state, entropy });
        }
      }
    });

    // Add backwards computation ready signals
    const derivableSignals = getDerivableSignals();
    derivableSignals.forEach(key => {
      const history = signalEvolution.get(key) || [];
      const latestStep = history[history.length - 1];
      if (latestStep && latestStep.integration_ready) {
        bands.backwards_ready.push({ key, latestStep });
      }
    });

    return bands;
  }, [pinStates, signalEvolution, getActiveSignals, getConvergedSignals, getDerivableSignals]);

  // Generate trading decisions based on entropy analysis
  const tradingDecisions = useMemo(() => {
    const decisions = [];

    // High confidence decisions from converged signals
    entropyBands.converged.forEach(({ key, state }) => {
      const backwardsDeriv = backwardsDerivations.get(key);
      if (backwardsDeriv && backwardsDeriv.verified) {
        decisions.push({
          signal_key: key,
          decision: 'TRADE',
          confidence: backwardsDeriv.confidence,
          reasoning: 'Pin state converged + backwards derivation verified',
          entropy_band: 'converged',
          action: state.metrics.stability > 0.7 ? 'BUY' : 'SELL'
        });
      }
    });

    // Medium confidence from stabilizing signals
    entropyBands.stabilizing.forEach(({ key, state, entropy }) => {
      if (state.transition_data?.unique_path) {
        decisions.push({
          signal_key: key,
          decision: 'WATCH',
          confidence: 0.7 - entropy * 0.2,
          reasoning: 'Stabilizing with unique path identified',
          entropy_band: 'stabilizing',
          action: 'PREPARE'
        });
      }
    });

    return decisions.sort((a, b) => b.confidence - a.confidence);
  }, [entropyBands, backwardsDerivations]);

  const getEntropyColor = (entropy) => {
    if (entropy >= 0.8) return 'from-red-500 to-orange-500';
    if (entropy >= 0.3) return 'from-orange-500 to-yellow-500';
    return 'from-teal-500 to-green-500';
  };

  const getBandLabel = (entropy) => {
    if (entropy >= 0.8) return 'High Flux';
    if (entropy >= 0.3) return 'Stabilizing';
    return 'Converged';
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Activity className="w-6 h-6 text-purple-400" />
          Entropy Band Analysis
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('bands')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'bands' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Bands
          </button>
          <button
            onClick={() => setViewMode('evolution')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'evolution' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Evolution
          </button>
          <button
            onClick={() => setViewMode('decisions')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'decisions' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Decisions
          </button>
        </div>
      </div>

      {viewMode === 'bands' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Entropy Band Distribution */}
          <div className="lg:col-span-2 space-y-4">
            <h3 className="text-lg font-semibold text-white mb-4">Live Entropy Bands</h3>
            
            {/* High Flux Band */}
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-gradient-to-r from-red-500 to-orange-500 rounded-full animate-pulse"></div>
                  <span className="font-medium">High Flux Band (0.8-1.0)</span>
                </div>
                <span className="text-red-400 font-bold">{entropyBands.flux.length}</span>
              </div>
              <div className="space-y-2">
                {entropyBands.flux.slice(0, 3).map(({ key, entropy }) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-gray-300 font-mono">{key.split(':')[0]}</span>
                    <span className="text-red-400">{entropy.toFixed(3)}</span>
                  </div>
                ))}
                {entropyBands.flux.length > 3 && (
                  <div className="text-xs text-gray-500">+{entropyBands.flux.length - 3} more...</div>
                )}
              </div>
            </div>

            {/* Stabilizing Band */}
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-full"></div>
                  <span className="font-medium">Stabilizing Band (0.3-0.8)</span>
                </div>
                <span className="text-orange-400 font-bold">{entropyBands.stabilizing.length}</span>
              </div>
              <div className="space-y-2">
                {entropyBands.stabilizing.slice(0, 3).map(({ key, entropy }) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-gray-300 font-mono">{key.split(':')[0]}</span>
                    <span className="text-orange-400">{entropy.toFixed(3)}</span>
                  </div>
                ))}
                {entropyBands.stabilizing.length > 3 && (
                  <div className="text-xs text-gray-500">+{entropyBands.stabilizing.length - 3} more...</div>
                )}
              </div>
            </div>

            {/* Converged Band */}
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-gradient-to-r from-teal-500 to-green-500 rounded-full"></div>
                  <span className="font-medium">Converged Band (0.0-0.3)</span>
                </div>
                <span className="text-green-400 font-bold">{entropyBands.converged.length}</span>
              </div>
              <div className="space-y-2">
                {entropyBands.converged.slice(0, 3).map(({ key, entropy }) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-gray-300 font-mono">{key.split(':')[0]}</span>
                    <span className="text-green-400">{entropy.toFixed(3)}</span>
                  </div>
                ))}
                {entropyBands.converged.length > 3 && (
                  <div className="text-xs text-gray-500">+{entropyBands.converged.length - 3} more...</div>
                )}
              </div>
            </div>
          </div>

          {/* Backwards Computation Ready */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white mb-4">Backwards Computation</h3>
            
            <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-600/30">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="w-5 h-5 text-purple-400" />
                <span className="font-medium text-purple-300">Integration Ready</span>
              </div>
              <div className="text-2xl font-bold text-purple-400 mb-2">
                {entropyBands.backwards_ready.length}
              </div>
              <div className="text-xs text-gray-400">
                Signals ready for backwards derivation
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4">
              <h4 className="font-medium mb-3">Mathematical Principle</h4>
              <div className="text-sm text-gray-300 space-y-2">
                <div>• <span className="text-cyan-300">Current Price + Metrics</span></div>
                <div>• <span className="text-purple-300">→ Unique Previous State</span></div>
                <div>• <span className="text-green-300">Only ONE path possible</span></div>
              </div>
              <div className="mt-3 text-xs font-mono text-yellow-400">
                ∫ backwards_derive(current + QBSA/QFH)
              </div>
            </div>
          </div>
        </div>
      )}

      {viewMode === 'evolution' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Signal Selection</h3>
              <div className="bg-gray-800 rounded-lg p-4 max-h-80 overflow-y-auto">
                {Array.from(signalEvolution.keys()).map(key => (
                  <div
                    key={key}
                    onClick={() => setSelectedSignal(key)}
                    className={`p-3 rounded cursor-pointer mb-2 transition-colors ${
                      selectedSignal === key 
                        ? 'bg-purple-600 text-white' 
                        : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                    }`}
                  >
                    <div className="font-mono text-sm">{key}</div>
                    <div className="text-xs opacity-75">
                      {signalEvolution.get(key)?.length || 0} evolution steps
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Evolution Timeline</h3>
              {selectedSignal && (
                <div className="bg-gray-800 rounded-lg p-4 max-h-80 overflow-y-auto">
                  {getSignalHistory(selectedSignal).map((step, index) => (
                    <div key={index} className="mb-4 p-3 bg-gray-700 rounded">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Step {step.step}</span>
                        <span className="text-xs text-gray-400">
                          {new Date(step.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <div className="text-gray-400">Entropy</div>
                          <div className="text-red-400">{step.current.entropy?.toFixed(3)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Stability</div>
                          <div className="text-orange-400">{step.current.stability?.toFixed(3)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Coherence</div>
                          <div className="text-green-400">{step.current.coherence?.toFixed(3)}</div>
                        </div>
                      </div>
                      {step.backwards_verified && (
                        <div className="mt-2 p-2 bg-purple-900/30 rounded text-xs">
                          <div className="text-purple-400 font-medium">✓ Backwards Verified</div>
                          <div className="text-purple-300">Confidence: {step.backwards_confidence?.toFixed(3)}</div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {viewMode === 'decisions' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white mb-4">Trading Decisions</h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {tradingDecisions.map((decision, index) => (
              <div key={index} className="bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Target className={`w-5 h-5 ${
                      decision.decision === 'TRADE' ? 'text-green-400' : 'text-yellow-400'
                    }`} />
                    <span className="font-medium">{decision.signal_key.split(':')[0]}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {decision.action === 'BUY' && <TrendingUp className="w-4 h-4 text-green-400" />}
                    {decision.action === 'SELL' && <TrendingDown className="w-4 h-4 text-red-400" />}
                    <span className={`text-sm font-medium ${
                      decision.action === 'BUY' ? 'text-green-400' : 
                      decision.action === 'SELL' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {decision.action}
                    </span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Decision:</span>
                    <span className={`font-medium ${
                      decision.decision === 'TRADE' ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      {decision.decision}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Confidence:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full bg-gradient-to-r ${
                            decision.confidence > 0.8 ? 'from-green-500 to-green-400' :
                            decision.confidence > 0.6 ? 'from-yellow-500 to-yellow-400' :
                            'from-red-500 to-red-400'
                          }`}
                          style={{ width: `${decision.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm">{(decision.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Entropy Band:</span>
                    <span className={`text-sm ${
                      decision.entropy_band === 'converged' ? 'text-green-400' :
                      decision.entropy_band === 'stabilizing' ? 'text-orange-400' :
                      'text-red-400'
                    }`}>
                      {decision.entropy_band}
                    </span>
                  </div>
                </div>
                
                <div className="mt-3 p-2 bg-gray-700 rounded text-xs text-gray-300">
                  {decision.reasoning}
                </div>
              </div>
            ))}
          </div>
          
          {tradingDecisions.length === 0 && (
            <div className="text-center py-8 text-gray-400">
              <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <div>No trading decisions available</div>
              <div className="text-sm">Waiting for signals to converge...</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default EntropyBandAnalysis;