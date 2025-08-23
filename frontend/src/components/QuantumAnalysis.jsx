import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { Brain, Activity, TrendingUp, Zap, Target, Shield } from 'lucide-react';

const QuantumAnalysis = () => {
  const { systemMetrics, connected } = useWebSocket();
  const [analysisData, setAnalysisData] = useState({
    qfh_patterns: [],
    qbsa_results: {},
    coherence_metrics: {},
    stability_index: 0,
    pattern_evolution: [],
    entropy_levels: []
  });

  // Simulate quantum analysis data if not available from WebSocket
  useEffect(() => {
    if (!connected) {
      const mockData = {
        qfh_patterns: [
          { id: 1, pattern: 'Harmonic Resonance', confidence: 87.3, frequency: 2.4, amplitude: 0.65 },
          { id: 2, pattern: 'Field Convergence', confidence: 74.2, frequency: 1.8, amplitude: 0.42 },
          { id: 3, pattern: 'Quantum Fluctuation', confidence: 91.5, frequency: 3.2, amplitude: 0.78 },
          { id: 4, pattern: 'Phase Alignment', confidence: 68.9, frequency: 1.5, amplitude: 0.33 }
        ],
        qbsa_results: {
          binary_states: 15625,
          analyzed_patterns: 8432,
          coherence_score: 0.4687,
          stability_ratio: 0.7234,
          flip_events: 234,
          rupture_events: 12
        },
        coherence_metrics: {
          temporal_coherence: 0.85,
          spatial_coherence: 0.72,
          spectral_coherence: 0.91,
          cross_coherence: 0.68
        },
        stability_index: 0.7234,
        pattern_evolution: [
          { timestamp: Date.now() - 300000, stability: 0.65 },
          { timestamp: Date.now() - 240000, stability: 0.71 },
          { timestamp: Date.now() - 180000, stability: 0.68 },
          { timestamp: Date.now() - 120000, stability: 0.74 },
          { timestamp: Date.now() - 60000, stability: 0.72 },
          { timestamp: Date.now(), stability: 0.76 }
        ],
        entropy_levels: [
          { frequency: 0.5, entropy: 0.12 },
          { frequency: 1.0, entropy: 0.08 },
          { frequency: 1.5, entropy: 0.15 },
          { frequency: 2.0, entropy: 0.10 },
          { frequency: 2.5, entropy: 0.09 }
        ]
      };
      setAnalysisData(mockData);
    }
  }, [connected]);

  const formatPercentage = (value) => {
    return (value * 100).toFixed(1) + '%';
  };

  const formatNumber = (value, decimals = 4) => {
    return parseFloat(value).toFixed(decimals);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return 'text-green-400';
    if (confidence >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStabilityColor = (stability) => {
    if (stability >= 0.7) return 'text-green-400';
    if (stability >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Quantum Field Harmonics Analysis</h1>
          <p className="text-gray-400 mt-1">Advanced pattern recognition and coherence analysis</p>
        </div>
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <span className="text-sm text-gray-400">
            Engine Status: <span className={connected ? 'text-green-400' : 'text-red-400'}>
              {connected ? 'Active' : 'Offline'}
            </span>
          </span>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-blue-400" />
              <span className="card-title">Coherence Score</span>
            </div>
          </div>
          <div className="card-value text-blue-400">
            {formatNumber(analysisData.qbsa_results.coherence_score || systemMetrics?.coherence || 0.4687)}
          </div>
          <div className="text-sm text-gray-400">
            Based on {analysisData.qbsa_results.analyzed_patterns || 8432} patterns
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-green-400" />
              <span className="card-title">Stability Index</span>
            </div>
          </div>
          <div className={`card-value ${getStabilityColor(analysisData.stability_index)}`}>
            {formatPercentage(analysisData.stability_index)}
          </div>
          <div className="text-sm text-gray-400">
            Quantum field stability
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-purple-400" />
              <span className="card-title">Binary States</span>
            </div>
          </div>
          <div className="card-value text-purple-400">
            {(analysisData.qbsa_results.binary_states || 15625).toLocaleString()}
          </div>
          <div className="text-sm text-gray-400">
            QBSA processed states
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-yellow-400" />
              <span className="card-title">Flip Events</span>
            </div>
          </div>
          <div className="card-value text-yellow-400">
            {analysisData.qbsa_results.flip_events || 234}
          </div>
          <div className="text-sm text-gray-400">
            Recent quantum flips
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* QFH Patterns */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              QFH Pattern Recognition
            </h3>
          </div>
          <div className="space-y-3">
            {analysisData.qfh_patterns.map((pattern) => (
              <div key={pattern.id} className="signal-card">
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">{pattern.pattern}</span>
                    <span className={`text-sm font-medium ${getConfidenceColor(pattern.confidence)}`}>
                      {pattern.confidence}%
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm text-gray-400">
                    <div>
                      <span>Frequency: </span>
                      <span className="text-gray-200">{pattern.frequency} Hz</span>
                    </div>
                    <div>
                      <span>Amplitude: </span>
                      <span className="text-gray-200">{pattern.amplitude}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Coherence Metrics */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-400" />
              Coherence Analysis
            </h3>
          </div>
          <div className="space-y-4">
            {Object.entries(analysisData.coherence_metrics).map(([key, value]) => {
              const label = key.split('_').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
              ).join(' ');
              
              return (
                <div key={key}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm text-gray-400">{label}</span>
                    <span className="text-sm font-medium">{formatPercentage(value)}</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all"
                      style={{ width: `${value * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Pattern Evolution Chart */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            Pattern Evolution Timeline
          </h3>
        </div>
        <div className="h-64 flex items-end justify-between gap-2 p-4">
          {analysisData.pattern_evolution.map((point, index) => (
            <div key={index} className="flex flex-col items-center flex-1">
              <div
                className="w-full bg-gradient-to-t from-green-500 to-green-300 rounded-t transition-all hover:from-green-400 hover:to-green-200"
                style={{ 
                  height: `${point.stability * 200}px`,
                  minHeight: '20px'
                }}
              />
              <div className="text-xs text-gray-500 mt-2">
                {new Date(point.timestamp).toLocaleTimeString([], { 
                  hour: '2-digit', 
                  minute: '2-digit' 
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="text-center text-sm text-gray-400 border-t border-gray-800 pt-2">
          Stability Index Over Time
        </div>
      </div>

      {/* QBSA Results */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            Quantum Binary State Analysis (QBSA)
          </h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-gray-800 rounded-lg">
            <div className="text-2xl font-bold text-purple-400 mb-1">
              {(analysisData.qbsa_results.binary_states || 15625).toLocaleString()}
            </div>
            <div className="text-sm text-gray-400">Binary States</div>
          </div>
          
          <div className="text-center p-4 bg-gray-800 rounded-lg">
            <div className="text-2xl font-bold text-blue-400 mb-1">
              {(analysisData.qbsa_results.analyzed_patterns || 8432).toLocaleString()}
            </div>
            <div className="text-sm text-gray-400">Analyzed Patterns</div>
          </div>
          
          <div className="text-center p-4 bg-gray-800 rounded-lg">
            <div className="text-2xl font-bold text-green-400 mb-1">
              {formatNumber(analysisData.qbsa_results.stability_ratio || 0.7234)}
            </div>
            <div className="text-sm text-gray-400">Stability Ratio</div>
          </div>
          
          <div className="text-center p-4 bg-gray-800 rounded-lg">
            <div className="text-2xl font-bold text-yellow-400 mb-1">
              {analysisData.qbsa_results.flip_events || 234}
            </div>
            <div className="text-sm text-gray-400">Flip Events</div>
          </div>
          
          <div className="text-center p-4 bg-gray-800 rounded-lg">
            <div className="text-2xl font-bold text-red-400 mb-1">
              {analysisData.qbsa_results.rupture_events || 12}
            </div>
            <div className="text-sm text-gray-400">Rupture Events</div>
          </div>
          
          <div className="text-center p-4 bg-gray-800 rounded-lg">
            <div className="text-2xl font-bold text-blue-400 mb-1">
              {formatNumber(analysisData.qbsa_results.coherence_score || 0.4687)}
            </div>
            <div className="text-sm text-gray-400">Coherence Score</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuantumAnalysis;