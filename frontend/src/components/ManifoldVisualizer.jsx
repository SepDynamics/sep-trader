import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { 
  Brain, 
  Target, 
  Layers, 
  GitBranch, 
  Zap,
  Clock,
  TrendingUp,
  Activity,
  Database,
  Cpu,
  Eye
} from 'lucide-react';

const ManifoldVisualizer = () => {
  const {
    connected,
    quantumSignals,
    signalHistory,
    valkeyMetrics,
    livePatterns
  } = useWebSocket();

  const [selectedTimeHorizon, setSelectedTimeHorizon] = useState('1h');
  const [kernelMode, setKernelMode] = useState('pattern_matching'); // pattern_matching, path_prediction, entropy_analysis
  const canvasRef = useRef(null);

  // Simulate manifold data structure - temporal bands with entropy gradients
  const manifoldData = useMemo(() => {
    const now = Date.now();
    const bands = {
      // Fresh outer band - high entropy, most signals die immediately
      fresh: {
        timeRange: [now - 300000, now], // Last 5 minutes
        entropy: 0.85,
        survival_rate: 0.15,
        signals: Object.values(quantumSignals).filter(s => 
          new Date(s.timestamp).getTime() > now - 300000
        ).length,
        color: '#ff4444'
      },
      // Middle band - consolidation zone
      consolidation: {
        timeRange: [now - 1800000, now - 300000], // 5-30 minutes ago
        entropy: 0.45,
        survival_rate: 0.60,
        signals: Object.values(quantumSignals).filter(s => {
          const t = new Date(s.timestamp).getTime();
          return t > now - 1800000 && t <= now - 300000;
        }).length,
        color: '#ffaa44'
      },
      // Inner band - settled, long-tail patterns
      settled: {
        timeRange: [now - 7200000, now - 1800000], // 30min - 2 hours ago
        entropy: 0.12,
        survival_rate: 0.85,
        signals: Object.values(quantumSignals).filter(s => {
          const t = new Date(s.timestamp).getTime();
          return t > now - 7200000 && t <= now - 1800000;
        }).length,
        color: '#44ff44'
      }
    };
    return bands;
  }, [quantumSignals]);

  // Von Neumann kernel pattern matching simulation
  const kernelAnalysis = useMemo(() => {
    const patterns = Object.values(livePatterns);
    const historicalMatches = signalHistory.slice(0, 100); // Recent history for pattern matching
    
    const pathAnalysis = patterns.map(pattern => {
      // Simulate path recognition against historical data
      const similarPaths = historicalMatches.filter(h => 
        Math.abs(h.coherence - pattern.coherence) < 0.1 &&
        Math.abs(h.entropy - (pattern.entropy || 0.5)) < 0.15
      );

      const matchConfidence = similarPaths.length > 0 ? 
        similarPaths.reduce((sum, p) => sum + (p.confidence || 0.5), 0) / similarPaths.length :
        0.1;

      return {
        patternId: pattern.id || `pattern_${Date.now()}_${pattern.valkeyKey || 'unknown'}`,
        currentPath: {
          coherence: pattern.coherence,
          entropy: pattern.entropy || 0.5,
          stability: pattern.stability || 0.6,
          timestamp: pattern.lastUpdate
        },
        historicalMatches: similarPaths.length,
        matchConfidence,
        predictedOutcome: matchConfidence > 0.7 ? 'favorable' : matchConfidence > 0.4 ? 'neutral' : 'unfavorable',
        bettingWeight: matchConfidence * (pattern.strength || 0.5)
      };
    });

    return {
      totalPatterns: patterns.length,
      activeMatches: pathAnalysis.filter(p => p.historicalMatches > 0).length,
      avgMatchConfidence: pathAnalysis.length > 0 ? 
        pathAnalysis.reduce((sum, p) => sum + p.matchConfidence, 0) / pathAnalysis.length : 0,
      strongPatterns: pathAnalysis.filter(p => p.matchConfidence > 0.7).length,
      pathAnalysis
    };
  }, [livePatterns, signalHistory]);

  // Contextual relevance calculation for timestamped keys
  const contextualRelevance = useMemo(() => {
    const now = Date.now();
    const allTimestamps = Object.values(quantumSignals).map(s => new Date(s.timestamp).getTime());
    
    if (allTimestamps.length === 0) return { avgRelevance: 0, temporalSpread: 0 };

    const relevanceScores = allTimestamps.map(timestamp => {
      const age = (now - timestamp) / 1000; // seconds
      const marketRelevance = Math.exp(-age / 3600); // Exponential decay with 1-hour half-life
      const temporalCoherence = 1 / (1 + Math.abs(age - 1800)); // Peak relevance at 30 minutes
      return marketRelevance * temporalCoherence;
    });

    return {
      avgRelevance: relevanceScores.reduce((sum, r) => sum + r, 0) / relevanceScores.length,
      temporalSpread: Math.max(...allTimestamps) - Math.min(...allTimestamps),
      keyDensity: allTimestamps.length / ((Math.max(...allTimestamps) - Math.min(...allTimestamps)) / 1000 || 1)
    };
  }, [quantumSignals]);

  // Draw manifold visualization on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Draw entropy bands as concentric regions
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) * 0.4;

    Object.entries(manifoldData).forEach(([bandName, band], index) => {
      const radius = maxRadius * (1 - index * 0.25);
      const gradient = ctx.createRadialGradient(centerX, centerY, radius * 0.7, centerX, centerY, radius);
      gradient.addColorStop(0, band.color + '40');
      gradient.addColorStop(1, band.color + '10');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.fill();

      // Add band labels
      ctx.fillStyle = band.color;
      ctx.font = '12px monospace';
      ctx.fillText(`${bandName.toUpperCase()}: ${band.signals} signals`, 10, 30 + index * 20);
      ctx.fillText(`Entropy: ${band.entropy.toFixed(2)} | Survival: ${(band.survival_rate * 100).toFixed(0)}%`, 10, 45 + index * 20);
    });

    // Draw pattern paths as connecting lines
    if (kernelAnalysis.pathAnalysis.length > 0) {
      kernelAnalysis.pathAnalysis.forEach((pattern, index) => {
        const angle = (index / kernelAnalysis.pathAnalysis.length) * 2 * Math.PI;
        const startRadius = maxRadius * 0.3;
        const endRadius = maxRadius * (0.7 + pattern.matchConfidence * 0.3);
        
        const startX = centerX + startRadius * Math.cos(angle);
        const startY = centerY + startRadius * Math.sin(angle);
        const endX = centerX + endRadius * Math.cos(angle);
        const endY = centerY + endRadius * Math.sin(angle);

        ctx.strokeStyle = pattern.predictedOutcome === 'favorable' ? '#44ff44' : 
                         pattern.predictedOutcome === 'neutral' ? '#ffaa44' : '#ff4444';
        ctx.lineWidth = Math.max(1, pattern.matchConfidence * 3);
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Add pattern dots
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.arc(endX, endY, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

  }, [manifoldData, kernelAnalysis]);

  const timeHorizons = ['5m', '15m', '1h', '4h', '1d'];
  const kernelModes = [
    { value: 'pattern_matching', label: 'Pattern Matching', icon: Brain },
    { value: 'path_prediction', label: 'Path Prediction', icon: TrendingUp },
    { value: 'entropy_analysis', label: 'Entropy Analysis', icon: Zap }
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-transparent">
              Manifold Visualizer
            </h1>
            <p className="text-gray-400 mt-1">Von Neumann Kernel • Phase Space Mapping • Temporal Manifold Analysis</p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Kernel Mode Selector */}
            <select
              value={kernelMode}
              onChange={(e) => setKernelMode(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm"
            >
              {kernelModes.map(mode => (
                <option key={mode.value} value={mode.value}>{mode.label}</option>
              ))}
            </select>

            {/* Time Horizon */}
            <div className="flex items-center gap-2">
              {timeHorizons.map(horizon => (
                <button
                  key={horizon}
                  onClick={() => setSelectedTimeHorizon(horizon)}
                  className={`px-3 py-1 text-sm rounded transition-colors ${
                    selectedTimeHorizon === horizon
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                  }`}
                >
                  {horizon}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Manifold Metrics */}
        <div className="col-span-3 space-y-4">
          {/* Entropy Band Analysis */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Layers className="w-5 h-5 text-purple-400" />
              Entropy Bands
            </h3>
            <div className="space-y-3">
              {Object.entries(manifoldData).map(([bandName, band]) => (
                <div key={bandName} className="bg-gray-800 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium capitalize">{bandName}</span>
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: band.color }}
                    />
                  </div>
                  <div className="text-xs space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Signals:</span>
                      <span>{band.signals}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Entropy:</span>
                      <span>{band.entropy.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Survival:</span>
                      <span>{(band.survival_rate * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Von Neumann Kernel Stats */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Brain className="w-5 h-5 text-blue-400" />
              Kernel Analysis
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Active Patterns</span>
                <span className="font-medium">{kernelAnalysis.totalPatterns}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Historical Matches</span>
                <span className="font-medium">{kernelAnalysis.activeMatches}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Match Confidence</span>
                <span className="font-medium text-green-400">
                  {(kernelAnalysis.avgMatchConfidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Strong Patterns</span>
                <span className="font-medium text-blue-400">{kernelAnalysis.strongPatterns}</span>
              </div>
            </div>
          </div>

          {/* Contextual Relevance */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Clock className="w-5 h-5 text-yellow-400" />
              Temporal Context
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Avg Relevance</span>
                <span className="font-medium">
                  {(contextualRelevance.avgRelevance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Temporal Spread</span>
                <span className="font-medium text-gray-200">
                  {(contextualRelevance.temporalSpread / 1000 / 60).toFixed(1)}m
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Key Density</span>
                <span className="font-medium text-purple-400">
                  {contextualRelevance.keyDensity.toFixed(2)}/s
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Center Panel - Manifold Canvas */}
        <div className="col-span-6">
          <div className="bg-gray-900 rounded-lg p-4 h-full">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-purple-400" />
              Database Manifold Surface
            </h3>
            <div className="relative">
              <canvas
                ref={canvasRef}
                width={600}
                height={500}
                className="border border-gray-800 rounded-lg bg-gray-950"
              />
              <div className="absolute bottom-4 left-4 text-xs text-gray-500 space-y-1">
                <div>• Fresh Band (Red): High entropy, immediate decay</div>
                <div>• Consolidation (Orange): Pattern formation</div>
                <div>• Settled (Green): Stable, long-tail patterns</div>
                <div>• Lines: Von Neumann kernel path predictions</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Pattern History */}
        <div className="col-span-3 space-y-4">
          {/* Pattern Matching Display */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <GitBranch className="w-5 h-5 text-green-400" />
              Path Recognition
            </h3>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {kernelAnalysis.pathAnalysis.slice(0, 5).map((pattern, idx) => (
                <div key={idx} className="bg-gray-800 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Path {pattern.patternId.slice(-4)}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      pattern.predictedOutcome === 'favorable' ? 'bg-green-900 text-green-300' :
                      pattern.predictedOutcome === 'neutral' ? 'bg-yellow-900 text-yellow-300' :
                      'bg-red-900 text-red-300'
                    }`}>
                      {pattern.predictedOutcome}
                    </span>
                  </div>
                  <div className="text-xs space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Matches:</span>
                      <span>{pattern.historicalMatches}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Confidence:</span>
                      <span>{(pattern.matchConfidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Weight:</span>
                      <span>{pattern.bettingWeight.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              ))}
              {kernelAnalysis.pathAnalysis.length === 0 && (
                <div className="text-center text-gray-500 py-4 text-sm">
                  No active path recognition
                </div>
              )}
            </div>
          </div>

          {/* System Status */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Database className="w-5 h-5 text-cyan-400" />
              Manifold Status
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Valkey Connection</span>
                <span className={connected ? 'text-green-400' : 'text-red-400'}>
                  {connected ? 'Active' : 'Offline'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Key Population</span>
                <span className="text-gray-200">{Object.keys(quantumSignals).length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Pattern Space</span>
                <span className="text-gray-200">{Object.keys(livePatterns).length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Kernel Mode</span>
                <span className="text-purple-400">{kernelMode.replace('_', ' ')}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManifoldVisualizer;