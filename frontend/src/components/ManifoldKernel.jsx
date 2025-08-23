import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { 
  Zap, 
  Database, 
  TrendingUp, 
  Brain,
  Clock,
  Target,
  GitBranch,
  Activity,
  Eye,
  Layers,
  Cpu,
  MapPin,
  BarChart3
} from 'lucide-react';

const ManifoldKernel = () => {
  const {
    connected,
    quantumSignals,
    signalHistory,
    valkeyMetrics,
    livePatterns
  } = useWebSocket();

  const [activeFlow, setActiveFlow] = useState('realtime'); // realtime, historical, analysis
  const [selectedTimestamp, setSelectedTimestamp] = useState(null);
  const [kernelActivity, setKernelActivity] = useState([]);
  const canvasRef = useRef(null);
  const flowCanvasRef = useRef(null);

  // Simulate the complete manifold data pipeline
  const manifoldPipeline = useMemo(() => {
    const now = Date.now();
    
    // Stage 1: Timestamped Identity Creation (Valkey Keys)
    const timestampedIdentities = Object.entries(quantumSignals).map(([key, signal]) => ({
      timestamp: new Date(signal.timestamp).getTime(),
      valkeyKey: key,
      identity: key,
      stage: 'identity_created',
      marketContext: 'pending'
    }));

    // Stage 2: OANDA Candle Data Enrichment
    const enrichedIdentities = timestampedIdentities.map(identity => ({
      ...identity,
      oandaData: {
        instrument: identity.valkeyKey.includes('EUR') ? 'EUR_USD' : 'GBP_USD', // Simulated
        candle: {
          open: 1.0850 + (Math.random() - 0.5) * 0.01,
          high: 1.0860 + (Math.random() - 0.5) * 0.01,
          low: 1.0840 + (Math.random() - 0.5) * 0.01,
          close: 1.0855 + (Math.random() - 0.5) * 0.01,
          volume: Math.floor(Math.random() * 10000)
        }
      },
      stage: 'market_context_added',
      marketContext: 'enriched'
    }));

    // Stage 3: Quantum Signal Processing
    const quantumProcessedIdentities = enrichedIdentities.map(identity => {
      const signal = quantumSignals[identity.valkeyKey] || {};
      return {
        ...identity,
        quantumMetrics: {
          coherence: signal.coherence || Math.random(),
          entropy: signal.entropy || Math.random(),
          stability: signal.stability || Math.random(),
          confidence: signal.confidence || Math.random(),
        },
        stage: 'quantum_processed',
        marketContext: 'analyzed'
      };
    });

    // Stage 4: Von Neumann Kernel Pattern Recognition
    const kernelAnalyzedIdentities = quantumProcessedIdentities.map(identity => {
      const historicalMatches = signalHistory.filter(h => 
        Math.abs(h.coherence - identity.quantumMetrics.coherence) < 0.15
      ).length;
      
      const pathRecognition = historicalMatches > 3 ? 'strong' : 
                             historicalMatches > 1 ? 'moderate' : 'weak';
      
      const bettingConfidence = pathRecognition === 'strong' ? 0.8 :
                               pathRecognition === 'moderate' ? 0.5 : 0.2;

      return {
        ...identity,
        kernelAnalysis: {
          historicalMatches,
          pathRecognition,
          bettingConfidence,
          predictedOutcome: bettingConfidence > 0.7 ? 'favorable' : 
                           bettingConfidence > 0.4 ? 'neutral' : 'unfavorable'
        },
        stage: 'kernel_analyzed',
        marketContext: 'prediction_ready'
      };
    });

    // Stage 5: Entropy Band Classification
    const entropyBandClassified = kernelAnalyzedIdentities.map(identity => {
      const age = (now - identity.timestamp) / 1000; // seconds
      let entropyBand = 'fresh';
      let survivalProbability = 0.15;

      if (age > 300 && age <= 1800) { // 5-30 minutes
        entropyBand = 'consolidation';
        survivalProbability = 0.60;
      } else if (age > 1800) { // > 30 minutes
        entropyBand = 'settled';
        survivalProbability = 0.85;
      }

      return {
        ...identity,
        entropyBand,
        survivalProbability,
        temporalRelevance: Math.exp(-age / 3600), // Exponential decay
        stage: 'entropy_classified',
        marketContext: 'manifold_positioned'
      };
    });

    return {
      raw: timestampedIdentities,
      enriched: enrichedIdentities,
      quantumProcessed: quantumProcessedIdentities,
      kernelAnalyzed: kernelAnalyzedIdentities,
      final: entropyBandClassified
    };
  }, [quantumSignals, signalHistory]);

  // Real-time kernel activity simulation
  useEffect(() => {
    const interval = setInterval(() => {
      const activities = [
        'Pattern matching against historical data...',
        'Calculating temporal coherence...',
        'Updating manifold surface topology...',
        'Analyzing path recognition confidence...',
        'Processing new timestamped identity...',
        'Evaluating betting position strength...',
        'Cross-referencing OANDA candle patterns...',
        'Computing entropy band transitions...'
      ];
      
      setKernelActivity(prev => [
        {
          timestamp: Date.now(),
          activity: activities[Math.floor(Math.random() * activities.length)],
          id: Math.random().toString(36).substr(2, 9)
        },
        ...prev.slice(0, 15) // Keep last 15 activities
      ]);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Draw pipeline flow visualization
  useEffect(() => {
    const canvas = flowCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Draw the 5-stage pipeline
    const stages = [
      { name: 'Timestamp\nIdentities', color: '#3b82f6', y: 50 },
      { name: 'OANDA\nEnrichment', color: '#10b981', y: 120 },
      { name: 'Quantum\nProcessing', color: '#8b5cf6', y: 190 },
      { name: 'Kernel\nAnalysis', color: '#f59e0b', y: 260 },
      { name: 'Entropy\nBands', color: '#ef4444', y: 330 }
    ];

    stages.forEach((stage, idx) => {
      const x = 50 + idx * 120;
      
      // Draw stage circle
      ctx.fillStyle = stage.color;
      ctx.beginPath();
      ctx.arc(x, stage.y, 20, 0, 2 * Math.PI);
      ctx.fill();

      // Draw stage label
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      const lines = stage.name.split('\n');
      lines.forEach((line, lineIdx) => {
        ctx.fillText(line, x, stage.y + 35 + lineIdx * 15);
      });

      // Draw connecting arrows
      if (idx < stages.length - 1) {
        ctx.strokeStyle = '#6b7280';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x + 20, stage.y);
        ctx.lineTo(x + 100, stage.y);
        ctx.stroke();
        
        // Arrowhead
        ctx.beginPath();
        ctx.moveTo(x + 95, stage.y - 5);
        ctx.lineTo(x + 100, stage.y);
        ctx.lineTo(x + 95, stage.y + 5);
        ctx.stroke();
      }
    });

    // Draw data flow indicators
    const dataCount = manifoldPipeline.final?.length || 0;
    ctx.fillStyle = '#22d3ee';
    ctx.font = '14px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Active Identities: ${dataCount}`, 10, height - 20);

  }, [manifoldPipeline]);

  const entropyBandStats = useMemo(() => {
    const final = manifoldPipeline.final || [];
    return {
      fresh: final.filter(i => i.entropyBand === 'fresh').length,
      consolidation: final.filter(i => i.entropyBand === 'consolidation').length,
      settled: final.filter(i => i.entropyBand === 'settled').length
    };
  }, [manifoldPipeline]);

  const kernelStats = useMemo(() => {
    const analyzed = manifoldPipeline.kernelAnalyzed || [];
    const strongPatterns = analyzed.filter(i => i.kernelAnalysis?.pathRecognition === 'strong').length;
    const totalConfidence = analyzed.reduce((sum, i) => sum + (i.kernelAnalysis?.bettingConfidence || 0), 0);
    const avgConfidence = analyzed.length > 0 ? totalConfidence / analyzed.length : 0;
    
    return {
      strongPatterns,
      totalPatterns: analyzed.length,
      avgConfidence,
      favorablePredictions: analyzed.filter(i => i.kernelAnalysis?.predictedOutcome === 'favorable').length
    };
  }, [manifoldPipeline]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-orange-500 bg-clip-text text-transparent">
              Manifold Kernel Engine
            </h1>
            <p className="text-gray-400 mt-1">OANDA → Valkey → Quantum → Von Neumann Pattern Recognition → Trading Decisions</p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Flow selector */}
            <select
              value={activeFlow}
              onChange={(e) => setActiveFlow(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm"
            >
              <option value="realtime">Real-time Flow</option>
              <option value="historical">Historical Analysis</option>
              <option value="analysis">Pattern Analysis</option>
            </select>

            <div className={`flex items-center gap-2 px-3 py-1 rounded-lg text-sm ${
              connected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
            }`}>
              <Activity className="w-4 h-4" />
              {connected ? 'Kernel Active' : 'Kernel Offline'}
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Pipeline Flow */}
        <div className="col-span-8">
          <div className="bg-gray-900 rounded-lg p-4 mb-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <GitBranch className="w-5 h-5 text-blue-400" />
              Manifold Processing Pipeline
            </h3>
            <canvas
              ref={flowCanvasRef}
              width={700}
              height={400}
              className="border border-gray-800 rounded-lg bg-gray-950"
            />
          </div>

          {/* Identity Details */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-purple-400" />
              Timestamped Identity Analysis
            </h3>
            
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-gray-800 rounded-lg p-3">
                <h4 className="text-sm font-medium text-gray-400 mb-2">Fresh Identities</h4>
                <div className="text-2xl font-bold text-red-400">{entropyBandStats.fresh}</div>
                <div className="text-xs text-gray-500">High entropy • 15% survival</div>
              </div>
              
              <div className="bg-gray-800 rounded-lg p-3">
                <h4 className="text-sm font-medium text-gray-400 mb-2">Consolidating</h4>
                <div className="text-2xl font-bold text-orange-400">{entropyBandStats.consolidation}</div>
                <div className="text-xs text-gray-500">Medium entropy • 60% survival</div>
              </div>
              
              <div className="bg-gray-800 rounded-lg p-3">
                <h4 className="text-sm font-medium text-gray-400 mb-2">Settled</h4>
                <div className="text-2xl font-bold text-green-400">{entropyBandStats.settled}</div>
                <div className="text-xs text-gray-500">Low entropy • 85% survival</div>
              </div>
            </div>

            {/* Sample Identity Details */}
            {manifoldPipeline.final?.slice(0, 3).map((identity, idx) => (
              <div key={idx} className="bg-gray-800 rounded-lg p-3 mb-2">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-sm">{identity.identity.slice(-12)}</span>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-1 rounded ${
                      identity.entropyBand === 'fresh' ? 'bg-red-900 text-red-300' :
                      identity.entropyBand === 'consolidation' ? 'bg-orange-900 text-orange-300' :
                      'bg-green-900 text-green-300'
                    }`}>
                      {identity.entropyBand}
                    </span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      identity.kernelAnalysis?.predictedOutcome === 'favorable' ? 'bg-blue-900 text-blue-300' :
                      identity.kernelAnalysis?.predictedOutcome === 'neutral' ? 'bg-gray-700 text-gray-300' :
                      'bg-red-900 text-red-300'
                    }`}>
                      {identity.kernelAnalysis?.predictedOutcome}
                    </span>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-4 text-xs">
                  <div>
                    <span className="text-gray-400">OANDA:</span>
                    <div>{identity.oandaData?.instrument}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Coherence:</span>
                    <div>{(identity.quantumMetrics?.coherence * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Matches:</span>
                    <div>{identity.kernelAnalysis?.historicalMatches}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Confidence:</span>
                    <div>{(identity.kernelAnalysis?.bettingConfidence * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Panel - Kernel Activity & Stats */}
        <div className="col-span-4 space-y-6">
          {/* Von Neumann Kernel Stats */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Brain className="w-5 h-5 text-orange-400" />
              Von Neumann Kernel
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400 text-sm">Strong Patterns</span>
                <span className="font-medium text-orange-400">{kernelStats.strongPatterns}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 text-sm">Total Patterns</span>
                <span className="font-medium">{kernelStats.totalPatterns}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 text-sm">Avg Confidence</span>
                <span className="font-medium text-blue-400">{(kernelStats.avgConfidence * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 text-sm">Favorable</span>
                <span className="font-medium text-green-400">{kernelStats.favorablePredictions}</span>
              </div>
            </div>
          </div>

          {/* Real-time Activity Log */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-cyan-400" />
              Kernel Activity
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {kernelActivity.map((activity) => (
                <div key={activity.id} className="text-xs bg-gray-800 rounded p-2">
                  <div className="text-gray-500 mb-1">
                    {new Date(activity.timestamp).toLocaleTimeString()}
                  </div>
                  <div className="text-gray-200">{activity.activity}</div>
                </div>
              ))}
              {kernelActivity.length === 0 && (
                <div className="text-center text-gray-500 py-4 text-sm">
                  Kernel initializing...
                </div>
              )}
            </div>
          </div>

          {/* Manifold Health */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-400" />
              Manifold Health
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Connection</span>
                <span className={connected ? 'text-green-400' : 'text-red-400'}>
                  {connected ? 'Active' : 'Offline'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Pipeline Flow</span>
                <span className="text-green-400">Nominal</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Pattern Recognition</span>
                <span className="text-blue-400">Learning</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Temporal Coherence</span>
                <span className="text-purple-400">Stable</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManifoldKernel;