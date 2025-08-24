import React, { useState, useEffect, useMemo } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { 
  Database, 
  Zap, 
  Clock, 
  TrendingUp, 
  Activity,
  Server,
  BarChart3,
  GitBranch,
  HardDrive,
  Cpu,
  Eye,
  AlertTriangle,
  CheckCircle,
  Gauge
} from 'lucide-react';

const ValkeyPipelineManager = () => {
  const {
    connected,
    quantumSignals,
    valkeyMetrics,
    livePatterns,
    sendMessage
  } = useWebSocket();

  const [pipelineStatus, setPipelineStatus] = useState('initializing');
  const [oandaFeedRate, setOandaFeedRate] = useState(0);
  const [valkeyKeyCount, setValkeyKeyCount] = useState(0);
  const [serverMetrics, setServerMetrics] = useState({
    memory: 0,
    cpu: 0,
    diskUsage: 0,
    networkIO: 0
  });

  // Simulate real-time pipeline monitoring
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate OANDA feed rate (keys/second)
      const newFeedRate = Math.max(0, oandaFeedRate + (Math.random() - 0.5) * 2);
      setOandaFeedRate(newFeedRate);
      
      // Simulate Valkey key growth
      setValkeyKeyCount(prev => prev + Math.floor(Math.random() * 5));

      // Simulate server metrics
      const newServerMetrics = {
        memory: 45 + Math.random() * 20, // 45-65%
        cpu: 25 + Math.random() * 30,    // 25-55%
        diskUsage: 60 + Math.random() * 15, // 60-75%
        networkIO: Math.random() * 100   // 0-100 MB/s
      };
      setServerMetrics(newServerMetrics);

      // Send feed rate data through WebSocket
      if (connected && sendMessage) {
        sendMessage({
          type: 'valkey_metrics',
          data: {
            feedRate: newFeedRate,
            keyCount: valkeyKeyCount + Math.floor(Math.random() * 5),
            serverMetrics: newServerMetrics
          }
        });
      }

      // Update pipeline status based on conditions
      if (connected && newFeedRate > 1) {
        setPipelineStatus('active');
      } else if (connected) {
        setPipelineStatus('waiting');
      } else {
        setPipelineStatus('offline');
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [connected, oandaFeedRate, valkeyKeyCount, sendMessage]);

  // Analyze current Valkey key distribution by age (entropy bands)
  const valkeyDistribution = useMemo(() => {
    const now = Date.now();
    const keys = Object.entries(quantumSignals);
    
    const fresh = keys.filter(([_, signal]) => 
      now - new Date(signal.timestamp).getTime() < 300000 // 5 minutes
    );
    
    const consolidation = keys.filter(([_, signal]) => {
      const age = now - new Date(signal.timestamp).getTime();
      return age >= 300000 && age < 1800000; // 5-30 minutes
    });
    
    const settled = keys.filter(([_, signal]) => 
      now - new Date(signal.timestamp).getTime() >= 1800000 // 30+ minutes
    );

    return {
      fresh: fresh.length,
      consolidation: consolidation.length,
      settled: settled.length,
      total: keys.length,
      oldestKey: keys.length > 0 ? Math.max(...keys.map(([_, s]) => now - new Date(s.timestamp).getTime())) : 0
    };
  }, [quantumSignals]);

  // Key naming pattern analysis
  const keyPatterns = useMemo(() => {
    const keys = Object.keys(quantumSignals);
    const instruments = {};
    const timePatterns = {};
    
    keys.forEach(key => {
      // Extract instrument from key pattern
      const instrumentMatch = key.match(/(EUR_USD|GBP_USD|USD_JPY|AUD_USD|USD_CHF|USD_CAD|NZD_USD)/);
      if (instrumentMatch) {
        const instrument = instrumentMatch[1];
        instruments[instrument] = (instruments[instrument] || 0) + 1;
      }
      
      // Extract time pattern
      const timestampMatch = key.match(/(\d{4}-\d{2}-\d{2})/);
      if (timestampMatch) {
        const date = timestampMatch[1];
        timePatterns[date] = (timePatterns[date] || 0) + 1;
      }
    });

    return { instruments, timePatterns };
  }, [quantumSignals]);

  // Valkey size management calculations
  const sizeManagement = useMemo(() => {
    const avgKeySize = 1.5; // KB per key (estimated)
    const currentSize = valkeyDistribution.total * avgKeySize;
    const projectedGrowth = oandaFeedRate * 60 * 60 * avgKeySize; // KB/hour
    const maxCapacity = 2000000; // 2GB limit
    const utilizationPercent = (currentSize / maxCapacity) * 100;
    
    // Calculate retention strategy
    const retentionTiers = {
      fresh: { maxAge: 5, retention: 100 }, // 5 minutes, keep all
      consolidation: { maxAge: 30, retention: 70 }, // 30 minutes, keep 70%
      settled: { maxAge: 720, retention: 20 } // 12 hours, keep 20%
    };

    return {
      currentSize,
      projectedGrowth,
      utilizationPercent,
      maxCapacity,
      retentionTiers,
      timeToCapacity: projectedGrowth > 0 ? (maxCapacity - currentSize) / projectedGrowth : Infinity
    };
  }, [valkeyDistribution, oandaFeedRate]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'waiting': return 'text-yellow-400';
      case 'offline': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-4 h-4" />;
      case 'waiting': return <Clock className="w-4 h-4" />;
      case 'offline': return <AlertTriangle className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-2">
          Valkey Pipeline Manager
        </h1>
        <p className="text-gray-400">OANDA Market Data → Valkey Timestamped Keys → GPU Processing Pipeline</p>
        
        {/* Pipeline Status */}
        <div className="mt-4 flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 ${getStatusColor(pipelineStatus)}`}>
            {getStatusIcon(pipelineStatus)}
            <span className="font-medium">Pipeline: {pipelineStatus.toUpperCase()}</span>
          </div>
          <div className="px-3 py-2 rounded-lg bg-gray-900 text-cyan-400">
            <span>Feed Rate: {oandaFeedRate.toFixed(1)} keys/sec</span>
          </div>
          <div className="px-3 py-2 rounded-lg bg-gray-900 text-blue-400">
            <span>Total Keys: {valkeyKeyCount.toLocaleString()}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Pipeline Flow & Key Management */}
        <div className="col-span-8">
          {/* OANDA → Valkey Flow */}
          <div className="bg-gray-900 rounded-lg p-4 mb-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <GitBranch className="w-5 h-5 text-cyan-400" />
              Market Data Pipeline Flow
            </h3>
            
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <TrendingUp className="w-8 h-8 text-green-400 mx-auto mb-2" />
                <h4 className="font-semibold">OANDA Feed</h4>
                <p className="text-sm text-gray-400">Live Market Data</p>
                <div className="mt-2 text-xl font-bold text-green-400">{oandaFeedRate.toFixed(1)}/s</div>
              </div>
              
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <Clock className="w-8 h-8 text-blue-400 mx-auto mb-2" />
                <h4 className="font-semibold">Key Generation</h4>
                <p className="text-sm text-gray-400">Timestamped Identities</p>
                <div className="mt-2 text-xl font-bold text-blue-400">{valkeyDistribution.total}</div>
              </div>
              
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <Database className="w-8 h-8 text-purple-400 mx-auto mb-2" />
                <h4 className="font-semibold">Valkey Storage</h4>
                <p className="text-sm text-gray-400">Manifold Database</p>
                <div className="mt-2 text-xl font-bold text-purple-400">
                  {sizeManagement.utilizationPercent.toFixed(1)}%
                </div>
              </div>
              
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <Cpu className="w-8 h-8 text-orange-400 mx-auto mb-2" />
                <h4 className="font-semibold">GPU Processing</h4>
                <p className="text-sm text-gray-400">Pattern Analysis</p>
                <div className="mt-2 text-xl font-bold text-orange-400">Ready</div>
              </div>
            </div>

            {/* Time-Intrinsic Identity Compression */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">Time-Intrinsic Identity Compression</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-cyan-500 rounded-full animate-pulse"></div>
                    <span>Hot Identities (0-5min)</span>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{valkeyDistribution.fresh}</div>
                    <div className="text-xs text-gray-400">Computing transitions</div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                    <span>Stabilizing (5-30min)</span>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{valkeyDistribution.consolidation}</div>
                    <div className="text-xs text-gray-400">Backwards derivable</div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span>Converged (30min+)</span>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{valkeyDistribution.settled}</div>
                    <div className="text-xs text-gray-400">Pin states locked</div>
                  </div>
                </div>
              </div>
              
              {/* Backwards Computation Principle */}
              <div className="mt-4 p-3 bg-purple-900/30 rounded border border-purple-600/30">
                <div className="text-xs text-purple-400 font-medium mb-2">⚡ Backwards Integration Principle</div>
                <div className="text-xs text-gray-300">
                  Given: Price + (Stability, Coherence, Entropy) → Unique Previous State
                  <br />
                  <span className="text-purple-300">Only ONE combination can produce these transitional metrics</span>
                </div>
              </div>
            </div>
          </div>

          {/* Backwards Computation & Identity Reconstruction */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Eye className="w-5 h-5 text-yellow-400" />
              Identity Reconstruction Matrix
            </h3>
            
            <div className="grid grid-cols-2 gap-6">
              {/* Time-Intrinsic Key Compression */}
              <div>
                <h4 className="font-medium mb-3">Time-Intrinsic Keys</h4>
                <div className="space-y-2">
                  {Object.entries(keyPatterns.instruments).slice(0, 5).map(([instrument, count]) => (
                    <div key={instrument} className="flex justify-between items-center">
                      <span className="text-sm font-mono">{instrument}</span>
                      <div className="text-right">
                        <div className="text-cyan-400 font-bold">{count}</div>
                        <div className="text-xs text-gray-500">identities</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-3 text-xs text-gray-400 italic">
                  Key identifier IS the time - no naming needed
                </div>
              </div>

              {/* Backwards Derivation Capability */}
              <div>
                <h4 className="font-medium mb-3">Backwards Integration</h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-2 bg-gray-800 rounded">
                    <span className="text-sm">Current Price</span>
                    <span className="text-green-400 font-mono">Known</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-gray-800 rounded">
                    <span className="text-sm">QBSA/QFH Metrics</span>
                    <span className="text-blue-400 font-mono">Computed</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-purple-800 rounded">
                    <span className="text-sm">Previous State</span>
                    <span className="text-purple-400 font-mono">Derivable</span>
                  </div>
                </div>
                <div className="mt-3 text-xs text-purple-300 italic">
                  Unique tangent: only ONE path leads here
                </div>
              </div>
            </div>
            
            {/* Mathematical Principle */}
            <div className="mt-6 p-4 bg-gradient-to-r from-purple-900/20 to-cyan-900/20 rounded border border-purple-500/30">
              <h5 className="font-medium text-purple-300 mb-2">Integration as Backwards Derivation</h5>
              <div className="text-sm text-gray-300 space-y-1">
                <div>• <span className="text-cyan-300">Time arrives intrinsically</span> → Market data follows</div>
                <div>• <span className="text-blue-300">OANDA data appends</span> → Valkey key instantiated</div>
                <div>• <span className="text-purple-300">Transitional metrics computed</span> → Previous state derivable</div>
                <div>• <span className="text-green-300">Pin state locked</span> → Identity converged</div>
              </div>
              <div className="mt-3 text-xs text-yellow-400 font-mono">
                ∫ f(t) dt = Backwards[Current_State + Quantum_Metrics] → Previous_State
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Server Metrics & Management */}
        <div className="col-span-4 space-y-6">
          {/* Server Resource Metrics */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Server className="w-5 h-5 text-green-400" />
              Droplet Resources
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Memory</span>
                  <span>{serverMetrics.memory.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-green-400 h-2 rounded-full transition-all"
                    style={{ width: `${serverMetrics.memory}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>CPU</span>
                  <span>{serverMetrics.cpu.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-blue-400 h-2 rounded-full transition-all"
                    style={{ width: `${serverMetrics.cpu}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Disk Usage</span>
                  <span>{serverMetrics.diskUsage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-orange-400 h-2 rounded-full transition-all"
                    style={{ width: `${serverMetrics.diskUsage}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Network I/O</span>
                  <span>{serverMetrics.networkIO.toFixed(1)} MB/s</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-cyan-400 h-2 rounded-full transition-all"
                    style={{ width: `${Math.min(serverMetrics.networkIO, 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Valkey Size Management */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <HardDrive className="w-5 h-5 text-purple-400" />
              Valkey Size Management
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400 text-sm">Current Size</span>
                <span className="font-medium">{(sizeManagement.currentSize / 1024).toFixed(2)} MB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 text-sm">Growth Rate</span>
                <span className="font-medium">{(sizeManagement.projectedGrowth / 1024).toFixed(2)} MB/hr</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 text-sm">Utilization</span>
                <span className={`font-medium ${sizeManagement.utilizationPercent > 80 ? 'text-red-400' : 'text-green-400'}`}>
                  {sizeManagement.utilizationPercent.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 text-sm">Time to Capacity</span>
                <span className="font-medium">
                  {sizeManagement.timeToCapacity === Infinity ? '∞' : `${sizeManagement.timeToCapacity.toFixed(1)}h`}
                </span>
              </div>
            </div>
          </div>

          {/* Retention Strategy */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Gauge className="w-5 h-5 text-orange-400" />
              Retention Strategy
            </h3>
            <div className="space-y-2 text-sm">
              {Object.entries(sizeManagement.retentionTiers).map(([tier, config]) => (
                <div key={tier} className="flex justify-between items-center py-1">
                  <span className="capitalize">{tier}</span>
                  <div className="text-right">
                    <div>{config.maxAge}min • {config.retention}%</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Pipeline Controls */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-400" />
              Pipeline Controls
            </h3>
            <div className="space-y-2">
              <button className="w-full py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition-colors">
                Start OANDA Feed
              </button>
              <button className="w-full py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors">
                Sync GPU Processing
              </button>
              <button className="w-full py-2 bg-orange-600 hover:bg-orange-700 rounded-lg text-sm font-medium transition-colors">
                Optimize Key Retention
              </button>
              <button className="w-full py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm font-medium transition-colors">
                Emergency Cleanup
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ValkeyPipelineManager;