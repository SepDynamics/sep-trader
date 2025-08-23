import React, { useState, useEffect, useMemo } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { apiClient } from '../services/api';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Zap, 
  Target, 
  Shield, 
  Brain, 
  Database,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Play,
  Pause,
  Settings,
  BarChart3
} from 'lucide-react';

const TradingCockpit = () => {
  const {
    connected,
    marketData,
    systemStatus,
    performanceData,
    quantumSignals,
    signalHistory,
    valkeyMetrics,
    livePatterns
  } = useWebSocket();

  const [activeInstrument, setActiveInstrument] = useState('EURUSD');
  const [signalFilter, setSignalFilter] = useState('all'); // all, active, settling, settled
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');

  // Real-time signal processing
  const activeSignals = useMemo(() => {
    return Object.entries(quantumSignals).filter(([key, signal]) => {
      if (signalFilter === 'all') return true;
      return signal.state === signalFilter;
    }).map(([key, signal]) => ({ key, ...signal }));
  }, [quantumSignals, signalFilter]);

  const instrumentSignals = useMemo(() => {
    return activeSignals.filter(signal => 
      signal.instrument === activeInstrument
    ).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }, [activeSignals, activeInstrument]);

  // Signal state statistics
  const signalStats = useMemo(() => {
    const stats = { active: 0, settling: 0, settled: 0, total: 0 };
    Object.values(quantumSignals).forEach(signal => {
      stats[signal.state] = (stats[signal.state] || 0) + 1;
      stats.total++;
    });
    return stats;
  }, [quantumSignals]);

  // Pattern coherence analysis
  const patternAnalysis = useMemo(() => {
    const patterns = Object.values(livePatterns);
    if (patterns.length === 0) return { avgCoherence: 0, activePatterns: 0, strongPatterns: 0 };

    const totalCoherence = patterns.reduce((sum, p) => sum + (p.coherence || 0), 0);
    const avgCoherence = totalCoherence / patterns.length;
    const strongPatterns = patterns.filter(p => (p.coherence || 0) > 0.8).length;

    return {
      avgCoherence,
      activePatterns: patterns.length,
      strongPatterns
    };
  }, [livePatterns]);

  const refreshData = async () => {
    setRefreshing(true);
    try {
      await Promise.all([
        apiClient.getMarketData(),
        apiClient.getPerformanceMetrics(),
        apiClient.getSystemStatus()
      ]);
    } catch (error) {
      console.error('Failed to refresh data:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const formatPrice = (value) => {
    if (value === null || value === undefined) return '--';
    return parseFloat(value).toFixed(5);
  };

  const formatPercentage = (value) => {
    if (value === null || value === undefined) return '0.00%';
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatCurrency = (value) => {
    if (value === null || value === undefined) return '$0.00';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const getSignalStateColor = (state) => {
    switch (state) {
      case 'active': return 'text-blue-400';
      case 'settling': return 'text-yellow-400';
      case 'settled': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getSignalStateIcon = (state) => {
    switch (state) {
      case 'active': return <Activity className="w-4 h-4" />;
      case 'settling': return <RefreshCw className="w-4 h-4 animate-spin" />;
      case 'settled': return <CheckCircle className="w-4 h-4" />;
      default: return <XCircle className="w-4 h-4" />;
    }
  };

  const instruments = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD'];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              SEP Trading Cockpit
            </h1>
            <p className="text-gray-400 mt-1">Real-time Quantum Signal Analysis & Trading Control</p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Connection Status */}
            <div className="flex items-center gap-2 px-3 py-2 bg-gray-900 rounded-lg">
              <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm text-gray-400">
                {connected ? 'Live' : 'Offline'}
              </span>
            </div>

            {/* Valkey Status */}
            <div className="flex items-center gap-2 px-3 py-2 bg-gray-900 rounded-lg">
              <Database className="w-4 h-4 text-purple-400" />
              <span className="text-sm text-gray-400">Valkey:</span>
              <span className="text-sm font-medium text-purple-400">
                {valkeyMetrics.status || 'Connected'}
              </span>
            </div>

            {/* Refresh Button */}
            <button
              onClick={refreshData}
              disabled={refreshing}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Signal Overview */}
        <div className="col-span-3 space-y-4">
          {/* Signal Statistics */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              Quantum Signals
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Active</span>
                <span className="text-blue-400 font-medium">{signalStats.active}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Settling</span>
                <span className="text-yellow-400 font-medium">{signalStats.settling}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Settled</span>
                <span className="text-green-400 font-medium">{signalStats.settled}</span>
              </div>
              <div className="border-t border-gray-800 pt-2">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Total</span>
                  <span className="text-gray-200 font-medium">{signalStats.total}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Pattern Analysis */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-400" />
              Pattern Analysis
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Avg Coherence</span>
                <span className="text-blue-400 font-medium">
                  {formatPercentage(patternAnalysis.avgCoherence)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Active Patterns</span>
                <span className="text-gray-200 font-medium">{patternAnalysis.activePatterns}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Strong Patterns</span>
                <span className="text-green-400 font-medium">{patternAnalysis.strongPatterns}</span>
              </div>
            </div>
          </div>

          {/* Performance Summary */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-green-400" />
              Performance
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Daily P&L</span>
                <span className={`font-medium ${(performanceData.daily_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(performanceData.daily_pnl)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Total P&L</span>
                <span className={`font-medium ${(performanceData.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(performanceData.total_pnl)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Win Rate</span>
                <span className="text-blue-400 font-medium">
                  {formatPercentage(performanceData.win_rate)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Center Panel - Main Signal Display */}
        <div className="col-span-6 space-y-4">
          {/* Instrument Selector */}
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Active Signals</h3>
              <div className="flex items-center gap-4">
                {/* Instrument Selector */}
                <select
                  value={activeInstrument}
                  onChange={(e) => setActiveInstrument(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
                >
                  {instruments.map(inst => (
                    <option key={inst} value={inst}>{inst}</option>
                  ))}
                </select>

                {/* State Filter */}
                <select
                  value={signalFilter}
                  onChange={(e) => setSignalFilter(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
                >
                  <option value="all">All States</option>
                  <option value="active">Active</option>
                  <option value="settling">Settling</option>
                  <option value="settled">Settled</option>
                </select>
              </div>
            </div>

            {/* Signals List */}
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {instrumentSignals.length > 0 ? (
                instrumentSignals.map((signal) => (
                  <div key={signal.key} className="bg-gray-800 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className={getSignalStateColor(signal.state)}>
                          {getSignalStateIcon(signal.state)}
                        </div>
                        <span className="font-medium">{signal.instrument}</span>
                        <span className="text-sm text-gray-400">
                          {new Date(signal.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="text-sm text-gray-400">
                        {signal.direction || 'neutral'}
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="text-gray-400 mb-1">Price</div>
                        <div className="font-medium">{formatPrice(signal.price)}</div>
                      </div>
                      <div>
                        <div className="text-gray-400 mb-1">Entropy</div>
                        <div className="font-medium">
                          <div className="flex items-center gap-2">
                            <span>{formatPercentage(signal.entropy)}</span>
                            <div className="w-12 h-1 bg-gray-700 rounded-full">
                              <div 
                                className="h-1 bg-yellow-500 rounded-full transition-all"
                                style={{ width: `${(signal.entropy || 0) * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400 mb-1">Stability</div>
                        <div className="font-medium">
                          <div className="flex items-center gap-2">
                            <span>{formatPercentage(signal.stability)}</span>
                            <div className="w-12 h-1 bg-gray-700 rounded-full">
                              <div 
                                className="h-1 bg-green-500 rounded-full transition-all"
                                style={{ width: `${(signal.stability || 0) * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400 mb-1">Coherence</div>
                        <div className="font-medium">
                          <div className="flex items-center gap-2">
                            <span>{formatPercentage(signal.coherence)}</span>
                            <div className="w-12 h-1 bg-gray-700 rounded-full">
                              <div 
                                className="h-1 bg-blue-500 rounded-full transition-all"
                                style={{ width: `${(signal.coherence || 0) * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Signal Evolution Timeline */}
                    <div className="mt-3 pt-3 border-t border-gray-700">
                      <div className="text-xs text-gray-400 mb-2">Evolution Timeline</div>
                      <div className="flex items-center gap-1">
                        {/* Simple timeline visualization */}
                        <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-500 via-yellow-500 to-green-500 transition-all"
                            style={{ 
                              width: signal.state === 'active' ? '30%' : 
                                     signal.state === 'settling' ? '70%' : '100%'
                            }}
                          />
                        </div>
                        <span className="text-xs text-gray-400 ml-2">
                          {signal.state === 'active' ? 'Initializing' :
                           signal.state === 'settling' ? 'Processing' : 'Complete'}
                        </span>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-gray-500 py-8">
                  <Brain className="w-12 h-12 mx-auto mb-3 text-gray-600" />
                  <p>No {signalFilter !== 'all' ? signalFilter : ''} signals for {activeInstrument}</p>
                  <p className="text-sm">Waiting for quantum analysis...</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Market Data & Controls */}
        <div className="col-span-3 space-y-4">
          {/* Market Overview */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3">Market Overview</h3>
            <div className="space-y-2">
              {instruments.slice(0, 4).map(instrument => {
                const data = marketData[instrument] || {};
                return (
                  <div key={instrument} className="flex items-center justify-between text-sm">
                    <span className="font-medium">{instrument}</span>
                    <div className="flex items-center gap-2">
                      <span>{formatPrice(data.price)}</span>
                      <span className={`text-xs ${(data.change || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(data.change || 0) >= 0 ? '▲' : '▼'}
                        {Math.abs(data.change || 0).toFixed(4)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Live Patterns */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-400" />
              Live Patterns
            </h3>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {Object.entries(livePatterns).slice(0, 5).map(([patternId, pattern]) => (
                <div key={patternId} className="bg-gray-800 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Pattern {patternId.slice(-4)}</span>
                    <span className="text-xs text-gray-400">
                      {new Date(pattern.lastUpdate).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="text-xs space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Coherence:</span>
                      <span className="text-blue-400">{formatPercentage(pattern.coherence)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Strength:</span>
                      <span className="text-green-400">{formatPercentage(pattern.strength || 0)}</span>
                    </div>
                  </div>
                </div>
              ))}
              {Object.keys(livePatterns).length === 0 && (
                <div className="text-center text-gray-500 py-4 text-sm">
                  No active patterns
                </div>
              )}
            </div>
          </div>

          {/* System Controls */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3">System Controls</h3>
            <div className="space-y-2">
              <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition-colors">
                <Play className="w-4 h-4" />
                Start Analysis
              </button>
              <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm font-medium transition-colors">
                <Pause className="w-4 h-4" />
                Pause Analysis
              </button>
              <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors">
                <Settings className="w-4 h-4" />
                Configuration
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingCockpit;