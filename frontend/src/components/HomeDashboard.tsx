import React, { useState } from 'react';
import RealTimeMarketFeed from './RealTimeMarketFeed';
import ManifoldVisualizer from './ManifoldVisualizer';
import MetricTimeSeries from './MetricTimeSeries';
import IdentityInspector from './IdentityInspector';
import AppHeader from './AppHeader';
import { ManifoldProvider } from '../context/ManifoldContext';
import { useSymbol } from '../context/SymbolContext';
import { useWebSocket } from '../context/WebSocketContext';
import { Symbol } from '../config/symbols';
import {
  Database,
  Activity,
  Eye,
  Target,
  TrendingUp,
  Zap,
  Clock,
  ArrowRight
} from 'lucide-react';

const HomeDashboard: React.FC = () => {
  const { selectedSymbol, setSelectedSymbol, symbols, isValidSymbol } = useSymbol();
  const { connected, quantumSignals, livePatterns } = useWebSocket();
  const [activeView, setActiveView] = useState<'market' | 'manifold' | 'timeseries' | 'inspector'>('market');
  const [selectedMetric, setSelectedMetric] = useState<'entropy' | 'stability' | 'coherence'>('entropy');

  const handleSymbolChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    if (isValidSymbol(value)) {
      setSelectedSymbol(value);
    }
  };

  // View configurations
  const views = [
    {
      id: 'market' as const,
      label: 'Market Feed',
      icon: TrendingUp,
      description: 'Real-time market data and price movements'
    },
    {
      id: 'manifold' as const,
      label: 'Manifold',
      icon: Target,
      description: 'Quantum manifold visualization and entropy bands'
    },
    {
      id: 'timeseries' as const,
      label: 'Time Series',
      icon: Activity,
      description: 'Advanced metric time series analysis'
    },
    {
      id: 'inspector' as const,
      label: 'Inspector',
      icon: Eye,
      description: 'Detailed quantum identity analysis'
    }
  ];

  const renderActiveView = () => {
    switch (activeView) {
      case 'market':
        return <RealTimeMarketFeed />;
      case 'manifold':
        return <ManifoldVisualizer />;
      case 'timeseries':
        return <MetricTimeSeries metric={selectedMetric} instrument={selectedSymbol} />;
      case 'inspector':
        return <IdentityInspector />;
      default:
        return <RealTimeMarketFeed />;
    }
  };

  return (
    <ManifoldProvider>
      <div className="min-h-screen bg-gray-950 text-gray-200 flex flex-col font-sans">
        <AppHeader />

        <main className="flex-grow container mx-auto p-4 sm:p-6 lg:p-8">
          {/* Dashboard Header */}
          <div className="mb-6">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-4">
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                  SEP Engine Dashboard
                </h1>
                <p className="text-gray-400 mt-1">
                  Quantum Manifold Analysis • Valkey Integration • Real-time Pattern Recognition
                </p>
              </div>

              {/* System Status */}
              <div className="flex items-center gap-4 mt-4 lg:mt-0">
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-900 rounded-lg">
                  <Database className="w-4 h-4" />
                  <span className="text-sm">Valkey:</span>
                  <span className={`text-sm font-medium ${connected ? 'text-green-400' : 'text-red-400'}`}>
                    {connected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-900 rounded-lg">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm">Signals:</span>
                  <span className="text-sm font-medium text-blue-400">
                    {Object.keys(quantumSignals).length}
                  </span>
                </div>

                <div className="flex items-center gap-2 px-3 py-2 bg-gray-900 rounded-lg">
                  <Target className="w-4 h-4 text-purple-400" />
                  <span className="text-sm">Patterns:</span>
                  <span className="text-sm font-medium text-cyan-400">
                    {Object.keys(livePatterns).length}
                  </span>
                </div>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                {views.map((view) => {
                  const Icon = view.icon;
                  return (
                    <button
                      key={view.id}
                      onClick={() => setActiveView(view.id)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                        activeView === view.id
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span className="font-medium">{view.label}</span>
                    </button>
                  );
                })}
              </div>

              <div className="flex items-center gap-4">
                {/* Metric Selector for Time Series */}
                {activeView === 'timeseries' && (
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-400">Metric:</span>
                    <select
                      value={selectedMetric}
                      onChange={(e) => setSelectedMetric(e.target.value as any)}
                      className="bg-gray-800 border border-gray-600 rounded-md px-3 py-1.5 text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="entropy">Entropy</option>
                      <option value="stability">Stability</option>
                      <option value="coherence">Coherence</option>
                    </select>
                  </div>
                )}

                {/* Instrument Selector */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-400">Instrument:</span>
                  <select
                    className="bg-gray-800 border border-gray-600 rounded-md px-3 py-1.5 text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={selectedSymbol}
                    onChange={handleSymbolChange}
                    disabled // UNTIL other pairs are seeded and verified
                  >
                    {symbols.map(s => <option key={s} value={s}>{s.replace('_', '/')}</option>)}
                  </select>
                </div>
              </div>
            </div>

            {/* View Description */}
            <div className="mt-3 p-3 bg-gray-900/50 rounded-lg border border-gray-800">
              <p className="text-sm text-gray-400">
                {views.find(v => v.id === activeView)?.description}
              </p>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="bg-gray-900/30 rounded-xl shadow-2xl backdrop-blur-md border border-gray-700/50 overflow-hidden">
            {activeView !== 'manifold' && activeView !== 'inspector' && (
              <div className="p-4 sm:p-6 border-b border-gray-700/50">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
                  <h2 className="text-2xl font-bold text-white mb-2 sm:mb-0 flex items-center gap-2">
                    {views.find(v => v.id === activeView)?.icon &&
                      React.createElement(views.find(v => v.id === activeView)!.icon, {
                        className: "w-6 h-6"
                      })
                    }
                    {activeView === 'market' && `${selectedSymbol.replace('_', '/')} Market Feed`}
                    {activeView === 'timeseries' && `${selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)} Analysis`}
                  </h2>
                  
                  {activeView === 'market' && (
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                      <Clock className="w-4 h-4" />
                      <span>Live Updates Every 500ms</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            <div className={activeView === 'manifold' || activeView === 'inspector' ? '' : 'p-2 sm:p-4 bg-black/20'}>
              {renderActiveView()}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="font-semibold mb-2 flex items-center gap-2">
                <Database className="w-5 h-5 text-cyan-400" />
                Valkey Health
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Connection:</span>
                  <span className={connected ? 'text-green-400' : 'text-red-400'}>
                    {connected ? 'Stable' : 'Lost'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Active Keys:</span>
                  <span className="text-blue-400">{Object.keys(quantumSignals).length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Pattern Space:</span>
                  <span className="text-purple-400">{Object.keys(livePatterns).length}</span>
                </div>
              </div>
            </div>

            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="font-semibold mb-2 flex items-center gap-2">
                <Activity className="w-5 h-5 text-green-400" />
                System Metrics
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Avg Entropy:</span>
                  <span className="text-red-400">
                    {Object.keys(quantumSignals).length > 0 ?
                      (Object.values(quantumSignals).reduce((sum: number, s: any) => sum + Number(s.entropy || 0.5), 0) / Object.keys(quantumSignals).length).toFixed(3) :
                      '0.000'
                    }
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Convergence:</span>
                  <span className="text-green-400">
                    {Object.values(quantumSignals).filter((s: any) => s.state === 'converged').length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Active Patterns:</span>
                  <span className="text-yellow-400">{Object.keys(livePatterns).length}</span>
                </div>
              </div>
            </div>

            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="font-semibold mb-2 flex items-center gap-2">
                <Target className="w-5 h-5 text-purple-400" />
                Quick Navigation
              </h3>
              <div className="space-y-2">
                {activeView !== 'manifold' && (
                  <button
                    onClick={() => setActiveView('manifold')}
                    className="flex items-center justify-between w-full p-2 bg-gray-800 hover:bg-gray-700 rounded transition-colors text-sm"
                  >
                    <span>View Manifold</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                )}
                {activeView !== 'inspector' && (
                  <button
                    onClick={() => setActiveView('inspector')}
                    className="flex items-center justify-between w-full p-2 bg-gray-800 hover:bg-gray-700 rounded transition-colors text-sm"
                  >
                    <span>Inspect Identities</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                )}
                {activeView !== 'timeseries' && (
                  <button
                    onClick={() => setActiveView('timeseries')}
                    className="flex items-center justify-between w-full p-2 bg-gray-800 hover:bg-gray-700 rounded transition-colors text-sm"
                  >
                    <span>Time Series Analysis</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </ManifoldProvider>
  );
};

export default HomeDashboard;