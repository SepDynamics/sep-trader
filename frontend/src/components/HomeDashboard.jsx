import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, TrendingUp, TrendingDown, DollarSign, AlertCircle, Play, Pause, Settings, BarChart3, Brain, Clock, ChevronUp, ChevronDown, CheckCircle, XCircle, RefreshCw, Database, Cpu, HardDrive, Zap, Target, Shield, Eye, Terminal, Upload, FileText, GitBranch, Server } from 'lucide-react';
import { useWebSocket } from '../context/WebSocketContext';
import QuickActionButton from './QuickActionButton';
import { uploadTrainingData, startModelTraining, generateReport, getConfiguration, getSystemStatus } from '../services/api';

// This dashboard connects to your actual backend services
// Backend API: http://localhost:5000/api/
// WebSocket: ws://localhost:8765/

const HomeDashboard = () => {
  const {
    connected: wsConnected,
    systemStatus,
    performanceData,
    marketData,
    tradingSignals,
    systemMetrics
  } = useWebSocket();

  const [apiHealth, setApiHealth] = useState('checking');
  
  
  // Live metrics from API
  const liveMetrics = systemMetrics || {
    confidence: 0,
    coherence: 0,
    stability: 0,
    flip_ratio: 0,
    rupture_ratio: 0,
    entropy: 0
  };
  
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedPairs, setSelectedPairs] = useState([]);
  const [commandHistory, setCommandHistory] = useState([]);
  const [commandInput, setCommandInput] = useState('');
  const [isExecutingCommand, setIsExecutingCommand] = useState(false);
  const [actionStatus, setActionStatus] = useState(null);

  // API base URL from environment or default
  const API_URL = window._env_?.REACT_APP_API_URL || 'http://localhost:5000';


  // Execute CLI command via API
  const executeCommand = async () => {
    if (!commandInput.trim()) return;
    
    setIsExecutingCommand(true);
    const timestamp = new Date().toISOString();
    
    try {
      const response = await fetch(`${API_URL}/api/commands/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: commandInput })
      });
      
      const result = await response.json();
      
      setCommandHistory(prev => [{
        command: commandInput,
        result: result,
        timestamp: timestamp,
        success: response.ok
      }, ...prev.slice(0, 49)]);
      
      setCommandInput('');
    } catch (error) {
      setCommandHistory(prev => [{
        command: commandInput,
        result: { error: error.message },
        timestamp: timestamp,
        success: false
      }, ...prev.slice(0, 49)]);
    } finally {
      setIsExecutingCommand(false);
    }
  };

  // Enable/disable trading pair
  const fetchSystemStatus = async () => {
    try {
      await getSystemStatus();
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  const togglePair = async (pair, enable) => {
    try {
      const endpoint = enable
        ? `${API_URL}/api/pairs/${pair}/enable`
        : `${API_URL}/api/pairs/${pair}/disable`;

      const response = await fetch(endpoint, { method: 'POST' });
      if (response.ok) {
        await fetchSystemStatus();
      }
    } catch (error) {
      console.error(`Failed to ${enable ? 'enable' : 'disable'} pair ${pair}:`, error);
    }
  };

  // Start/stop trading
  const toggleTrading = async () => {
    const isRunning = systemStatus.status === 'running';
    const endpoint = isRunning ? 
      `${API_URL}/api/trading/stop` : 
      `${API_URL}/api/trading/start`;
    
    try {
      const response = await fetch(endpoint, { method: 'POST' });
      if (response.ok) {
        await fetchSystemStatus();
      }
    } catch (error) {
      console.error(`Failed to ${isRunning ? 'stop' : 'start'} trading:`, error);
    }
  };

  const handleUploadData = async () => {
    try {
      await uploadTrainingData();
      setActionStatus({ type: 'success', message: 'Training data uploaded' });
    } catch (error) {
      setActionStatus({ type: 'error', message: 'Upload failed' });
    }
  };

  const handleStartTraining = async () => {
    try {
      await startModelTraining();
      setActionStatus({ type: 'success', message: 'Model training started' });
    } catch (error) {
      setActionStatus({ type: 'error', message: 'Training start failed' });
    }
  };

  const handleGenerateReport = async () => {
    try {
      await generateReport();
      setActionStatus({ type: 'success', message: 'Report generation started' });
    } catch (error) {
      setActionStatus({ type: 'error', message: 'Report generation failed' });
    }
  };

  const handleOpenConfig = async () => {
    try {
      await getConfiguration();
      setActionStatus({ type: 'success', message: 'Configuration loaded' });
    } catch (error) {
      setActionStatus({ type: 'error', message: 'Failed to load configuration' });
    }
  };

  // Initialize dashboard
  useEffect(() => {
    // Set up polling interval for API health
    const healthInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/api/status`);
        setApiHealth(response.ok ? 'healthy' : 'error');
      } catch (error) {
        setApiHealth('error');
      }
    }, 5000);
    
    return () => {
      clearInterval(healthInterval);
    };
  }, []);

  // Format currency
  const formatCurrency = (value) => {
    if (value === null || value === undefined) return '$0.00';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value) => {
    if (value === null || value === undefined) return '0.00%';
    const formatted = (value * 100).toFixed(2);
    const sign = value >= 0 ? '+' : '';
    return `${sign}${formatted}%`;
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'running':
      case 'healthy':
      case 'connected':
        return 'text-green-500';
      case 'stopped':
      case 'disconnected':
        return 'text-yellow-500';
      case 'error':
      case 'critical':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              SEP Trading System
            </h1>
            <p className="text-gray-400 mt-1">Quantum Field Harmonics Professional Trading</p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* System Status */}
            <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 rounded-lg">
              <Server className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">API:</span>
              <span className={`text-sm font-medium ${getStatusColor(apiHealth)}`}>
                {apiHealth}
              </span>
            </div>
            
            {/* WebSocket Status */}
            <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 rounded-lg">
              <Activity className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">WS:</span>
              <span className={`text-sm font-medium ${wsConnected ? 'text-green-500' : 'text-red-500'}`}>
                {wsConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {/* Trading Control */}
            <button
              onClick={toggleTrading}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                systemStatus.status === 'running'
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {systemStatus.status === 'running' ? (
                <>
                  <Pause className="w-4 h-4 inline mr-2" />
                  Stop Trading
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 inline mr-2" />
                  Start Trading
                </>
              )}
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-2 border-b border-gray-800">
          {['overview', 'pairs', 'signals', 'performance', 'metrics', 'terminal'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 capitalize transition-all ${
                activeTab === tab
                  ? 'border-b-2 border-blue-500 text-blue-400'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Key Metrics */}
        <div className="col-span-3 space-y-4">
          {/* Performance Card */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Performance</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Daily P&L</span>
                <span className={`font-medium ${performanceData.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(performanceData.daily_pnl)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Total Return</span>
                <span className={`font-medium ${performanceData.total_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercentage(performanceData.total_return)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Win Rate</span>
                <span className="font-medium text-blue-400">
                  {formatPercentage(performanceData.win_rate)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Sharpe Ratio</span>
                <span className="font-medium text-gray-200">
                  {performanceData.sharpe_ratio?.toFixed(2) || '0.00'}
                </span>
              </div>
            </div>
          </div>

          {/* QFH Metrics Card */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">QFH Engine</h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-gray-400 text-sm">Confidence</span>
                  <span className="text-sm font-medium">{liveMetrics.confidence?.toFixed(1) || 0}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-1.5">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all"
                    style={{ width: `${liveMetrics.confidence || 0}%` }}
                  />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-gray-400 text-sm">Coherence</span>
                  <span className="text-sm font-medium">{liveMetrics.coherence?.toFixed(1) || 0}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-1.5">
                  <div
                    className="bg-purple-500 h-1.5 rounded-full transition-all"
                    style={{ width: `${liveMetrics.coherence || 0}%` }}
                  />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-gray-400 text-sm">Stability</span>
                  <span className="text-sm font-medium">{liveMetrics.stability?.toFixed(1) || 0}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-1.5">
                  <div
                    className="bg-green-500 h-1.5 rounded-full transition-all"
                    style={{ width: `${liveMetrics.stability || 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* System Info */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">System Info</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Version</span>
                <span className="text-gray-200">{systemStatus.version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Market</span>
                <span className={`font-medium ${systemStatus.market === 'open' ? 'text-green-400' : 'text-yellow-400'}`}>
                  {systemStatus.market}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Active Pairs</span>
                <span className="text-gray-200">{systemStatus.pairs?.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Last Sync</span>
                <span className="text-gray-200 text-xs">
                  {systemStatus.last_sync ? new Date(systemStatus.last_sync).toLocaleTimeString() : 'Never'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Center Panel - Main Content Area */}
        <div className="col-span-6">
          {activeTab === 'overview' && (
            <div className="space-y-4">
              {/* Market Overview */}
              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-lg font-medium mb-4">Market Overview</h3>
                <div className="grid grid-cols-3 gap-4">
                  {systemStatus.pairs?.map(pair => (
                    <div key={pair} className="bg-gray-800 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-sm">{pair}</span>
                        <Activity className="w-4 h-4 text-green-400" />
                      </div>
                      {marketData[pair] && (
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Price</span>
                            <span>{marketData[pair].price?.toFixed(5)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">24h</span>
                            <span className={marketData[pair].change_24h >= 0 ? 'text-green-400' : 'text-red-400'}>
                              {formatPercentage(marketData[pair].change_24h / 100)}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Recent Signals */}
              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-lg font-medium mb-4">Recent Trading Signals</h3>
                <div className="space-y-2">
                  {tradingSignals.slice(0, 5).map((signal, idx) => (
                    <div key={idx} className="bg-gray-800 rounded-lg p-3 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`px-2 py-1 rounded text-xs font-medium ${
                          signal.signal_type === 'buy' ? 'bg-green-900 text-green-300' :
                          signal.signal_type === 'sell' ? 'bg-red-900 text-red-300' :
                          'bg-gray-700 text-gray-300'
                        }`}>
                          {signal.signal_type?.toUpperCase()}
                        </div>
                        <span className="font-medium">{signal.symbol}</span>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-gray-400">Confidence:</span>
                        <span className="font-medium">{(signal.confidence * 100).toFixed(1)}%</span>
                        <span className="text-gray-500 text-xs">
                          {new Date(signal.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ))}
                  {tradingSignals.length === 0 && (
                    <div className="text-center text-gray-500 py-8">
                      No signals yet. Waiting for market activity...
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'pairs' && (
            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-medium mb-4">Trading Pairs Management</h3>
              <div className="space-y-2">
                {['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD'].map(pair => {
                  const isEnabled = systemStatus.pairs?.includes(pair);
                  return (
                    <div key={pair} className="bg-gray-800 rounded-lg p-4 flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <span className="font-medium text-lg">{pair}</span>
                        <span className={`text-sm ${isEnabled ? 'text-green-400' : 'text-gray-500'}`}>
                          {isEnabled ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                      <button
                        onClick={() => togglePair(pair, !isEnabled)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                          isEnabled
                            ? 'bg-red-900 hover:bg-red-800 text-red-300'
                            : 'bg-green-900 hover:bg-green-800 text-green-300'
                        }`}
                      >
                        {isEnabled ? 'Disable' : 'Enable'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {activeTab === 'terminal' && (
            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
                <Terminal className="w-5 h-5" />
                CLI Terminal
              </h3>
              
              {/* Command Input */}
              <div className="flex gap-2 mb-4">
                <input
                  type="text"
                  value={commandInput}
                  onChange={(e) => setCommandInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && executeCommand()}
                  placeholder="Enter command..."
                  className="flex-1 bg-gray-800 text-gray-100 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isExecutingCommand}
                />
                <button
                  onClick={executeCommand}
                  disabled={isExecutingCommand}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium disabled:opacity-50"
                >
                  {isExecutingCommand ? 'Executing...' : 'Execute'}
                </button>
              </div>
              
              {/* Command History */}
              <div className="bg-gray-800 rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm">
                {commandHistory.map((entry, idx) => (
                  <div key={idx} className="mb-4">
                    <div className="text-green-400 mb-1">
                      $ {entry.command}
                    </div>
                    <div className={entry.success ? 'text-gray-300' : 'text-red-400'}>
                      {typeof entry.result === 'object' 
                        ? JSON.stringify(entry.result, null, 2)
                        : entry.result}
                    </div>
                    <div className="text-gray-600 text-xs mt-1">
                      {new Date(entry.timestamp).toLocaleString()}
                    </div>
                  </div>
                ))}
                {commandHistory.length === 0 && (
                  <div className="text-gray-500 text-center py-8">
                    No commands executed yet
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Additional Info */}
        <div className="col-span-3 space-y-4">
          {/* Quick Actions */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <QuickActionButton
                className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-left transition-colors"
                icon={<Upload className="w-4 h-4 inline mr-2" />}
                onClick={handleUploadData}
              >
                Upload Training Data
              </QuickActionButton>
              <QuickActionButton
                className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-left transition-colors"
                icon={<Brain className="w-4 h-4 inline mr-2" />}
                onClick={handleStartTraining}
              >
                Start Model Training
              </QuickActionButton>
              <QuickActionButton
                className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-left transition-colors"
                icon={<FileText className="w-4 h-4 inline mr-2" />}
                onClick={handleGenerateReport}
              >
                Generate Report
              </QuickActionButton>
              <QuickActionButton
                className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-left transition-colors"
                icon={<Settings className="w-4 h-4 inline mr-2" />}
                onClick={handleOpenConfig}
              >
                System Configuration
              </QuickActionButton>
            </div>
            {actionStatus && (
              <div className={`mt-2 text-sm ${actionStatus.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
                {actionStatus.message}
              </div>
            )}
          </div>

          {/* System Metrics */}
          {systemMetrics && (
            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">System Resources</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-gray-400 text-sm flex items-center gap-1">
                      <Cpu className="w-3 h-3" /> CPU
                    </span>
                    <span className="text-sm">{systemMetrics.cpu_usage?.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1.5">
                    <div 
                      className="bg-blue-500 h-1.5 rounded-full"
                      style={{ width: `${systemMetrics.cpu_usage || 0}%` }}
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-gray-400 text-sm flex items-center gap-1">
                      <Database className="w-3 h-3" /> Memory
                    </span>
                    <span className="text-sm">{systemMetrics.memory_usage?.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1.5">
                    <div 
                      className="bg-purple-500 h-1.5 rounded-full"
style={{ width: `${systemMetrics.memory_usage || 0}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Activity Feed */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Activity Feed</h3>
            <div className="space-y-2 text-sm">
              {tradingSignals.slice(0, 3).map((signal, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500 mt-1.5" />
                  <div className="flex-1">
                    <p className="text-gray-300">
                      {signal.signal_type} signal for {signal.symbol}
                    </p>
                    <p className="text-gray-500 text-xs">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              {tradingSignals.length === 0 && (
                <p className="text-gray-500 text-center py-4">No recent activity</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomeDashboard;
                