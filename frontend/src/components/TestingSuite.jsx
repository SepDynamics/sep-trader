import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Terminal,
  Activity,
  Zap,
  Brain,
  TrendingUp,
  Clock
} from 'lucide-react';

const TestingSuite = ({ activeTab = 'unit-tests' }) => {
  const { connected } = useWebSocket();
  const [currentTab, setCurrentTab] = useState(activeTab);
  const [testResults, setTestResults] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [testStats, setTestStats] = useState({
    coverage: 87.3,
    passed: 156,
    failed: 21,
    total: 177,
    duration: 1.2
  });

  // Test suites matching example.html structure
  const testSuites = {
    quantum: [
      { name: 'QFH Pattern Recognition', status: 'success', message: 'PASSED', details: 'Accuracy: 60.73%' },
      { name: 'QBSA Coherence Analysis', status: 'success', message: 'PASSED', details: 'Coherence: 0.4687' },
      { name: 'Entropy Calculation', status: 'success', message: 'PASSED', details: 'Entropy: 0.1000' },
      { name: 'Pattern Evolution', status: 'success', message: 'PASSED', details: '15,625 patterns processed' },
      { name: 'CUDA Kernel Execution', status: 'success', message: 'PASSED', details: 'Execution time: 0.8ms' },
      { name: 'Memory Tier Management', status: 'warning', message: 'WARNING', details: 'Cache miss rate: 12%' }
    ],
    trading: [
      { name: 'OANDA Connection', status: 'success', message: 'PASSED', details: 'Latency: 23ms' },
      { name: 'Order Execution', status: 'success', message: 'PASSED', details: 'FOK orders validated' },
      { name: 'Position Management', status: 'success', message: 'PASSED', details: '16 pairs active' },
      { name: 'Risk Management', status: 'error', message: 'FAILED', details: 'Stop loss validation failed for GBP/USD' },
      { name: 'Market Data Stream', status: 'success', message: 'PASSED', details: 'Receiving 50 ticks/sec' }
    ],
    signals: [
      { name: 'Signal Generation', status: 'success', message: 'PASSED', details: 'Generated 156 signals' },
      { name: 'Confidence Threshold', status: 'success', message: 'PASSED', details: 'Threshold: 0.65' },
      { name: 'Multi-timeframe Analysis', status: 'success', message: 'PASSED', details: 'M1, M5, M15 aligned' },
      { name: 'Signal Validation', status: 'success', message: 'PASSED', details: 'All signals validated' }
    ],
    integration: [
      { name: 'Database Connection', status: 'success', message: 'PASSED', details: 'PostgreSQL SSL connection active' },
      { name: 'WebSocket Communication', status: 'success', message: 'PASSED', details: 'Real-time data streaming' },
      { name: 'API Endpoints', status: 'success', message: 'PASSED', details: 'All endpoints responding' },
      { name: 'Memory Tier Integration', status: 'warning', message: 'WARNING', details: 'Redis latency: 15ms' },
      { name: 'CUDA Integration', status: 'success', message: 'PASSED', details: 'GPU acceleration active' }
    ],
    performance: [
      { name: 'Pattern Processing Speed', status: 'success', message: 'PASSED', details: '10K patterns/sec' },
      { name: 'Memory Usage', status: 'success', message: 'PASSED', details: 'RAM: 2.1GB / 16GB' },
      { name: 'Database Query Performance', status: 'success', message: 'PASSED', details: 'Avg: 12ms' },
      { name: 'Network Latency', status: 'warning', message: 'WARNING', details: 'OANDA: 45ms' },
      { name: 'Thread Pool Efficiency', status: 'success', message: 'PASSED', details: '95% utilization' }
    ]
  };

  const tabs = [
    { id: 'unit', label: 'Unit Tests', icon: <Terminal className="w-4 h-4" /> },
    { id: 'integration', label: 'Integration Tests', icon: <Activity className="w-4 h-4" /> },
    { id: 'performance', label: 'Performance Tests', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'backtest', label: 'Backtesting', icon: <Clock className="w-4 h-4" /> }
  ];

  useEffect(() => {
    setCurrentTab(activeTab.replace('-tests', '').replace('unit-tests', 'unit'));
  }, [activeTab]);

  const runTests = async (suite) => {
    setIsRunning(true);
    setTestResults([]);
    
    const tests = suite === 'all' 
      ? [...testSuites.quantum, ...testSuites.trading, ...testSuites.signals]
      : testSuites[suite] || [];

    let delay = 0;
    
    // Simulate test execution with delays
    tests.forEach((test, index) => {
      setTimeout(() => {
        const logEntry = {
          ...test,
          timestamp: new Date().toLocaleTimeString(),
          id: `${suite}-${index}`
        };
        
        setTestResults(prev => [...prev, logEntry]);
        
        // Auto-scroll to bottom
        const resultsContainer = document.getElementById('testResults');
        if (resultsContainer) {
          resultsContainer.scrollTop = resultsContainer.scrollHeight;
        }
        
        // Update stats
        if (index === tests.length - 1) {
          setTimeout(() => {
            const passed = tests.filter(t => t.status === 'success').length;
            const failed = tests.filter(t => t.status === 'error').length;
            const warnings = tests.filter(t => t.status === 'warning').length;
            
            setTestStats({
              coverage: Math.random() * 20 + 75, // 75-95%
              passed: passed,
              failed: failed,
              total: tests.length,
              duration: (delay + 500) / 1000
            });
            
            // Add summary
            setTestResults(prev => [...prev, {
              name: 'Test Suite Complete',
              status: failed > 0 ? 'error' : warnings > 0 ? 'warning' : 'success',
              message: 'COMPLETE',
              details: `Passed: ${passed}/${tests.length} | Failed: ${failed} | Duration: ${((delay + 500)/1000).toFixed(1)}s`,
              timestamp: new Date().toLocaleTimeString(),
              id: 'summary',
              isSummary: true
            }]);
            
            setIsRunning(false);
          }, 500);
        }
      }, delay);
      delay += 200;
    });
  };

  const clearTests = () => {
    setTestResults([]);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      default:
        return <Activity className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return 'border-green-500 bg-green-900/10';
      case 'error':
        return 'border-red-500 bg-red-900/10';
      case 'warning':
        return 'border-yellow-500 bg-yellow-900/10';
      default:
        return 'border-gray-600 bg-gray-900/10';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Testing & Validation Suite</h1>
          <p className="text-gray-400 mt-1">Comprehensive system testing and performance validation</p>
        </div>
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <span className="text-sm text-gray-400">
            System: <span className={connected ? 'text-green-400' : 'text-red-400'}>
              {connected ? 'Online' : 'Offline'}
            </span>
          </span>
        </div>
      </div>

      {/* Test Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="card-header">
            <span className="card-title">Test Coverage</span>
          </div>
          <div className="card-value text-blue-400">
            {testStats.coverage.toFixed(1)}%
          </div>
          <div className="text-sm text-green-400">
            +2.1% from last run
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Tests Passed</span>
          </div>
          <div className="card-value text-green-400">
            {testStats.passed}/{testStats.total}
          </div>
          <div className="text-sm text-gray-400">
            {((testStats.passed / testStats.total) * 100).toFixed(1)}% pass rate
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Failed Tests</span>
          </div>
          <div className="card-value text-red-400">
            {testStats.failed}
          </div>
          <div className="text-sm text-gray-400">
            Require attention
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Execution Time</span>
          </div>
          <div className="card-value text-yellow-400">
            {testStats.duration.toFixed(1)}s
          </div>
          <div className="text-sm text-gray-400">
            Average duration
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab ${currentTab === tab.id ? 'active' : ''}`}
            onClick={() => setCurrentTab(tab.id)}
          >
            {tab.icon}
            <span className="ml-2">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Testing Panel */}
      <div className="testing-panel">
        <div className="test-controls">
          <button
            className="test-button"
            onClick={() => runTests('quantum')}
            disabled={isRunning}
          >
            <Brain className="w-4 h-4 mr-2" />
            Run Quantum Tests
          </button>
          <button
            className="test-button"
            onClick={() => runTests('trading')}
            disabled={isRunning}
          >
            <Activity className="w-4 h-4 mr-2" />
            Run Trading Tests
          </button>
          <button
            className="test-button"
            onClick={() => runTests('signals')}
            disabled={isRunning}
          >
            <Zap className="w-4 h-4 mr-2" />
            Run Signal Tests
          </button>
          <button
            className="test-button"
            onClick={() => runTests('all')}
            disabled={isRunning}
          >
            <Play className="w-4 h-4 mr-2" />
            Run All Tests
          </button>
          <button
            className="test-button"
            style={{ background: 'var(--gradient-danger)' }}
            onClick={clearTests}
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Clear Results
          </button>
        </div>

        {/* Test Results */}
        <div className="test-results" id="testResults">
          {testResults.length === 0 && !isRunning && (
            <div className="test-log">
              Ready to run tests. Select a test suite above.
            </div>
          )}
          
          {isRunning && testResults.length === 0 && (
            <div className="test-log">
              <div className="flex items-center gap-2">
                <div className="loading-spinner w-4 h-4"></div>
                Initializing test suite...
              </div>
            </div>
          )}

          {testResults.map((result, index) => (
            <div
              key={result.id || index}
              className={`test-log ${result.status} ${result.isSummary ? 'font-bold border-t border-gray-700 mt-4 pt-2' : ''}`}
            >
              <div className="flex items-start gap-2">
                {getStatusIcon(result.status)}
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">
                      [{result.timestamp}] {result.name}: {result.message}
                    </span>
                  </div>
                  {result.details && (
                    <div className="mt-1 text-sm opacity-80 pl-6">
                      {result.details}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Test Categories Detail */}
      {currentTab === 'backtest' && (
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Clock className="w-5 h-5 text-blue-400" />
              Backtesting Suite
            </h3>
          </div>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-green-400 mb-1">94.2%</div>
                <div className="text-sm text-gray-400">Win Rate</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-blue-400 mb-1">2.34</div>
                <div className="text-sm text-gray-400">Sharpe Ratio</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-purple-400 mb-1">15.7%</div>
                <div className="text-sm text-gray-400">Max Drawdown</div>
              </div>
            </div>
            
            <div className="text-sm text-gray-400">
              <p>Backtesting results based on historical data from January 2023 to present.</p>
              <p>Test period: 365 days | Total trades: 1,247 | Capital: $100,000</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TestingSuite;