// SEP Trading System - Metric Time Series Visualizer
// Advanced time series charts for entropy, stability, and coherence

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useManifold } from '../context/ManifoldContext';
import { useWebSocket } from '../context/WebSocketContext';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ReferenceLine, 
  ResponsiveContainer,
  Brush,
  ScatterChart,
  Scatter,
  Cell
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Zap, 
  Target, 
  Clock,
  BarChart3,
  Filter,
  Download
} from 'lucide-react';

const MetricTimeSeries = ({ metric = 'entropy', instrument = 'EUR_USD', timeWindow = '1h' }) => {
  const {
    processedIdentities,
    manifoldBands,
    ruptureEvents,
    getIdentityHistory
  } = useManifold();

  const { connected } = useWebSocket();
  
  const [viewMode, setViewMode] = useState('line'); // line, scatter, bands, correlation
  const [showRuptures, setShowRuptures] = useState(true);
  const [smoothingFactor, setSmoothingFactor] = useState(0);

  // Time window configurations
  const timeWindows = {
    '5m': 300000,
    '15m': 900000,
    '1h': 3600000,
    '4h': 14400000,
    '1d': 86400000
  };

  // Prepare time series data
  const timeSeriesData = useMemo(() => {
    const identityHistory = getIdentityHistory(instrument);
    if (!identityHistory.length) return [];

    const now = Date.now();
    const windowMs = timeWindows[timeWindow];
    const startTime = now - windowMs;

    // Filter data within time window
    const windowedData = identityHistory.filter(point => 
      point.timestamp >= startTime
    ).sort((a, b) => a.timestamp - b.timestamp);

    // Apply smoothing if requested
    if (smoothingFactor > 0 && windowedData.length > 2) {
      for (let i = 1; i < windowedData.length - 1; i++) {
        const prevValue = windowedData[i - 1][metric];
        const currValue = windowedData[i][metric];
        const nextValue = windowedData[i + 1][metric];
        
        // Simple moving average smoothing
        windowedData[i][`${metric}_smoothed`] = (
          prevValue * smoothingFactor +
          currValue * (1 - 2 * smoothingFactor) +
          nextValue * smoothingFactor
        );
      }
    }

    // Format for Recharts
    return windowedData.map(point => ({
      timestamp: point.timestamp,
      time: new Date(point.timestamp).toLocaleTimeString(),
      entropy: point.entropy,
      stability: point.stability,
      coherence: point.coherence,
      state: point.state,
      price: point.price,
      volume: point.volume,
      valkey_key: point.valkey_key,
      // Smoothed values
      entropy_smoothed: point.entropy_smoothed,
      stability_smoothed: point.stability_smoothed,
      coherence_smoothed: point.coherence_smoothed,
      // Derived metrics
      quantum_momentum: point.entropy * point.stability,
      phase_coherence: point.coherence * (1 - point.entropy),
      convergence_strength: point.stability * point.coherence
    }));
  }, [instrument, timeWindow, processedIdentities, metric, smoothingFactor]);

  // Rupture event markers within time window
  const windowedRuptures = useMemo(() => {
    const now = Date.now();
    const windowMs = timeWindows[timeWindow];
    const startTime = now - windowMs;

    return ruptureEvents.filter(rupture => 
      rupture.instrument === instrument && 
      rupture.timestamp >= startTime
    );
  }, [ruptureEvents, instrument, timeWindow]);

  // Band transition analysis
  const bandTransitions = useMemo(() => {
    const transitions = [];
    
    for (let i = 1; i < timeSeriesData.length; i++) {
      const prev = timeSeriesData[i - 1];
      const curr = timeSeriesData[i];
      
      // Detect entropy band transitions
      const prevBand = prev.entropy > 0.7 ? 'hot' : prev.entropy > 0.3 ? 'warm' : 'cold';
      const currBand = curr.entropy > 0.7 ? 'hot' : curr.entropy > 0.3 ? 'warm' : 'cold';
      
      if (prevBand !== currBand) {
        transitions.push({
          timestamp: curr.timestamp,
          from: prevBand,
          to: currBand,
          entropy_delta: curr.entropy - prev.entropy,
          y_position: curr[metric]
        });
      }
    }
    
    return transitions;
  }, [timeSeriesData, metric]);

  // Metric-specific configuration
  const metricConfig = {
    entropy: {
      color: '#ff6b6b',
      label: 'Entropy',
      domain: [0, 1],
      description: 'Measure of system disorder and unpredictability'
    },
    stability: {
      color: '#4ecdc4',
      label: 'Stability',
      domain: [0, 1],
      description: 'Pattern persistence and resistance to change'
    },
    coherence: {
      color: '#45b7d1',
      label: 'Coherence',
      domain: [0, 1],
      description: 'Phase alignment and signal clarity'
    },
    quantum_momentum: {
      color: '#96ceb4',
      label: 'Quantum Momentum',
      domain: [0, 1],
      description: 'Product of entropy and stability'
    },
    phase_coherence: {
      color: '#feca57',
      label: 'Phase Coherence',
      domain: [0, 1],
      description: 'Coherence weighted by order'
    },
    convergence_strength: {
      color: '#ff9ff3',
      label: 'Convergence Strength',
      domain: [0, 1],
      description: 'Product of stability and coherence'
    }
  };

  const config = metricConfig[metric];

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-sm">
          <div className="text-gray-400 mb-2">{new Date(data.timestamp).toLocaleString()}</div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-gray-300">{config.label}:</span>
              <span className="font-medium" style={{ color: config.color }}>
                {payload[0].value.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">State:</span>
              <span className="font-medium text-purple-400">{data.state}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Key:</span>
              <span className="font-mono text-xs text-gray-400">{data.valkey_key.slice(-8)}</span>
            </div>
            {data.price && (
              <div className="flex justify-between">
                <span className="text-gray-300">Price:</span>
                <span className="font-medium text-green-400">{data.price.toFixed(5)}</span>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  // Export data function
  const exportData = () => {
    const dataStr = JSON.stringify(timeSeriesData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `${instrument}_${metric}_${timeWindow}_${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-xl font-semibold flex items-center gap-2">
            <Activity className="w-5 h-5" style={{ color: config.color }} />
            {config.label} Time Series
          </h3>
          <p className="text-sm text-gray-400 mt-1">{config.description}</p>
          <div className="text-xs text-gray-500 mt-1">
            {instrument} • {timeWindow} window • {timeSeriesData.length} data points
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* View Mode */}
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1 text-sm"
          >
            <option value="line">Line Chart</option>
            <option value="scatter">Scatter Plot</option>
            <option value="bands">Band Analysis</option>
            <option value="correlation">Correlation</option>
          </select>

          {/* Smoothing */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-400">Smooth:</span>
            <input
              type="range"
              min="0"
              max="0.3"
              step="0.05"
              value={smoothingFactor}
              onChange={(e) => setSmoothingFactor(parseFloat(e.target.value))}
              className="w-16"
            />
            <span className="text-xs text-gray-500 w-8">{(smoothingFactor * 100).toFixed(0)}%</span>
          </div>

          {/* Controls */}
          <button
            onClick={() => setShowRuptures(!showRuptures)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              showRuptures 
                ? 'bg-red-600 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Ruptures
          </button>

          <button
            onClick={exportData}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded transition-colors"
            title="Export Data"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          {viewMode === 'scatter' ? (
            <ScatterChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                type="number"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString()}
                stroke="#9ca3af"
                fontSize={12}
              />
              <YAxis 
                domain={config.domain}
                stroke="#9ca3af"
                fontSize={12}
              />
              <Tooltip content={<CustomTooltip />} />
              <Scatter 
                dataKey={metric}
                fill={config.color}
              >
                {timeSeriesData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={
                      entry.state === 'converged' ? '#22c55e' :
                      entry.state === 'divergent' ? '#ef4444' :
                      config.color
                    }
                  />
                ))}
              </Scatter>
            </ScatterChart>
          ) : (
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="time"
                stroke="#9ca3af"
                fontSize={12}
                interval="preserveStartEnd"
              />
              <YAxis 
                domain={config.domain}
                stroke="#9ca3af"
                fontSize={12}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              
              {/* Main metric line */}
              <Line
                type="monotone"
                dataKey={metric}
                stroke={config.color}
                strokeWidth={2}
                dot={false}
                name={config.label}
              />
              
              {/* Smoothed line if enabled */}
              {smoothingFactor > 0 && (
                <Line
                  type="monotone"
                  dataKey={`${metric}_smoothed`}
                  stroke={config.color}
                  strokeWidth={3}
                  strokeOpacity={0.7}
                  dot={false}
                  name={`${config.label} (Smoothed)`}
                  strokeDasharray="5 5"
                />
              )}

              {/* Band reference lines */}
              {metric === 'entropy' && (
                <>
                  <ReferenceLine y={0.7} stroke="#ff6b6b" strokeDasharray="2 2" />
                  <ReferenceLine y={0.3} stroke="#ffaa44" strokeDasharray="2 2" />
                </>
              )}
              
              {/* Convergence threshold */}
              {(metric === 'stability' || metric === 'coherence') && (
                <ReferenceLine y={0.8} stroke="#22c55e" strokeDasharray="2 2" />
              )}

              {/* Rupture event markers */}
              {showRuptures && windowedRuptures.map((rupture, idx) => (
                <ReferenceLine
                  key={idx}
                  x={new Date(rupture.timestamp).toLocaleTimeString()}
                  stroke="#ff4444"
                  strokeWidth={2}
                  strokeOpacity={0.6}
                />
              ))}

              <Brush 
                dataKey="time" 
                height={30} 
                stroke={config.color}
                fill="transparent"
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Statistics */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-xs text-gray-400">Current Value</div>
          <div className="text-lg font-semibold" style={{ color: config.color }}>
            {timeSeriesData.length > 0 ? timeSeriesData[timeSeriesData.length - 1][metric].toFixed(4) : '0.0000'}
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-xs text-gray-400">Average</div>
          <div className="text-lg font-semibold text-gray-200">
            {timeSeriesData.length > 0 ? 
              (timeSeriesData.reduce((sum, d) => sum + d[metric], 0) / timeSeriesData.length).toFixed(4) : 
              '0.0000'
            }
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-xs text-gray-400">Volatility</div>
          <div className="text-lg font-semibold text-yellow-400">
            {timeSeriesData.length > 1 ? 
              Math.sqrt(
                timeSeriesData.reduce((sum, d, i, arr) => {
                  if (i === 0) return 0;
                  const diff = d[metric] - arr[i-1][metric];
                  return sum + diff * diff;
                }, 0) / (timeSeriesData.length - 1)
              ).toFixed(4) : 
              '0.0000'
            }
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-xs text-gray-400">Trend</div>
          <div className="text-lg font-semibold flex items-center gap-1">
            {timeSeriesData.length > 1 ? (
              timeSeriesData[timeSeriesData.length - 1][metric] > timeSeriesData[0][metric] ? (
                <>
                  <TrendingUp className="w-4 h-4 text-green-400" />
                  <span className="text-green-400">Up</span>
                </>
              ) : (
                <>
                  <TrendingDown className="w-4 h-4 text-red-400" />
                  <span className="text-red-400">Down</span>
                </>
              )
            ) : (
              <span className="text-gray-400">N/A</span>
            )}
          </div>
        </div>
      </div>

      {/* Band Transitions */}
      {bandTransitions.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <Target className="w-4 h-4" />
            Band Transitions ({bandTransitions.length})
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
            {bandTransitions.slice(-4).map((transition, idx) => (
              <div key={idx} className="bg-gray-800 rounded p-2 flex items-center justify-between">
                <span className="text-gray-400">
                  {new Date(transition.timestamp).toLocaleTimeString()}
                </span>
                <span className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded text-xs ${
                    transition.from === 'hot' ? 'bg-red-900 text-red-300' :
                    transition.from === 'warm' ? 'bg-yellow-900 text-yellow-300' :
                    'bg-green-900 text-green-300'
                  }`}>
                    {transition.from}
                  </span>
                  →
                  <span className={`px-2 py-1 rounded text-xs ${
                    transition.to === 'hot' ? 'bg-red-900 text-red-300' :
                    transition.to === 'warm' ? 'bg-yellow-900 text-yellow-300' :
                    'bg-green-900 text-green-300'
                  }`}>
                    {transition.to}
                  </span>
                  <span className="text-gray-400">
                    ({transition.entropy_delta > 0 ? '+' : ''}{transition.entropy_delta.toFixed(3)})
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricTimeSeries;