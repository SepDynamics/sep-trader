// SEP Engine - OANDA Candle Chart Component
// Displays real-time OANDA candle data from Valkey server with 2-week history

import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Brush,
  Area,
  AreaChart
} from 'recharts';
import { useWebSocket } from '../context/WebSocketContext';
import { apiClient } from '../services/api';
import { symbolInfo } from '../config/symbols';
import { useSymbol } from '../context/SymbolContext';
import { TrendingUp, Database, Clock, BarChart3 } from 'lucide-react';

const OandaCandleChart = ({ height = 400, showControls = true }) => {
  const [candleData, setCandleData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('line'); // 'line' or 'area'
  const [timeRange, setTimeRange] = useState('2weeks'); // '1week', '2weeks'

  const { valkeyMetrics, connected } = useWebSocket();
  const { selectedSymbol, setSelectedSymbol, symbols } = useSymbol();

  // Fetch candle data from Valkey
  const fetchCandleData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Calculate time range (2 weeks = 14 days)
      const now = Date.now();
      const twoWeeksAgo = now - (14 * 24 * 60 * 60 * 1000);
      
      // First fetch fresh data from OANDA if needed
      await apiClient.fetchCandleData([selectedSymbol]);

      // Then get stored candles from Valkey
      const response = await apiClient.getStoredCandles(selectedSymbol);
      
      if (response && response.candles) {
        // Transform OANDA candle format to chart format
        const transformedData = response.candles
          .filter(candle => {
            const timestamp = new Date(candle.time || candle.timestamp).getTime();
            return timestamp >= twoWeeksAgo && timestamp <= now;
          })
          .map((candle, index) => ({
            timestamp: new Date(candle.time || candle.timestamp).getTime(),
            time: new Date(candle.time || candle.timestamp).toISOString(),
            open: parseFloat(candle.mid?.o || candle.o || candle.open || 0),
            high: parseFloat(candle.mid?.h || candle.h || candle.high || 0),
            low: parseFloat(candle.mid?.l || candle.l || candle.low || 0),
            close: parseFloat(candle.mid?.c || candle.c || candle.close || 0),
            volume: parseInt(candle.volume || 0),
            // Add Valkey key identifier (timestamp-based)
            valkeyKey: `oanda:${selectedSymbol}:${new Date(candle.time || candle.timestamp).getTime()}`,
            index
          }))
          .sort((a, b) => a.timestamp - b.timestamp);
        
        setCandleData(transformedData);
      } else {
        // No data available from Valkey server
        setError(`No candle data available for ${selectedSymbol} in Valkey server`);
        setCandleData([]);
      }
      
    } catch (err) {
      console.error('Failed to fetch candle data:', err);
      setError(`Failed to load candle data: ${err.message}`);
      setCandleData([]);
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol]);


  // Load data on component mount and instrument change
  useEffect(() => {
    fetchCandleData();
  }, [fetchCandleData]);

  // Auto-refresh every 60 seconds when connected
  useEffect(() => {
    if (!connected) return;
    
    const interval = setInterval(() => {
      fetchCandleData();
    }, 60000);
    
    return () => clearInterval(interval);
  }, [connected, fetchCandleData]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-900/95 backdrop-blur-sm border border-gray-600/50 rounded-lg p-4 shadow-lg">
          <p className="text-sm text-gray-300 font-semibold mb-2">
            {new Date(label).toLocaleString()}
          </p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Open:</span>
              <span className="text-blue-400 font-mono">{data.open?.toFixed(5)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">High:</span>
              <span className="text-green-400 font-mono">{data.high?.toFixed(5)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Low:</span>
              <span className="text-red-400 font-mono">{data.low?.toFixed(5)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Close:</span>
              <span className="text-yellow-400 font-mono font-bold">{data.close?.toFixed(5)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Volume:</span>
              <span className="text-purple-400">{data.volume?.toLocaleString()}</span>
            </div>
            <div className="mt-2 pt-2 border-t border-gray-700">
              <span className="text-xs text-gray-500 font-mono">{data.valkeyKey}</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  const formatPrice = (price) => {
    const precision = symbolInfo[selectedSymbol]?.precision || 5;
    return price?.toFixed(precision) || '0.00000';
  };

  const getCurrentPrice = () => {
    if (candleData.length === 0) return null;
    return candleData[candleData.length - 1]?.close || 0;
  };

  const getPriceChange = () => {
    if (candleData.length < 2) return { change: 0, percentage: 0 };
    const current = candleData[candleData.length - 1]?.close || 0;
    const previous = candleData[candleData.length - 2]?.close || 0;
    const change = current - previous;
    const percentage = previous > 0 ? (change / previous) * 100 : 0;
    return { change, percentage };
  };

  if (loading) {
    return (
      <div className="bg-gray-900/50 rounded-lg border border-gray-700/50 p-6">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
          <span className="ml-3 text-gray-400">Loading OANDA candle data...</span>
        </div>
      </div>
    );
  }

  const currentPrice = getCurrentPrice();
  const { change, percentage } = getPriceChange();
  const isPositive = change >= 0;

  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-700/50 p-4">
      {showControls && (
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-400" />
              <h3 className="text-lg font-semibold text-white">OANDA Candle Data</h3>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center gap-2 px-3 py-1 bg-gray-800 rounded-lg">
              <Database className={`w-4 h-4 ${connected ? 'text-green-400' : 'text-red-400'}`} />
              <span className="text-sm text-gray-300">
                Valkey: {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Instrument Selector */}
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="bg-gray-800 border border-gray-600 rounded px-3 py-1 text-sm text-white"
            >
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>
                  {symbolInfo[symbol]?.name || symbol}
                </option>
              ))}
            </select>

            {/* View Mode Toggle */}
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value)}
              className="bg-gray-800 border border-gray-600 rounded px-3 py-1 text-sm text-white"
            >
              <option value="line">Line Chart</option>
              <option value="area">Area Chart</option>
            </select>

            {/* Refresh Button */}
            <button
              onClick={fetchCandleData}
              disabled={loading}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-sm text-white transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      )}

      {/* Price Display */}
      <div className="flex items-center justify-between mb-4 p-3 bg-gray-800/50 rounded-lg">
        <div className="flex items-center gap-4">
          <div>
            <div className="text-sm text-gray-400">{symbolInfo[selectedSymbol]?.name || selectedSymbol}</div>
            <div className="text-2xl font-bold text-white">
              {formatPrice(currentPrice)}
            </div>
          </div>
          <div className={`flex items-center gap-1 px-2 py-1 rounded ${isPositive ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
            <TrendingUp className={`w-4 h-4 transform ${isPositive ? '' : 'rotate-180'}`} />
            <span className="text-sm font-mono">
              {isPositive ? '+' : ''}{formatPrice(change)} ({percentage.toFixed(2)}%)
            </span>
          </div>
        </div>
        
        <div className="text-right">
          <div className="text-sm text-gray-400 flex items-center gap-1">
            <Clock className="w-4 h-4" />
            Last Update: {candleData.length > 0 ? new Date(candleData[candleData.length - 1]?.timestamp).toLocaleTimeString() : 'N/A'}
          </div>
          <div className="text-sm text-gray-500">
            Data Points: {candleData.length.toLocaleString()}
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
          <div className="text-red-400 text-sm">{error}</div>
        </div>
      )}

      {/* Chart Display */}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          {viewMode === 'area' ? (
            <AreaChart data={candleData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="timestamp" 
                type="number" 
                domain={['dataMin', 'dataMax']} 
                tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()} 
                stroke="#94a3b8" 
                tick={{ fill: '#94a3b8', fontSize: 12 }}
              />
              <YAxis 
                domain={['auto', 'auto']} 
                tickFormatter={(value) => formatPrice(value)} 
                stroke="#94a3b8"
                orientation="right"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                mirror
              />
              <Tooltip content={<CustomTooltip />} />
              <Area 
                type="monotone" 
                dataKey="close" 
                stroke="#3b82f6" 
                strokeWidth={2}
                fill="url(#priceGradient)"
                dot={false}
              />
              <Brush 
                dataKey="timestamp" 
                height={30} 
                stroke="#3b82f6" 
                fill="#1e293b" 
                tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()} 
              />
            </AreaChart>
          ) : (
            <LineChart data={candleData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="timestamp" 
                type="number" 
                domain={['dataMin', 'dataMax']} 
                tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()} 
                stroke="#94a3b8" 
                tick={{ fill: '#94a3b8', fontSize: 12 }}
              />
              <YAxis 
                domain={['auto', 'auto']} 
                tickFormatter={(value) => formatPrice(value)} 
                stroke="#94a3b8"
                orientation="right"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                mirror
              />
              <Tooltip content={<CustomTooltip />} />
              <Line 
                type="monotone" 
                dataKey="close" 
                stroke="#3b82f6" 
                strokeWidth={2} 
                dot={false} 
              />
              <Brush 
                dataKey="timestamp" 
                height={30} 
                stroke="#3b82f6" 
                fill="#1e293b" 
                tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()} 
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Statistics Footer */}
      <div className="mt-4 grid grid-cols-4 gap-4 text-center">
        <div className="p-2 bg-gray-800/50 rounded">
          <div className="text-xs text-gray-400">24h High</div>
          <div className="text-sm font-mono text-green-400">
            {candleData.length > 0 ? formatPrice(Math.max(...candleData.slice(-24).map(d => d.high))) : '--'}
          </div>
        </div>
        <div className="p-2 bg-gray-800/50 rounded">
          <div className="text-xs text-gray-400">24h Low</div>
          <div className="text-sm font-mono text-red-400">
            {candleData.length > 0 ? formatPrice(Math.min(...candleData.slice(-24).map(d => d.low))) : '--'}
          </div>
        </div>
        <div className="p-2 bg-gray-800/50 rounded">
          <div className="text-xs text-gray-400">Avg Volume</div>
          <div className="text-sm font-mono text-purple-400">
            {candleData.length > 0 ? Math.floor(candleData.reduce((sum, d) => sum + d.volume, 0) / candleData.length).toLocaleString() : '--'}
          </div>
        </div>
        <div className="p-2 bg-gray-800/50 rounded">
          <div className="text-xs text-gray-400">Data Range</div>
          <div className="text-sm font-mono text-blue-400">2 weeks</div>
        </div>
      </div>
    </div>
  );
};

export default OandaCandleChart;