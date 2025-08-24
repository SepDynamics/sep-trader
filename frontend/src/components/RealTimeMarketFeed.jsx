import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Brush } from 'recharts';
import { useSymbol } from '../context/SymbolContext';
import { apiClient } from '../services/api';
import { symbolInfo } from '../config/symbols';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-gray-800/80 backdrop-blur-sm border border-gray-600/50 rounded-lg p-3 shadow-lg">
        <p className="text-sm text-gray-300 font-semibold">{new Date(label).toLocaleString()}</p>
        <p className="text-lg font-bold text-blue-400">{`Price: ${payload[0].value.toFixed(5)}`}</p>
        <div className="text-xs text-gray-400 mt-1">
          <span>O: {data.o.toFixed(5)}</span> | <span>H: {data.h.toFixed(5)}</span> | <span>L: {data.l.toFixed(5)}</span>
        </div>
      </div>
    );
  }
  return null;
};

const RealTimeMarketFeed = ({ hours = 48 }) => {
  const { selectedSymbol } = useSymbol();
  const [priceHistory, setPriceHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const to = Date.now();
        const from = to - hours * 3600 * 1000;
        const response = await apiClient.getMarketData(selectedSymbol, from, to);
        const data = (response.rows || []).filter(candle => candle.t && candle.c);
        setPriceHistory(data);
        if (data.length === 0) {
            console.warn(`No market data returned for ${selectedSymbol}`);
        }
      } catch (err) {
        setError(`Failed to load market data for ${selectedSymbol}. Is the backend running?`);
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    if (selectedSymbol) {
      fetchData();
    }
  }, [selectedSymbol, hours]);

  const formatPrice = (price) => {
    if (price === null || price === undefined) return '--';
    const symbolPrecision = symbolInfo[selectedSymbol]?.precision || 5;
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: symbolPrecision,
      maximumFractionDigits: symbolPrecision
    }).format(price);
  };

  if (loading) return <div className="h-[400px] flex items-center justify-center text-gray-400">Loading Market Data...</div>;
  if (error) return <div className="h-[400px] flex items-center justify-center text-red-400 bg-red-900/20 rounded-lg">{error}</div>;
  if (priceHistory.length === 0) return <div className="h-[400px] flex items-center justify-center text-yellow-400">No data available for {selectedSymbol}. Ensure Valkey is seeded.</div>;

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={priceHistory} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="t" 
          type="number" 
          domain={['dataMin', 'dataMax']} 
          tickFormatter={(ts) => new Date(ts).toLocaleTimeString()} 
          stroke="#94a3b8" 
          tick={{ fill: '#94a3b8', fontSize: 12 }}
        />
        <YAxis 
          dataKey="c" 
          type="number" 
          domain={['auto', 'auto']} 
          tickFormatter={(value) => formatPrice(value)} 
          stroke="#94a3b8"
          orientation="right"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          mirror
        />
        <Tooltip content={<CustomTooltip />} />
        <Line type="monotone" dataKey="c" stroke="#3b82f6" strokeWidth={2} dot={false} />
        <Brush dataKey="t" height={30} stroke="#3b82f6" fill="#1e293b" tickFormatter={(ts) => new Date(ts).toLocaleDateString()} />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default RealTimeMarketFeed;
