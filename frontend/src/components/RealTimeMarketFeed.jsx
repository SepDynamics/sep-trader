import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';
import { useSymbol } from '../context/SymbolContext';
import { apiClient } from '../services/api';
import { symbolInfo } from '../config/symbols';

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

  if (loading) return <div className="text-center py-10">Loading Market Data from Valkey...</div>;
  if (error) return <div className="text-center py-10 text-red-500">{error}</div>;
  if (priceHistory.length === 0) return <div className="text-center py-10">No data available for {selectedSymbol}. Ensure Valkey is seeded.</div>;

  return (
    <ResponsiveContainer width="100%" height={340}>
      <LineChart data={priceHistory} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(ts) => new Date(ts).toLocaleTimeString()} stroke="#94a3b8" />
        <YAxis dataKey="c" domain={['auto', 'auto']} tickFormatter={(value) => formatPrice(value)} stroke="#94a3b8" />
        <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #334155' }}
            labelStyle={{ color: '#cbd5e1' }}
            labelFormatter={(ts) => new Date(ts).toLocaleString()}
            formatter={(value) => [formatPrice(value), 'Price']}
         />
        <Line type="monotone" dataKey="c" stroke="#3b82f6" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default RealTimeMarketFeed;