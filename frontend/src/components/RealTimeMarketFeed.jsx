// File: /sep/frontend/src/components/RealTimeMarketFeed.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';
import { useSymbol } from '../context/SymbolContext';
import { apiClient } from '../services/api'; // Use your existing apiClient
import { symbolInfo } from '../config/symbols';

const RealTimeMarketFeed = ({ hours = 48 }) => {
  const { selectedSymbol } = useSymbol();
  const [priceHistory, setPriceHistory] = useState([]);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch historical data on component mount or when symbol changes
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const to = Date.now();
        const from = to - hours * 3600 * 1000;
        
        // Use your apiClient to fetch data from the new endpoint
        const response = await apiClient.request(`/api/market-data?instrument=${selectedSymbol}&from=${from}&to=${to}`);
        
        const data = (response.rows || []).map(candle => ({
          t: candle.t, // timestamp
          c: candle.c  // close price
        }));
        
        setPriceHistory(data);
        if (data.length > 0) {
          setCurrentPrice(data[data.length - 1].c);
        }
      } catch (err) {
        setError('Failed to load market data.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [selectedSymbol, hours]);

  // (WebSocket logic for live updates can be added here later)

  const formatPrice = (price) => {
    if (price === null || price === undefined) return '--';
    const symbolPrecision = symbolInfo[selectedSymbol.replace('/', '_')]?.precision || 5;
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: symbolPrecision,
      maximumFractionDigits: symbolPrecision
    }).format(price);
  };

  if (loading) return <div className="text-center py-10">Loading Market Data...</div>;
  if (error) return <div className="text-center py-10 text-red-500">{error}</div>;
  if (priceHistory.length === 0) return <div className="text-center py-10">No data available for {selectedSymbol}.</div>;

  return (
    <div className="real-time-market-feed">
      <div className="price-chart-container">
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={priceHistory} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(ts) => new Date(ts).toLocaleTimeString()} stroke="#94a3b8" />
            <YAxis dataKey="c" domain={['auto', 'auto']} tickFormatter={(value) => formatPrice(value)} stroke="#94a3b8" />
            <Tooltip
                labelFormatter={(ts) => new Date(ts).toLocaleString()}
                formatter={(v) => [formatPrice(v), 'Price']}
             />
            <Line type="monotone" dataKey="c" stroke="#3b82f6" dot={false} strokeWidth={1.8} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default RealTimeMarketFeed;
