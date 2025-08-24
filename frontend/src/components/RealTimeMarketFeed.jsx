import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';
import { useWebSocket } from '../context/WebSocketContext';
import { useSymbol } from '../context/SymbolContext';
import { symbols, symbolInfo } from '../config/symbols';

const RealTimeMarketFeed = ({ hours = 48 }) => {
  const { marketData } = useWebSocket();
  const { selectedSymbol } = useSymbol();
  
  // Convert symbol format for display (EUR_USD -> EUR/USD)
  const formatSymbolForDisplay = (symbol) => {
    return symbol.replace('_', '/');
  };
  
  // Convert symbol format for backend (EUR/USD -> EUR_USD)
  const formatSymbolForBackend = (symbol) => {
    return symbol.replace('/', '_');
  };
  
  // Ensure we're using the correct format for backend
  const selectedSymbolBackend = formatSymbolForBackend(selectedSymbol);
  
  // Get price history for the selected symbol from WebSocket data
  const priceHistory = useMemo(() => {
    const history = [];
    
    // Get the market data for the selected symbol
    const symbolData = marketData[selectedSymbolBackend];
    
    if (symbolData && symbolData.history) {
      // Use the history from WebSocket data
      return symbolData.history.map(item => ({
        t: new Date(item.timestamp).getTime(),
        c: item.price
      }));
    }
    
    return history;
  }, [marketData, selectedSymbolBackend]);
  
  // Get current price for the selected symbol
  const currentPrice = useMemo(() => {
    const symbolData = marketData[selectedSymbolBackend];
    return symbolData ? symbolData.price : null;
  }, [marketData, selectedSymbolBackend]);
  
  // Format price with appropriate precision
  const formatPrice = (price) => {
    if (price === null || price === undefined) return '--';
    const symbolPrecision = symbolInfo[selectedSymbol.replace('/', '_')]?.precision || 5;
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: symbolPrecision,
      maximumFractionDigits: symbolPrecision
    }).format(price);
  };
  
  // Format percentage change
  const formatPercentage = (value) => {
    if (value === null || value === undefined) return '--';
    const formatted = (value * 100).toFixed(2);
    const sign = value >= 0 ? '+' : '';
    return `${sign}${formatted}%`;
  };
  
  return (
    <div className="real-time-market-feed">
      <div className="market-header">
        <div className="symbol-info">
          <h3>{formatSymbolForDisplay(selectedSymbol)}</h3>
          <div className="price-display">
            <span className="current-price">{formatPrice(currentPrice)}</span>
            {marketData[selectedSymbolBackend]?.change && (
              <span className={`price-change ${marketData[selectedSymbolBackend].change >= 0 ? 'positive' : 'negative'}`}>
                {formatPercentage(marketData[selectedSymbolBackend].change)}
              </span>
            )}
          </div>
        </div>
      </div>
      
      <div className="price-chart-container">
        {priceHistory.length > 0 ? (
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={priceHistory} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="t"
                domain={['auto', 'auto']}
                type="number"
                tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
              />
              <YAxis 
                dataKey="c" 
                domain={['auto', 'auto']} 
                tickFormatter={(value) => formatPrice(value)}
              />
              <Tooltip
                labelFormatter={(ts) => new Date(ts).toLocaleString()}
                formatter={(v) => [formatPrice(v), 'Price']}
              />
              <Line 
                type="monotone" 
                dataKey="c" 
                dot={false} 
                strokeWidth={1.8} 
                stroke="#1e40af" // Blue color to match the theme
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="no-data">
            <p>Waiting for real-time market data...</p>
            <p>Ensure WebSocket connection is active and Valkey is populated with {formatSymbolForDisplay(selectedSymbol)} data.</p>
          </div>
        )}
      </div>
      
      <div className="market-details">
        <div className="detail-item">
          <span className="label">24h High:</span>
          <span className="value">
            {marketData[selectedSymbolBackend]?.high ?
              formatPrice(marketData[selectedSymbolBackend].high) : '--'}
          </span>
        </div>
        <div className="detail-item">
          <span className="label">24h Low:</span>
          <span className="value">
            {marketData[selectedSymbolBackend]?.low ?
              formatPrice(marketData[selectedSymbolBackend].low) : '--'}
          </span>
        </div>
        <div className="detail-item">
          <span className="label">Volume:</span>
          <span className="value">
            {marketData[selectedSymbolBackend]?.volume ?
              marketData[selectedSymbolBackend].volume.toLocaleString() : '--'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default RealTimeMarketFeed;