import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

const MarketData = () => {
  const { connected, marketData } = useWebSocket();
  const [selectedSymbol, setSelectedSymbol] = useState('EUR/USD');

  const formatPrice = (value) => {
    if (value === null || value === undefined) return '--';
    return new Intl.NumberFormat('en-US', {
      style: 'decimal',
      minimumFractionDigits: 4,
      maximumFractionDigits: 5
    }).format(value);
  };

  const formatCurrency = (value) => {
    if (value === null || value === undefined) return '--';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const formatPercentage = (value) => {
    if (value === null || value === undefined) return '--';
    const formatted = (value * 100).toFixed(2);
    const sign = value >= 0 ? '+' : '';
    return `${sign}${formatted}%`;
  };

  return (
    <div className="market-data">
      <div className="market-header">
        <h1>Market Data</h1>
        <div className="connection-status">
          <span className={connected ? 'connected' : 'disconnected'}>
            {connected ? '🟢 Live Data' : '🔴 Offline'}
          </span>
        </div>
      </div>

      <div className="market-grid">
        {Object.entries(marketData).map(([symbol, data]) => (
          <div key={symbol} className="market-card">
            <h3>{symbol}</h3>
            <div className="price-info">
              <div className="current-price">{formatPrice(data.price)}</div>
              <div className={`price-change ${(data.change || 0) >= 0 ? 'positive' : 'negative'}`}>
                {formatPercentage(data.change / 100)} ({formatPrice(data.change)})
              </div>
            </div>
            <div className="market-details">
              <div><label>Volume:</label> <span>{data.volume?.toLocaleString() || '--'}</span></div>
              <div><label>High:</label> <span>{formatPrice(data.high)}</span></div>
              <div><label>Low:</label> <span>{formatPrice(data.low)}</span></div>
              <div><label>Spread:</label> <span>{data.spread ? formatPrice(data.spread) : '--'}</span></div>
            </div>
          </div>
        ))}
        
        {Object.keys(marketData).length === 0 && (
          <div className="no-data">
            <p>No market data available</p>
            <p>Connect to WebSocket service to see live market data</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MarketData;