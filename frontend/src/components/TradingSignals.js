import React, { useState } from 'react';
import { useWebSocket } from '../context/WebSocketContext';

const TradingSignals = () => {
  const { connected, tradingSignals } = useWebSocket();
  const [filter, setFilter] = useState('all');

  const formatCurrency = (value) => {
    if (value === null || value === undefined) return '--';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const filteredSignals = tradingSignals.filter(signal => {
    if (filter === 'all') return true;
    return signal.type?.toLowerCase() === filter;
  });

  return (
    <div className="trading-signals">
      <div className="signals-header">
        <h1>Trading Signals</h1>
        <div className="signals-controls">
          <div className="filter-buttons">
            <button 
              className={filter === 'all' ? 'active' : ''} 
              onClick={() => setFilter('all')}
            >
              All
            </button>
            <button 
              className={filter === 'buy' ? 'active' : ''} 
              onClick={() => setFilter('buy')}
            >
              Buy
            </button>
            <button 
              className={filter === 'sell' ? 'active' : ''} 
              onClick={() => setFilter('sell')}
            >
              Sell
            </button>
          </div>
          <div className="connection-status">
            <span className={connected ? 'connected' : 'disconnected'}>
              {connected ? '🟢 Live Signals' : '🔴 Offline'}
            </span>
          </div>
        </div>
      </div>

      <div className="signals-list">
        {filteredSignals.length > 0 ? (
          filteredSignals.map((signal, index) => (
            <div key={index} className="signal-card">
              <div className="signal-header">
                <span className={`signal-type ${signal.type?.toLowerCase()}`}>
                  {signal.type}
                </span>
                <span className="signal-time">
                  {signal.timestamp ? new Date(signal.timestamp).toLocaleString() : '--'}
                </span>
              </div>
              <div className="signal-body">
                <div className="signal-details">
                  <div className="detail-item">
                    <label>Symbol:</label>
                    <span className="symbol">{signal.symbol || '--'}</span>
                  </div>
                  <div className="detail-item">
                    <label>Price:</label>
                    <span className="price">{formatCurrency(signal.price)}</span>
                  </div>
                  <div className="detail-item">
                    <label>Confidence:</label>
                    <span className={`confidence confidence-${Math.floor((signal.confidence || 0) * 10)}`}>
                      {((signal.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                {signal.reason && (
                  <div className="signal-reason">
                    <label>Reason:</label>
                    <p>{signal.reason}</p>
                  </div>
                )}
              </div>
            </div>
          ))
        ) : (
          <div className="no-signals">
            <p>No {filter !== 'all' ? filter : ''} signals available</p>
            <p>Connect to WebSocket service to receive live trading signals</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradingSignals;