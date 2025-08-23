import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import { useSymbol } from '../context/SymbolContext';
import { symbols } from '../config/symbols';
import { submitOrder } from '../services/api';
import { buildOrder } from '../utils/order';
import { useConfig } from '../context/ConfigContext';
import '../styles/TradingPanel.css';

const TradingPanel: React.FC = () => {
  const { connected, marketData, tradingSignals } = useWebSocket();
  const { selectedSymbol, setSelectedSymbol } = useSymbol();
  const { config } = useConfig();
  const [orderType, setOrderType] = useState<string>('market');
  const [quantity, setQuantity] = useState<number>(10000);
  const [price, setPrice] = useState<string>('');
  const [side, setSide] = useState<string>('buy');
  const [loading, setLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>('');

  useEffect(() => {
    const defaultQty = config.trading?.default_quantity;
    if (typeof defaultQty === 'number') {
      setQuantity(defaultQty);
    }
  }, [config.trading?.default_quantity]);

  const handleSubmitOrder = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');

    try {
      const orderData = buildOrder({
        symbol: selectedSymbol,
        side,
        quantity: quantity,
        type: orderType,
        price: orderType === 'limit' ? parseFloat(price) : undefined,
      });

      const response = await submitOrder(orderData);
      setMessage(`Order submitted successfully: ${response.message || 'Order processed'}`);

      const defaultQty = config.trading?.default_quantity;
      if (typeof defaultQty === 'number') {
        setQuantity(defaultQty);
      }
      setPrice('');
    } catch (error: any) {
      setMessage(`Order failed: ${error?.response?.data?.message || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getCurrentPrice = (symbol: string) => {
    return marketData[symbol]?.price || 0;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'decimal',
      minimumFractionDigits: 4,
      maximumFractionDigits: 5
    }).format(value);
  };

  const getEstimatedTotal = () => {
    const currentPrice = getCurrentPrice(selectedSymbol);
    const orderPrice = orderType === 'limit' ? parseFloat(price) || currentPrice : currentPrice;
    return orderPrice * quantity;
  };

  return (
    <div className="trading-panel">
      <div className="panel-header">
        <h1>Trading Panel</h1>
        <div className="connection-indicator">
          <span className={`indicator ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '●' : '○'}
          </span>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>

      <div className="trading-layout">
        <div className="order-form-section">
          <div className="card">
            <div className="card-header">
              <h3>Place Order</h3>
            </div>
            <div className="card-content">
              <form onSubmit={handleSubmitOrder} className="order-form">
                <div className="form-row">
                  <div className="form-group">
                    <label>Symbol</label>
                    <select
                      value={selectedSymbol}
                      onChange={(e) => setSelectedSymbol(e.target.value)}
                      className="form-control"
                    >
                      {symbols.map(symbol => (
                        <option key={symbol} value={symbol}>{symbol}</option>
                      ))}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Side</label>
                    <div className="button-group">
                      <button
                        type="button"
                        className={`btn ${side === 'buy' ? 'btn-success active' : 'btn-outline'}`}
                        onClick={() => setSide('buy')}
                      >
                        Buy
                      </button>
                      <button
                        type="button"
                        className={`btn ${side === 'sell' ? 'btn-danger active' : 'btn-outline'}`}
                        onClick={() => setSide('sell')}
                      >
                        Sell
                      </button>
                    </div>
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>Order Type</label>
                    <select
                      value={orderType}
                      onChange={(e) => setOrderType(e.target.value)}
                      className="form-control"
                    >
                      <option value="market">Market</option>
                      <option value="limit">Limit</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Quantity (Units)</label>
                    <input
                      type="number"
                      value={quantity}
                      onChange={(e) => setQuantity(parseInt(e.target.value))}
                      className="form-control"
                      min="1000"
                      step="1000"
                      required
                    />
                  </div>
                </div>

                {orderType === 'limit' && (
                  <div className="form-row">
                    <div className="form-group">
                      <label>Price</label>
                      <input
                        type="number"
                        value={price}
                        onChange={(e) => setPrice(e.target.value)}
                        className="form-control"
                        step="0.01"
                        min="0"
                        placeholder="Enter limit price"
                        required
                      />
                    </div>
                  </div>
                )}

                <div className="order-summary">
                  <div className="summary-row">
                    <span>Current Price:</span>
                    <span className="price">{formatCurrency(getCurrentPrice(selectedSymbol))}</span>
                  </div>
                  <div className="summary-row">
                    <span>Estimated Total:</span>
                    <span className="total">{formatCurrency(getEstimatedTotal())}</span>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className={`btn btn-primary submit-btn ${loading ? 'loading' : ''}`}
                >
                  {loading ? 'Submitting...' : `${side.toUpperCase()} ${quantity} ${selectedSymbol}`}
                </button>

                {message && (
                  <div className={`message ${message.includes('failed') ? 'error' : 'success'}`}>
                    {message}
                  </div>
                )}
              </form>
            </div>
          </div>
        </div>

        <div className="market-info-section">
          <div className="card">
            <div className="card-header">
              <h3>Market Information</h3>
            </div>
            <div className="card-content">
              <div className="symbol-info">
                <h4>{selectedSymbol}</h4>
                {marketData[selectedSymbol] ? (
                  <div className="price-info">
                    <div className="current-price">
                      {formatCurrency(marketData[selectedSymbol].price)}
                    </div>
                    <div className={`price-change ${marketData[selectedSymbol].change >= 0 ? 'positive' : 'negative'}`}>
                      {marketData[selectedSymbol].change >= 0 ? '+' : ''}{marketData[selectedSymbol].change?.toFixed(5) || '0.00000'}
                      ({((marketData[selectedSymbol].change / marketData[selectedSymbol].price) * 100).toFixed(2)}%)
                    </div>
                  </div>
                ) : (
                  <div className="no-data">No market data available</div>
                )}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3>Recent Signals</h3>
            </div>
            <div className="card-content">
              {tradingSignals.length > 0 ? (
                <div className="signals-list">
                  {tradingSignals.slice(0, 5).map((signal, index) => (
                    <div key={index} className="signal-item">
                      <div className="signal-header">
                        <span className={`signal-type ${signal.type?.toLowerCase()}`}>
                          {signal.type}
                        </span>
                        <span className="signal-symbol">{signal.symbol}</span>
                      </div>
                      <div className="signal-details">
                        <span className="signal-price">{formatCurrency(signal.price)}</span>
                        <span className="signal-confidence">
                          {((signal.confidence || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-data">No recent signals</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingPanel;
