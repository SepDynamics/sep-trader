import React, { useState, useEffect, useContext } from 'react';
import { ConfigContext } from '../context/ConfigContext';
import { SymbolContext } from '../context/SymbolContext';
import '../styles/PairManager.css';

const PairManager = () => {
  const { config, updateConfig } = useContext(ConfigContext);
  const { currentSymbol, setCurrentSymbol } = useContext(SymbolContext);
  const [pairs, setPairs] = useState([]);
  const [newPair, setNewPair] = useState('');

  // Available currency pairs for the system
  const availablePairs = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF',
    'AUD_USD', 'USD_CAD', 'NZD_USD', 'EUR_GBP',
    'EUR_JPY', 'GBP_JPY', 'CHF_JPY', 'AUD_JPY',
    'CAD_JPY', 'NZD_JPY', 'EUR_AUD', 'EUR_CAD'
  ];

  useEffect(() => {
    if (config && config.enabledPairs) {
      setPairs(config.enabledPairs);
    }
  }, [config]);

  const togglePair = (pair) => {
    const updatedPairs = pairs.includes(pair)
      ? pairs.filter(p => p !== pair)
      : [...pairs, pair];
    
    setPairs(updatedPairs);
    updateConfig({ ...config, enabledPairs: updatedPairs });
  };

  const addCustomPair = () => {
    if (newPair && !pairs.includes(newPair)) {
      const updatedPairs = [...pairs, newPair.toUpperCase()];
      setPairs(updatedPairs);
      updateConfig({ ...config, enabledPairs: updatedPairs });
      setNewPair('');
    }
  };

  const removePair = (pair) => {
    const updatedPairs = pairs.filter(p => p !== pair);
    setPairs(updatedPairs);
    updateConfig({ ...config, enabledPairs: updatedPairs });
    
    // If we removed the current symbol, switch to first available
    if (pair === currentSymbol && updatedPairs.length > 0) {
      setCurrentSymbol(updatedPairs[0]);
    }
  };

  return (
    <div className="pair-manager">
      <div className="manager-header">
        <h1>Currency Pair Configuration</h1>
        <p>Manage which currency pairs are analyzed by the SEP Engine</p>
      </div>

      <div className="pair-controls">
        <div className="add-pair-section">
          <h2>Add Custom Pair</h2>
          <div className="add-pair-form">
            <input
              type="text"
              value={newPair}
              onChange={(e) => setNewPair(e.target.value)}
              placeholder="e.g., EUR_USD"
              className="pair-input"
            />
            <button onClick={addCustomPair} className="add-button">
              Add Pair
            </button>
          </div>
        </div>
      </div>

      <div className="pairs-grid">
        <h2>Available Pairs ({pairs.length} enabled)</h2>
        <div className="pairs-container">
          {availablePairs.map((pair) => (
            <div key={pair} className={`pair-card ${pairs.includes(pair) ? 'enabled' : 'disabled'}`}>
              <div className="pair-info">
                <span className="pair-symbol">{pair}</span>
                <span className={`pair-status ${pairs.includes(pair) ? 'active' : 'inactive'}`}>
                  {pairs.includes(pair) ? 'ENABLED' : 'DISABLED'}
                </span>
              </div>
              <div className="pair-actions">
                <button
                  onClick={() => togglePair(pair)}
                  className={`toggle-button ${pairs.includes(pair) ? 'disable' : 'enable'}`}
                >
                  {pairs.includes(pair) ? 'Disable' : 'Enable'}
                </button>
                {pairs.includes(pair) && (
                  <button
                    onClick={() => removePair(pair)}
                    className="remove-button"
                  >
                    Remove
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="current-analysis">
        <h2>Current Analysis Target</h2>
        <div className="current-symbol-display">
          <span className="symbol">{currentSymbol}</span>
          <button 
            onClick={() => {
              const enabledPairs = pairs.filter(p => p !== currentSymbol);
              if (enabledPairs.length > 0) {
                setCurrentSymbol(enabledPairs[0]);
              }
            }}
            className="switch-button"
            disabled={pairs.length <= 1}
          >
            Switch Pair
          </button>
        </div>
      </div>
    </div>
  );
};

export default PairManager;