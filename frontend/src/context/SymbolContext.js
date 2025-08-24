// SEP Trading System - Symbol Context
// Manages selected trading symbol state across the application

import React, { createContext, useContext, useState } from 'react';
import { symbols } from '../config/symbols';

const SymbolContext = createContext(null);

export const SymbolProvider = ({ children }) => {
  const [selectedSymbol, setSelectedSymbol] = useState(symbols[0] || 'EUR_USD');

  const contextValue = {
    selectedSymbol,
    setSelectedSymbol,
    symbols,
    isValidSymbol: (symbol) => symbols.includes(symbol)
  };

  return (
    <SymbolContext.Provider value={contextValue}>
      {children}
    </SymbolContext.Provider>
  );
};

export const useSymbol = () => {
  const context = useContext(SymbolContext);
  if (!context) {
    throw new Error('useSymbol must be used within a SymbolProvider');
  }
  return context;
};

export { SymbolContext };
export default SymbolContext;