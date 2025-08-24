import React, { createContext, useContext, useState, ReactNode } from 'react';
import { symbols, Symbol } from '../config/symbols';

interface SymbolContextType {
  selectedSymbol: Symbol;
  setSelectedSymbol: (symbol: Symbol) => void;
  symbols: readonly Symbol[];
  isValidSymbol: (symbol: string) => symbol is Symbol;
}

const SymbolContext = createContext<SymbolContextType | null>(null);

export const SymbolProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [selectedSymbol, setSelectedSymbol] = useState<Symbol>(symbols[0]);

  const contextValue: SymbolContextType = {
    selectedSymbol,
    setSelectedSymbol,
    symbols,
    isValidSymbol: (symbol: string): symbol is Symbol => symbols.includes(symbol as Symbol),
  };

  return (
    <SymbolContext.Provider value={contextValue}>
      {children}
    </SymbolContext.Provider>
  );
};

export const useSymbol = (): SymbolContextType => {
  const context = useContext(SymbolContext);
  if (!context) {
    throw new Error('useSymbol must be used within a SymbolProvider');
  }
  return context;
};

export default SymbolContext;
