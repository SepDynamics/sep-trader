import React, { createContext, useContext, useState } from 'react';
import { symbols } from '../config/symbols';

const SymbolContext = createContext({
  selectedSymbol: symbols[0],
  setSelectedSymbol: () => {}
});

export const SymbolProvider = ({ children }) => {
  const [selectedSymbol, setSelectedSymbol] = useState(symbols[0]);
  return (
    <SymbolContext.Provider value={{ selectedSymbol, setSelectedSymbol }}>
      {children}
    </SymbolContext.Provider>
  );
};

export const useSymbol = () => useContext(SymbolContext);
