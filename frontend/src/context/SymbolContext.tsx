import React, { createContext, useContext, useState, ReactNode } from 'react';
import { symbols } from '../config/symbols';

interface SymbolContextType {
  selectedSymbol: string;
  setSelectedSymbol: (symbol: string) => void;
}

const SymbolContext = createContext<SymbolContextType>({
  selectedSymbol: symbols[0],
  setSelectedSymbol: () => {}
});

interface SymbolProviderProps {
  children: ReactNode;
}

export const SymbolProvider: React.FC<SymbolProviderProps> = ({ children }) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbols[0]);
  
  return (
    <SymbolContext.Provider value={{ selectedSymbol, setSelectedSymbol }}>
      {children}
    </SymbolContext.Provider>
  );
};

export const useSymbol = (): SymbolContextType => useContext(SymbolContext);