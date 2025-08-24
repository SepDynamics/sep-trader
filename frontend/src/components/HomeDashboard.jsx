import React from 'react';
import RealTimeMarketFeed from './RealTimeMarketFeed';
import AppHeader from './AppHeader';
import { useSymbol } from '../context/SymbolContext';

const HomeDashboard = () => {
  const { selectedSymbol, setSelectedSymbol, symbols } = useSymbol();

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <AppHeader />
      
      <main className="flex-grow p-6">
        <div className="bg-gray-900 rounded-lg p-4 shadow-xl">
            <div className="flex items-center gap-4 mb-4">
              <h3 className="text-lg font-semibold text-gray-200">Market Price: {selectedSymbol.replace('_', '/')}</h3>
              <select
                className="bg-gray-800 border border-gray-700 rounded-md px-3 py-1 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                disabled // UNTIL other pairs are seeded and verified
              >
                {symbols.map(s => <option key={s} value={s}>{s.replace('_', '/')}</option>)}              
              </select>
            </div>
          
            <RealTimeMarketFeed /> 
        </div>

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Other components can be added here */}
        </div>
      </main>
    </div>
  );
};

export default HomeDashboard;
