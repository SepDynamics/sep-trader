import React from 'react';
import RealTimeMarketFeed from './RealTimeMarketFeed';
import AppHeader from './AppHeader';
import { useSymbol } from '../context/SymbolContext';
import { Symbol } from '../config/symbols';

const HomeDashboard: React.FC = () => {
  const { selectedSymbol, setSelectedSymbol, symbols } = useSymbol();

  return (
    <div className="min-h-screen bg-gray-950 text-gray-200 flex flex-col font-sans">
      <AppHeader />

      <main className="flex-grow container mx-auto p-4 sm:p-6 lg:p-8">
        <div className="bg-gray-900/50 rounded-xl shadow-2xl backdrop-blur-md border border-gray-700/50 overflow-hidden">
          <div className="p-4 sm:p-6 border-b border-gray-700/50">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-2xl font-bold text-white mb-2 sm:mb-0">EUR/USD Market Feed</h2>
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-400">Instrument:</span>
                <select
                  className="bg-gray-800/70 border border-gray-600 rounded-md px-3 py-1.5 text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-150"
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value as Symbol)}
                  disabled // UNTIL other pairs are seeded and verified
                >
                  {symbols.map(s => <option key={s} value={s}>{s.replace('_', '/')}</option>)}              
                </select>
              </div>
            </div>
          </div>
          
          <div className="p-2 sm:p-4 bg-black/20">
            <RealTimeMarketFeed />
          </div>
        </div>

        {/* Placeholder for future components */}
        <div className="mt-8 text-center">
          <p className="text-gray-500">Further analysis components will be displayed here.</p>
        </div>
      </main>
    </div>
  );
};

export default HomeDashboard;
