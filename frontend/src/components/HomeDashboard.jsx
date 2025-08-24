// File: /sep/frontend/src/components/HomeDashboard.jsx
import React from 'react';
import RealTimeMarketFeed from './RealTimeMarketFeed';
import AppHeader from './AppHeader';
import { useSymbol } from '../context/SymbolContext';
import { symbols } from '../config/symbols';

const HomeDashboard = () => {
  const { selectedSymbol, setSelectedSymbol } = useSymbol();

  return (
    <div class="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <AppHeader />
      
      <main class="flex-grow p-6">
        <div class="bg-gray-900 rounded-lg p-4 shadow-xl">
            <div class="flex items-center gap-4 mb-4">
              <h3 class="text-lg font-semibold text-gray-200">Market Price: {selectedSymbol}</h3>
              <select
                class="bg-gray-800 border border-gray-700 rounded-md px-3 py-1 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.targe.value)}
                disabled // UNTIL other pairs are seeded and verified
              >
                {symbols.map(s => <option key={s} value={s.replace('_', '/')}>{s.replace('_', '/')}</option>)}
              </select>
            </div>
          
            {/* THIS IS THE KEY PART - It will now render the live chart */}
            <RealTimeMarketFeed /> 
        </div>

        {/* You can add your other components like EntropyBandAnalysis here later */}
        <div class="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* <div class="bg-gray-900 rounded-lg p-4 shadow-xl"> <EntropyBandAnalysis /> </div> */}
            {/* <div class="bg-gray-900 rounded-lg p-4 shadow-xl"> <PathHistoryMatching /> </div> */}
        </div>
      </main>
    </div>
  );
};

export default HomeDashboard;
