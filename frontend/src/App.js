import React from 'react';
import { WebSocketProvider } from './context/WebSocketContext';
import { SymbolProvider } from './context/SymbolContext';
import { ConfigProvider } from './context/ConfigContext';
import RealTimeMarketFeed from './components/RealTimeMarketFeed';
import './styles/App.css';

const App = () => {
  return (
    <WebSocketProvider>
      <SymbolProvider>
        <ConfigProvider>
          <div className="app dark">
            <div className="background-animation"></div>
            
            {/* Simple Header */}
            <header className="simple-header">
              <h1>SEP Market Data</h1>
              <div className="connection-status">
                <span className="status-dot"></span>
                <span>Live Feed</span>
              </div>
            </header>

            {/* Main Content - Just the Chart */}
            <main className="main-chart-container">
              <div className="chart-wrapper">
                <RealTimeMarketFeed />
              </div>
            </main>
          </div>
        </ConfigProvider>
      </SymbolProvider>
    </WebSocketProvider>
  );
};

export default App;