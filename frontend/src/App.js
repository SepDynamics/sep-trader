import React from 'react';
import { WebSocketProvider } from './context/WebSocketContext';
import { SymbolProvider } from './context/SymbolContext';
import { ConfigProvider } from './context/ConfigContext';
import HomeDashboard from './components/HomeDashboard';
import './styles/App.css';

const App = () => {
  return (
    <WebSocketProvider>
      <SymbolProvider>
        <ConfigProvider>
          <div className="app dark">
            <HomeDashboard />
          </div>
        </ConfigProvider>
      </SymbolProvider>
    </WebSocketProvider>
  );
};

export default App;
