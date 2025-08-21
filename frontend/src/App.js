import React, { useState, useEffect } from 'react';
import { WebSocketProvider } from './hooks/useWebSocket';
import Dashboard from './components/Dashboard';
import TradingPanel from './components/TradingPanel';
import SystemStatus from './components/SystemStatus';
import PerformanceMetrics from './components/PerformanceMetrics';
import MarketData from './components/MarketData';
import TradingSignals from './components/TradingSignals';
import ConfigurationPanel from './components/ConfigurationPanel';
import './styles/App.css';

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Load theme preference from localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      setDarkMode(savedTheme === 'dark');
    } else {
      // Default to system preference
      setDarkMode(window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
  }, []);

  useEffect(() => {
    // Apply theme to document
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'trading', label: 'Trading', icon: '💹' },
    { id: 'market', label: 'Market Data', icon: '📈' },
    { id: 'signals', label: 'Signals', icon: '🔔' },
    { id: 'performance', label: 'Performance', icon: '📊' },
    { id: 'system', label: 'System', icon: '⚙️' },
    { id: 'config', label: 'Config', icon: '🔧' }
  ];

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'trading':
        return <TradingPanel />;
      case 'market':
        return <MarketData />;
      case 'signals':
        return <TradingSignals />;
      case 'performance':
        return <PerformanceMetrics />;
      case 'system':
        return <SystemStatus />;
      case 'config':
        return <ConfigurationPanel />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <WebSocketProvider>
      <div className={`app ${darkMode ? 'dark' : 'light'}`}>
        <header className="app-header">
          <div className="header-left">
            <h1 className="app-title">
              <span className="title-icon">⚡</span>
              SEP Professional Trading System
            </h1>
          </div>
          
          <div className="header-right">
            <button
              className="theme-toggle"
              onClick={() => setDarkMode(!darkMode)}
              aria-label="Toggle dark mode"
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </header>

        <nav className="app-nav">
          <div className="nav-tabs">
            {tabs.map(tab => (
              <button
                key={tab.id}
                className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <span className="tab-icon">{tab.icon}</span>
                <span className="tab-label">{tab.label}</span>
              </button>
            ))}
          </div>
        </nav>

        <main className="app-main">
          <div className="main-content">
            {renderActiveComponent()}
          </div>
        </main>

        <footer className="app-footer">
          <div className="footer-content">
            <span className="footer-text">
              SEP Engine v1.0 | Status: Online
            </span>
            <span className="footer-timestamp">
              {new Date().toLocaleTimeString()}
            </span>
          </div>
        </footer>
      </div>
    </WebSocketProvider>
  );
};

export default App;