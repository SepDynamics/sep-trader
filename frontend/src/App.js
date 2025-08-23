import React, { useState, useEffect } from 'react';
import { WebSocketProvider } from './context/WebSocketContext';
import { SymbolProvider } from './context/SymbolContext';
import { ConfigProvider } from './context/ConfigContext';
import HomeDashboard from './components/HomeDashboard';
import TradingPanel from './components/TradingPanel';
import SystemStatus from './components/SystemStatus';
import PerformanceMetrics from './components/PerformanceMetrics';
import MarketData from './components/MarketData';
import TradingSignals from './components/TradingSignals';
import ConfigurationPanel from './components/ConfigurationPanel';
import ConnectionStatusIndicator from './components/ConnectionStatusIndicator';
import QuantumAnalysis from './components/QuantumAnalysis';
import TestingSuite from './components/TestingSuite';
import './styles/App.css';

const App = () => {
  const [activeSection, setActiveSection] = useState('dashboard');
  const [testMode, setTestMode] = useState(false);
  const [darkMode, setDarkMode] = useState(true); // Default to dark mode

  useEffect(() => {
    // Load theme preference from localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      setDarkMode(savedTheme === 'dark');
    } else {
      // Default to dark mode for professional trading interface
      setDarkMode(true);
    }
  }, []);

  useEffect(() => {
    // Apply theme to document
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
    
    // Add body classes for theme
    document.body.className = darkMode ? 'dark-theme' : 'light-theme';
  }, [darkMode]);

  // Navigation sections aligned with example.html
  const navSections = [
    {
      title: 'Main',
      items: [
        { id: 'dashboard', label: 'Dashboard', icon: '📊' },
        { id: 'trading', label: 'Trading', icon: '💹' },
        { id: 'signals', label: 'Signals', icon: '🔔' },
      ]
    },
    {
      title: 'Analysis',
      items: [
        { id: 'quantum', label: 'Quantum Analysis', icon: '⚛️' },
        { id: 'performance', label: 'Performance', icon: '📈' },
        { id: 'market', label: 'Market Data', icon: '🌍' },
      ]
    },
    {
      title: 'Testing',
      items: [
        { id: 'unit-tests', label: 'Unit Tests', icon: '🧪' },
        { id: 'integration', label: 'Integration', icon: '🔗' },
        { id: 'backtest', label: 'Backtesting', icon: '⏪' },
      ]
    },
    {
      title: 'System',
      items: [
        { id: 'system', label: 'System Status', icon: '⚙️' },
        { id: 'config', label: 'Configuration', icon: '🔧' },
      ]
    }
  ];

  const handleSectionChange = (sectionId) => {
    setActiveSection(sectionId);
    
    // Update nav item active states
    document.querySelectorAll('.nav-item').forEach(item => {
      item.classList.remove('active');
    });
    
    // Handle test mode sections
    if (['unit-tests', 'integration', 'backtest'].includes(sectionId)) {
      setTestMode(true);
    } else if (testMode && !['unit-tests', 'integration', 'backtest'].includes(sectionId)) {
      setTestMode(false);
    }
  };

  const toggleTestMode = () => {
    setTestMode(!testMode);
    if (!testMode) {
      setActiveSection('unit-tests');
    } else {
      setActiveSection('dashboard');
    }
  };

  const renderActiveComponent = () => {
    switch (activeSection) {
      case 'dashboard':
        return <HomeDashboard />;
      case 'trading':
        return <TradingPanel />;
      case 'market':
        return <MarketData />;
      case 'signals':
        return <TradingSignals />;
      case 'quantum':
        return <QuantumAnalysis />;
      case 'performance':
        return <PerformanceMetrics />;
      case 'system':
        return <SystemStatus />;
      case 'config':
        return <ConfigurationPanel />;
      case 'unit-tests':
      case 'integration':
      case 'backtest':
        return <TestingSuite activeTab={activeSection} />;
      default:
        return <HomeDashboard />;
    }
  };

  return (
    <WebSocketProvider>
      <SymbolProvider>
        <ConfigProvider>
          <div className={`app ${darkMode ? 'dark' : 'light'}`}>
            {/* Animated Background */}
            <div className="background-animation"></div>
            
            {/* Header */}
            <header className="header">
              <div className="logo">
                <div className="logo-icon">⚡</div>
                <span>SEP Professional Trading System</span>
              </div>
              <div className="header-controls">
                <ConnectionStatusIndicator />
                <button
                  className="test-button"
                  onClick={toggleTestMode}
                  aria-label="Toggle test mode"
                >
                  Test Mode
                </button>
              </div>
            </header>

            {/* Main Container */}
            <div className="main-container">
              {/* Sidebar Navigation */}
              <aside className="sidebar">
                {navSections.map((section, sectionIndex) => (
                  <div key={sectionIndex} className="nav-section">
                    <div className="nav-title">{section.title}</div>
                    {section.items.map((item) => (
                      <a
                        key={item.id}
                        className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
                        onClick={() => handleSectionChange(item.id)}
                        role="button"
                        tabIndex={0}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            handleSectionChange(item.id);
                          }
                        }}
                      >
                        <span>{item.icon}</span> {item.label}
                      </a>
                    ))}
                  </div>
                ))}
              </aside>

              {/* Main Content */}
              <main className="content" id="mainContent">
                <div className="main-content">
                  {renderActiveComponent()}
                </div>
              </main>
            </div>

            {/* Footer */}
            <footer className="app-footer">
              <div className="footer-content">
                <span className="footer-text">
                  SEP Engine v4.1.0 | CUDA: Enabled | Quantum Engine: Active
                </span>
                <span className="footer-timestamp">
                  {new Date().toLocaleTimeString()}
                </span>
              </div>
            </footer>
          </div>
        </ConfigProvider>
      </SymbolProvider>
    </WebSocketProvider>
  );
};

export default App;