import React, { useState, useEffect } from 'react';
import { WebSocketProvider } from './context/WebSocketContext';
import { SymbolProvider } from './context/SymbolContext';
import { ConfigProvider } from './context/ConfigContext';
import SEPDashboard from './components/SEPDashboard';
import HomeDashboard from './components/HomeDashboard';
import PatternAnalysis from './components/PatternAnalysis';
import ValkeyPipelineManager from './components/ValkeyPipelineManager';
import ManifoldKernel from './components/ManifoldKernel';
import QuantumAnalysis from './components/QuantumAnalysis';
import SystemMonitor from './components/SystemMonitor';
import PairManager from './components/PairManager';
import AppHeader from './components/AppHeader';
import './styles/App.css';

const App = () => {
  const [activeSection, setActiveSection] = useState('dashboard');
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      setDarkMode(savedTheme === 'dark');
    }
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
    document.body.className = darkMode ? 'dark-theme' : 'light-theme';
  }, [darkMode]);

  const navSections = [
    {
      title: 'SEP Engine',
      items: [
        { id: 'dashboard', label: 'System Overview', icon: '⚡' },
        { id: 'pipeline', label: 'Data Pipeline', icon: '🌊' },
        { id: 'pairs', label: 'Currency Pairs', icon: '💱' },
      ]
    },
    {
      title: 'Pattern Analysis',
      items: [
        { id: 'patterns', label: 'Live Patterns', icon: '🧠' },
        { id: 'kernel', label: 'Manifold Kernel', icon: '⚛️' },
        { id: 'quantum', label: 'Quantum States', icon: '🔬' },
      ]
    },
    {
      title: 'Monitoring',
      items: [
        { id: 'system', label: 'System Health', icon: '📊' },
      ]
    }
  ];

  const renderActiveComponent = () => {
    switch (activeSection) {
      case 'dashboard':
        return <HomeDashboard />;
      case 'pipeline':
        return <ValkeyPipelineManager />;
      case 'pairs':
        return <PairManager />;
      case 'patterns':
        return <PatternAnalysis />;
      case 'kernel':
        return <ManifoldKernel />;
      case 'quantum':
        return <QuantumAnalysis />;
      case 'system':
        return <SystemMonitor />;
      default:
        return <HomeDashboard />;
    }
  };

  return (
    <WebSocketProvider>
      <SymbolProvider>
        <ConfigProvider>
          <div className={`app ${darkMode ? 'dark' : 'light'}`}>
            <div className="background-animation"></div>
            
            {/* Header */}
            <AppHeader />

            <div className="main-container">
              {/* Sidebar */}
              <aside className="sidebar">
                {navSections.map((section, sectionIndex) => (
                  <div key={sectionIndex} className="nav-section">
                    <div className="nav-title">{section.title}</div>
                    {section.items.map((item) => (
                      <a
                        key={item.id}
                        className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
                        onClick={() => setActiveSection(item.id)}
                        role="button"
                        tabIndex={0}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            setActiveSection(item.id);
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
              <main className="content">
                <div className="main-content">
                  {renderActiveComponent()}
                </div>
              </main>
            </div>

            {/* Footer */}
            <footer className="app-footer">
              <div className="footer-content">
                <span className="footer-text">
                  SEP Engine v4.1.0 | CUDA: Enabled | Quantum Pattern Recognition: Active
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