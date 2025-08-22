import React, { useState, useEffect } from 'react';
import { apiClient } from '../services/api';

const ConfigurationPanel = () => {
  const [config, setConfig] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    loadConfiguration();
  }, []);

  const loadConfiguration = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getConfiguration();
      setConfig(response.data);
    } catch (error) {
      console.error('Failed to load configuration:', error);
      setMessage('Failed to load configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      await apiClient.setConfiguration(config);
      setMessage('Configuration saved successfully');
      setTimeout(() => setMessage(''), 3000);
    } catch (error) {
      console.error('Failed to save configuration:', error);
      setMessage('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const handleInputChange = (section, key, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  if (loading) {
    return <div className="loading-container">Loading configuration...</div>;
  }

  return (
    <div className="configuration-panel">
      <div className="config-header">
        <h1>System Configuration</h1>
        <button 
          onClick={handleSave} 
          disabled={saving}
          className="save-btn"
        >
          {saving ? 'Saving...' : 'Save Configuration'}
        </button>
      </div>

      <div className="config-sections">
        <div className="config-section">
          <h3>Trading Settings</h3>
          <div className="config-fields">
            <div className="field-group">
              <label>Risk Level:</label>
              <select
                value={config.trading?.risk_level || 'medium'}
                onChange={(e) => handleInputChange('trading', 'risk_level', e.target.value)}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            <div className="field-group">
              <label>Max Position Size:</label>
              <input
                type="number"
                value={config.trading?.max_position_size || 10000}
                onChange={(e) => handleInputChange('trading', 'max_position_size', parseInt(e.target.value))}
              />
            </div>
            <div className="field-group">
              <label>Stop Loss (%):</label>
              <input
                type="number"
                step="0.01"
                value={config.trading?.stop_loss_percent || 5}
                onChange={(e) => handleInputChange('trading', 'stop_loss_percent', parseFloat(e.target.value))}
              />
            </div>
          </div>
        </div>

        <div className="config-section">
          <h3>System Settings</h3>
          <div className="config-fields">
            <div className="field-group">
              <label>Refresh Interval (seconds):</label>
              <input
                type="number"
                value={config.system?.refresh_interval || 30}
                onChange={(e) => handleInputChange('system', 'refresh_interval', parseInt(e.target.value))}
              />
            </div>
            <div className="field-group">
              <label>Debug Mode:</label>
              <input
                type="checkbox"
                checked={config.system?.debug_mode || false}
                onChange={(e) => handleInputChange('system', 'debug_mode', e.target.checked)}
              />
            </div>
            <div className="field-group">
              <label>Log Level:</label>
              <select
                value={config.system?.log_level || 'INFO'}
                onChange={(e) => handleInputChange('system', 'log_level', e.target.value)}
              >
                <option value="DEBUG">Debug</option>
                <option value="INFO">Info</option>
                <option value="WARNING">Warning</option>
                <option value="ERROR">Error</option>
              </select>
            </div>
          </div>
        </div>

        <div className="config-section">
          <h3>API Settings</h3>
          <div className="config-fields">
            <div className="field-group">
              <label>API Timeout (seconds):</label>
              <input
                type="number"
                value={config.api?.timeout || 30}
                onChange={(e) => handleInputChange('api', 'timeout', parseInt(e.target.value))}
              />
            </div>
            <div className="field-group">
              <label>Rate Limit (requests/min):</label>
              <input
                type="number"
                value={config.api?.rate_limit || 60}
                onChange={(e) => handleInputChange('api', 'rate_limit', parseInt(e.target.value))}
              />
            </div>
          </div>
        </div>
      </div>

      {message && (
        <div className={`config-message ${message.includes('Failed') ? 'error' : 'success'}`}>
          {message}
        </div>
      )}
    </div>
  );
};

export default ConfigurationPanel;