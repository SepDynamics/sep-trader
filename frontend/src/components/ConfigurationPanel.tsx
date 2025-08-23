import React, { useState } from 'react';
import { useConfig } from '../context/ConfigContext';

const ConfigurationPanel: React.FC = () => {
  const { config, setConfig, loading, saveConfig } = useConfig();
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  const handleSave = async () => {
    try {
      setSaving(true);
      await saveConfig(config);
      setMessage('Configuration saved successfully');
      setTimeout(() => setMessage(''), 3000);
    } catch (error) {
      console.error('Failed to save configuration:', error);
      setMessage('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const handleInputChange = (section: string, key: string, value: any) => {
    setConfig((prev: any) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
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
                value={config.trading?.risk_level ?? ''}
                onChange={(e) => handleInputChange('trading', 'risk_level', e.target.value)}
              >
                <option value="" disabled>Select</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            <div className="field-group">
              <label>Max Position Size:</label>
              <input
                type="number"
                value={config.trading?.max_position_size ?? ''}
                onChange={(e) => handleInputChange('trading', 'max_position_size', parseInt(e.target.value))}
              />
            </div>
            <div className="field-group">
              <label>Stop Loss (%):</label>
              <input
                type="number"
                step="0.01"
                value={config.trading?.stop_loss_percent ?? ''}
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
                value={config.system?.refresh_interval ?? ''}
                onChange={(e) => handleInputChange('system', 'refresh_interval', parseInt(e.target.value))}
              />
            </div>
            <div className="field-group">
              <label>Debug Mode:</label>
              <input
                type="checkbox"
                checked={config.system?.debug_mode ?? false}
                onChange={(e) => handleInputChange('system', 'debug_mode', e.target.checked)}
              />
            </div>
            <div className="field-group">
              <label>Log Level:</label>
              <select
                value={config.system?.log_level ?? ''}
                onChange={(e) => handleInputChange('system', 'log_level', e.target.value)}
              >
                <option value="" disabled>Select</option>
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
                value={config.api?.timeout ?? ''}
                onChange={(e) => handleInputChange('api', 'timeout', parseInt(e.target.value))}
              />
            </div>
            <div className="field-group">
              <label>Rate Limit (requests/min):</label>
              <input
                type="number"
                value={config.api?.rate_limit ?? ''}
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

