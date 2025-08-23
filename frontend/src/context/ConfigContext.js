import React, { createContext, useContext, useEffect, useState } from 'react';
import { apiClient } from '../services/api';

const ConfigContext = createContext(null);

export const ConfigProvider = ({ children }) => {
  const [config, setConfig] = useState(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const cfg = await apiClient.getConfiguration();
        setConfig(cfg);
      } catch (err) {
        console.error('Failed to load configuration', err);
        setConfig({});
      }
    };
    fetchConfig();
  }, []);

  return (
    <ConfigContext.Provider value={config}>
      {children}
    </ConfigContext.Provider>
  );
};

export const useConfig = () => useContext(ConfigContext);
export default ConfigContext;
