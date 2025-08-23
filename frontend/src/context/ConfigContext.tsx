import React, { createContext, useContext, useEffect, useState } from 'react';
import { getConfig, updateConfig } from '../services/api';

interface ConfigContextType {
  config: any;
  setConfig: React.Dispatch<React.SetStateAction<any>>;
  loading: boolean;
  refreshConfig: () => Promise<void>;
  saveConfig: (cfg: any) => Promise<void>;
}

const ConfigContext = createContext<ConfigContextType | undefined>(undefined);

export const ConfigProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [config, setConfig] = useState<any>({});
  const [loading, setLoading] = useState<boolean>(true);

  const refreshConfig = async () => {
    setLoading(true);
    try {
      const response = await getConfig();
      // Some APIs wrap the data, handle both cases
      setConfig((response as any).data ?? response);
    } finally {
      setLoading(false);
    }
  };

  const saveConfig = async (cfg: any) => {
    await updateConfig(cfg);
    setConfig(cfg);
  };

  useEffect(() => {
    refreshConfig();
  }, []);

  return (
    <ConfigContext.Provider value={{ config, setConfig, loading, refreshConfig, saveConfig }}>
      {children}
    </ConfigContext.Provider>
  );
};

export const useConfig = () => {
  const ctx = useContext(ConfigContext);
  if (!ctx) {
    throw new Error('useConfig must be used within ConfigProvider');
  }
  return ctx;
};

