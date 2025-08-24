// SEP Trading System - Symbol Configuration
// OANDA-compatible forex pairs for manifold trading

export const symbols = [
  'EUR_USD',
  'GBP_USD', 
  'USD_JPY',
  'AUD_USD',
  'USD_CHF',
  'USD_CAD',
  'NZD_USD',
  'EUR_GBP',
  'EUR_JPY',
  'GBP_JPY'
];

export const symbolInfo = {
  EUR_USD: { name: 'Euro/US Dollar', minSpread: 0.0001, precision: 5 },
  GBP_USD: { name: 'British Pound/US Dollar', minSpread: 0.0001, precision: 5 },
  USD_JPY: { name: 'US Dollar/Japanese Yen', minSpread: 0.001, precision: 3 },
  AUD_USD: { name: 'Australian Dollar/US Dollar', minSpread: 0.0001, precision: 5 },
  USD_CHF: { name: 'US Dollar/Swiss Franc', minSpread: 0.0001, precision: 5 },
  USD_CAD: { name: 'US Dollar/Canadian Dollar', minSpread: 0.0001, precision: 5 },
  NZD_USD: { name: 'New Zealand Dollar/US Dollar', minSpread: 0.0001, precision: 5 },
  EUR_GBP: { name: 'Euro/British Pound', minSpread: 0.0001, precision: 5 },
  EUR_JPY: { name: 'Euro/Japanese Yen', minSpread: 0.001, precision: 3 },
  GBP_JPY: { name: 'British Pound/Japanese Yen', minSpread: 0.001, precision: 3 }
};

export const getSymbolName = (symbol) => symbolInfo[symbol]?.name || symbol;
export const getSymbolPrecision = (symbol) => symbolInfo[symbol]?.precision || 5;
export const getSymbolMinSpread = (symbol) => symbolInfo[symbol]?.minSpread || 0.0001;
