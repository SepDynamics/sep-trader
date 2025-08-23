# Components Overview

## Dashboard
- **Context**: `useWebSocket` provides `connected`, `systemStatus`, `marketData`, `performanceData`, and `tradingSignals`.
- **State**: `systemInfo`, `performance`, `loading`, and `error` manage initial data load and error handling.
- **Hardcoded/Placeholders**:
  - Displays only the first four market symbols and first five signals.
  - Quick action buttons ("Start Trading", "Pause System", etc.) have no handlers.

## TradingPanel
- **Context**: `useWebSocket` supplies `connected`, `marketData`, and `tradingSignals`.
- **State**: `selectedSymbol` (`'EUR/USD'`), `orderType` (`'market'`), `quantity` (`10000`), `price`, `side` (`'buy'`), `loading`, and `message`.
- **Hardcoded/Placeholders**:
  - Currency `symbols` array and default order quantity.
  - After submitting an order, `quantity` resets to `100`.

## ConfigurationPanel
- **State**: `config`, `loading`, `saving`, and `message` drive configuration forms.
- **Hardcoded Defaults**: risk level `'medium'`, max position size `10000`, stop loss `5%`, refresh interval `30s`, debug mode `false`, log level `'INFO'`, API timeout `30s`, rate limit `60`.

## HomeDashboard
- **Context**: `useWebSocket` exposes connection state, system metrics, and live data.
- **State**: `apiHealth`, `activeTab`, `selectedPairs`, `commandHistory`, `commandInput`, and `isExecutingCommand` control dashboard interaction.
- **Hardcoded/Placeholders**:
  - `API_URL` defaults to `http://localhost:5000` if environment variable is missing.
  - Quick actions ("Upload Training Data", "Start Model Training", etc.) are UI stubs.

## PerformanceMetrics
- **Context**: Receives `connected` and `performanceData` from `useWebSocket`.
- **State**: `metrics`, `loading`, and `error` manage API fetch and display.

## MarketData
- **Context**: `useWebSocket` supplies live `marketData`.
- **State**: `selectedSymbol` initialized to `'EUR/USD'` (unused by current layout).

## SystemStatus
- **Context**: `useWebSocket` provides `connected` and `systemStatus`.
- **State**: `systemInfo`, `loading`, `error`, and `lastRefresh` handle polling logic.
- **Hardcoded/Placeholders**:
  - Refresh interval fixed at `30s`.
  - Component list (SEP Engine, Memory Tiers, Trading System, WebSocket Service) is static.

## TradingSignals
- **Context**: `useWebSocket` provides `connected` and `tradingSignals`.
- **State**: `filter` toggles between `all`, `buy`, and `sell` views.
- **Hardcoded/Placeholders**: Filter options and empty-state message strings.

