# Components Overview

Currency pairs are defined once in `src/config/symbols.ts` and shared across the UI.
`SymbolContext` exposes the active `selectedSymbol` so components no longer maintain their own defaults.

- **TradingPanel** now posts orders to the backend via `submitOrder` using the `buildOrder` utility and resets quantities from `ConfigContext`.
- **MarketData** highlights data for the `selectedSymbol` and shows all configured pairs.
