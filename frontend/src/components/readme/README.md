# Components Overview

Currency pairs are defined once in `src/config/symbols.ts` and shared across the UI.
`SymbolContext` exposes the active `selectedSymbol` so components no longer maintain their own defaults.

- **TradingPanel** populates its symbol selector from the shared list.
- **MarketData** highlights data for the `selectedSymbol` and shows all configured pairs.
