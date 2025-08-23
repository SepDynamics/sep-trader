
### Frontend Bring‑Up Tasks

1. **Tab rendering** – Replace the placeholder returns in `App.js` so each tab displays its corresponding component.
2. **HomeDashboard** – Build a unified dashboard with account balance, latest signal, and system health.
3. **TradingPanel** – Create an order entry form tied to `apiClient.placeOrder()`/`apiClient.getPositions()`. Include symbol selection from `symbols.ts`.
4. **MarketData** – Plot real‑time candles from `market` messages and historical data from `/api/market-data`.
5. **TradingSignals** – List recent signals from the WebSocket `signals` channel with type, confidence, and timestamp.
6. **PerformanceMetrics** – Fix the API call to use `getPerformanceMetrics()`. Display P\&L, win rate, total trades, Sharpe ratio, and drawdown.
7. **SystemStatus** – Show engine status and metrics (latency, throughput) from `/api/system-status` and the `system` channel.
8. **ConfigurationPanel** – Fetch current config via `/api/config/get` and allow editing/saving via `/api/config/set`. Provide form validation and feedback.
9. **Redis Metrics** – Add a `/api/metrics/redis` endpoint to expose key metrics. Create a UI component to poll and display these stats.
10. **Theme & UX** – Finalise dark/light theme support, responsive layout, and accessible navigation. Use Tailwind CSS and lucide icons consistently.
11. **Tests** – Implement unit tests for each component and integration tests for tab switching and order placement.
12. **Documentation** – Update this TODO and the README when tasks are completed. Remove any leftover stub references.

## ✅ Combined task list for the frontend trading UI

1. **Bootstrap & environment**

   * Add a `.env` file in `frontend/` with:

     ```
     REACT_APP_API_URL=http://localhost:5000
     REACT_APP_WS_URL=ws://localhost:8765
     ```

     allowing overrides for production (e.g. droplet IP).
   * Update `package.json` to define Node version and install scripts if missing.
   * Run `npm install` or `pnpm install` to fetch dependencies.

2. **Fix `App.js` to render tabs correctly**

   * The `App.js` currently defines navigation tabs (`dashboard`, `trading`, `market`, etc.) but does not render the corresponding components. Replace the empty `return` statements in `renderActiveComponent()` with `<HomeDashboard />`, `<TradingPanel />`, `<MarketData />`, `<TradingSignals />`, `<PerformanceMetrics />`, `<SystemStatus />`, and `<ConfigurationPanel />` in the appropriate cases.

3. **Implement missing components**

   * **HomeDashboard**: summarise key metrics—latest P\&L, account balance, recent signal triggers, and system status.
   * **TradingPanel**: allow selecting a currency pair from `symbols.ts`, entering order size/direction, and calling `apiClient.placeOrder()` or `apiClient.submitOrder()`. Display open positions from `apiClient.getPositions()`.
   * **MarketData**: fetch price history via `apiClient.getMarketData()` and subscribe to WebSocket `market` messages to update a candlestick or line chart (use Recharts). Allow switching between symbols.
   * **TradingSignals**: display a table of recent signals from WebSocket `signals` channel; include fields like `signal_type`, `confidence`, `timestamp`.
   * **PerformanceMetrics**: fix the API call (the component currently calls `apiClient.getPerformanceCurrent` but `api.ts` exports `getPerformanceMetrics()`). Show total P\&L, daily P\&L, win rate, total trades, Sharpe ratio, and max drawdown. Provide conditional formatting (e.g. green for positive P\&L) and a refresh button.
   * **SystemStatus**: display system health from `apiClient.getSystemStatus()` and WebSocket `system` messages. Show connection state (connected/disconnected) and metrics (latency, throughput) from `data.metrics`.
   * **ConfigurationPanel**: load current config via `apiClient.getConfig()` and allow editing values. Save updates via `apiClient.updateConfig()`. Include validation and success/error notifications.

4. **WebSocket enhancements**

   * Use the `WS_URL` from environment or fallback `ws://localhost:8765`.
   * Keep the reconnection logic and heartbeat in `WebSocketContext.js` but add logging to UI (e.g. toast notifications for reconnect attempts).
   * Surface `connectionStatus` (`connecting`, `connected`, `disconnected`, `reconnecting`, `failed`) to `SystemStatus` or a status indicator in the header.

5. **Connect Redis metrics**

   * Add a new backend endpoint (e.g. `/api/metrics/redis`) that reads the latest system counters (latency, throughput, cache hits) from Redis.
   * Create a `RedisMetrics` React component that calls this endpoint every few seconds or subscribes to `systemMetrics` from WebSocket and displays live graphs.

6. **User experience**

   * Implement a dark/light theme toggle (already partially done in `App.js` with localStorage and `data-theme` attributes) and ensure Tailwind classes react to `[data-theme="dark"]`.
   * Use Tailwind CSS (already included in the project) and `lucide-react` icons consistently across components.
   * Ensure layout is responsive and mobile-friendly: use a sidebar on desktop and a bottom nav on small screens.

7. **Testing & validation**

   * Write unit tests with `@testing-library/react` for each component to verify that API calls succeed and UI updates when WebSocket messages arrive.
   * Mock API responses and WebSocket events to test reconnection logic and error handling.
   * Add Cypress or Playwright tests to simulate user navigation between tabs and verify forms (order placement and config editing).

8. **Build & deployment**

   * Document the build command (`npm run build`) and configure Nginx to serve the `frontend/build` directory.
   * Update `docker-compose.yml` or deployment scripts to copy the built UI into the Nginx container and expose port 80.
   * Ensure production `REACT_APP_API_URL` and `REACT_APP_WS_URL` are set via environment or `.env.production` on the droplet.

9. **Documentation and cleanup**

   * Update the `docs/README.md` to include instructions for running the web UI and pointers to configuration variables.
   * Add API usage examples in `docs/README.md` for each endpoint used by the UI.
   * Remove any leftover placeholder code or commented stubs in `frontend/` and document the removal in commit messages.

