**SEP Trading UI**

This directory contains the React + TypeScript front‑end for the SEP Professional Trading System. The UI provides real‑time charts, trading controls, performance analytics, system health, and configuration editing.

**Prerequisites**

* Node.js 18.x (see `.nvmrc`).
* A running SEP engine backend with REST and WebSocket endpoints configured via `.env`.

**Getting Started**

```bash
cd frontend
npm install        # or pnpm install
cp .env.template .env  # edit API / WS URLs if necessary
npm start          # Runs the app in development mode on http://localhost:3000
```

**Environment Variables**

* `REACT_APP_API_URL`: Base URL for REST API.
* `REACT_APP_WS_URL`: WebSocket endpoint for real‑time data.

**Building for production**

```bash
npm run build
```

The static files will be output to `build/`. They can be served via Nginx using the provided Docker/Nginx configuration.

**File structure**

* `src/App.js`: top‑level component with tab navigation.
* `src/components/`: individual panels (Dashboard, Trading, MarketData, etc.).
* `src/context/`: React contexts for WebSocket, symbols, and configuration.
* `src/services/api.ts`: centralized client for all REST calls.
* `src/config/symbols.ts`: list of supported currency pairs.

**Connecting to the backend**

The UI connects to the SEP engine via:

* REST endpoints on `/api/**` for configuration, orders, positions, and metrics.
* WebSocket channels (`market`, `system`, `signals`, `performance`) handled by `WebSocketContext`.

Ensure the engine is running and OANDA credentials are configured via `EnvLoader` before launching the UI.