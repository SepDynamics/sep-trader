# WebSocket Context

Provides a React context for real-time communication with the trading backend via WebSocket.

## Responsibilities
- Establishes and maintains the WebSocket connection.
- Shares market data, system status, signals, and performance metrics across components.
- Manages reconnect logic, channel subscriptions, and heartbeat pings.

## Configuration
- `REACT_APP_WS_URL`: WebSocket endpoint (default `ws://localhost:8765`).
- `maxReconnectAttempts`: how many times to retry connecting (default `10`).
- `reconnectDelay`: delay in ms between reconnection attempts (default `3000`).
