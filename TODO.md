# Demo Implementation Plan

- [ ] **Phase 1: Core Metrics Extraction**
  - [ ] Identify and isolate the core functions for calculating coherence, stability, and entropy from the existing `sep-trader` engine.
  - [ ] Create a lightweight, callable library (C API or Python bindings) to expose these metrics.

- [ ] **Phase 2: Metrics API Development**
  - [ ] Build a web service (using FastAPI or Flask) that exposes an endpoint for retrieving metrics (e.g., `/metrics?symbol=EURUSD&start=...&end=...`).
  - [ ] Integrate the metrics library from Phase 1.
  - [ ] Implement a caching layer using the existing PostgreSQL/TimescaleDB infrastructure to store and quickly retrieve computed metrics.

- [ ] **Phase 3: Trading Bot and Backtesting**
  - [ ] Develop a simple Python-based trading bot that consumes the metrics API.
  - [ ] Implement a basic trading strategy based on coherence, stability, and entropy thresholds.
  - [ ] Build a backtesting harness that feeds historical data to the bot and simulates trades.
  - [ ] Utilize the `SEP_BACKTESTING` mode in `sep-trader` to generate signals for comparison.

- [ ] **Phase 4: Performance Analysis and Reporting**
  - [ ] Implement logic to calculate key performance indicators (KPIs) such as P&L, maximum drawdown, and Sharpe ratio.
  - [ ] Create a report or dashboard to compare the metrics-driven strategy against a buy-and-hold baseline.
  - [ ] Incorporate existing performance data (e.g., 60.73% accuracy) to validate the engine's potential.