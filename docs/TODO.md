Below is a **structured execution plan** for building a suite of visual aids to accompany the SEP Engine whitepaper.  The plan is grounded in an investigation of the **`SepDynamics/sep‑trader` repo**—particularly the `frontend/`, `docs/`, and `src/` folders—and is designed to explain SEP’s concepts to a **high‑school‑level audience**.  Each task references the relevant code and documentation lines for context.

- [x] Quote `-Dnoexcept(x)=` in CMake for stable NVCC builds; default all build scripts to Docker for CUDA.

---

## 0. Build System Maintenance

- Fixed NVCC failure by removing unquoted `-Dnoexcept(x)=` flag from `CMakeLists.txt` to ensure CUDA builds succeed within Docker.

## 1. Groundwork: Understand the Existing Code and Metrics

1. **Review the whitepaper and docs** – The whitepaper outlines the theoretical framework (entropy, stability, coherence) and the idea of a time‑anchored manifold of identities.  The docs README specifies how the frontend talks to the backend via REST and WebSockets.  The TODO lists missing UI components and tasks.
2. **Locate the metric definitions** – In the `src` folder, `pattern_metric_engine.cpp` computes a “coherence” measure from pattern variance (inverse of the coefficient of variation) and a stability measure via `calculateStability()` and `calculateEntropy()`.  These functions are your reference point when describing the metrics.  For bit‑level coherence/stability/entropy, use the formulas from the whitepaper (entropy = average Shannon entropy of each bit; stability = 1 – EMA of bit flips; coherence = 1 – weighted sum of rupture counts).
3. **Identify data sources** – The API client (`frontend/src/services/api.ts`) supports endpoints for market data, performance metrics, signals, Valkey metrics, and live patterns.  These endpoints will feed the visualisations.  The WebSocket context already subscribes to channels (`market`, `signals`, `performance`, `valkey_metrics`, `manifold_update`, etc.), storing data in React state.  This infrastructure is ready to serve the visual components.

---

## 2. Define the Visual Aids

1. **Metric time‑series graphs** – For a given identity (signal), display how entropy, stability, and coherence evolve as new data arrives.  Use a line chart with three coloured lines (one per metric).  Recharts is already a dependency, so leverage its `<LineChart>` component.
2. **3D manifold scatter** – Show a scatter or path in (entropy, stability, coherence) space.  Each point corresponds to a Valkey key (timestamp).  Colour points by time (older = darker, newer = brighter) to illustrate convergence.  Use a React 3D library (e.g. `react-three-fiber`) or fallback to a 2D projection if 3D is too heavy for the first release.
3. **Band/radial diagram** – Represent the “hot”, “warm” and “cold” bands of the manifold.  A simple radial chart or concentric ring chart can illustrate that most newly created identities reside in the outer ring (high entropy), while longer‑lived ones occupy inner rings.  Each band’s colour intensity can reflect average coherence.
4. **Rupture histogram** – Plot the distribution of rupture counts (bit overlaps) over time to visualise when the system experiences state changes.
5. **Pipeline flow diagram** – A high‑level flowchart showing: (1) raw market ticks → (2) deterministic kernel (computes pin transitions) → (3) Valkey store/manifold → (4) GPU updater (for deeper bands) → (5) front‑end UI.  This helps students understand the separation of concerns.
6. **Interactive “identity inspector”** – A panel that, upon clicking a data point in any chart, shows: the timestamp, price, entropy, stability, coherence, state (“flux”, “stabilizing”, “converged”), and predecessor state if available.  This demonstrates the deterministic, invertible nature of the kernel.

---

## 3. Implementation Tasks (Frontend)

1. **Bootstrap environment** – Ensure `frontend/.env` defines `REACT_APP_API_URL` and `REACT_APP_WS_URL`.  Run `npm install` to install dependencies.
2. **Fix `App.js`** – Wire up the tabs to render the correct components (HomeDashboard and other planned panels).  Ensure each new component mounts properly and receives data via the WebSocket context.
3. **Create a `ManifoldContext`** – Extend `WebSocketContext` or create a new context to store manifold‑specific data: arrays of {timestamp, entropy, stability, coherence, state} keyed by instrument.  This context will feed the visual components.
4. **Develop visual components**:

   * **`MetricTimeSeries.js`** – Accepts an identity key or instrument; fetches its metric history via `/api/valkey/metrics` or WebSocket; renders a Recharts line chart.  Provide tooltips and legends.
   * **`ManifoldScatter.js`** – Renders a 3D scatter using `react-three-fiber`.  Use small spheres for points; allow basic rotation and zoom.
   * **`BandsDiagram.js`** – Implements the radial band using Recharts’ `<RadialBarChart>` or a custom SVG.  The outer ring corresponds to keys with high entropy; the inner ring to settled keys.
   * **`RuptureHistogram.js`** – Plots histogram data from a `rupture_count` array (derived from pin state overlaps); uses a bar chart.
   * **`PipelineDiagram.js`** – Static SVG showing the data pipeline.  Use simple shapes and arrows; annotate each stage (OANDA → Kernel → Manifold → GPU → UI).
   * **`IdentityInspector.js`** – Panel that appears on point click; shows metrics and a link to view the previous state (if available).
5. **Integrate with WebSocket** – Update `WebSocketContext` to dispatch `manifold_update`, `pin_state_change`, and `signal_evolution` into the new `ManifoldContext`.  Provide derived selectors like `getIdentityHistory(key)` for the visual components.
6. **Hook to API** – Add new endpoints in `api.ts` if missing (e.g. `/api/valkey/metrics`) to fetch precomputed histories; implement fallback logic using WebSocket data.
7. **Theme & UX** – Use Tailwind for layout and styling; ensure charts adapt to dark and light themes.  Provide tooltips and friendly labels explaining the meanings of entropy, stability, and coherence.
8. **Testing** – Write unit tests for each component using `@testing-library/react` (already included).  Replay captured testbed WebSocket streams to test state updates.  Use Cypress to verify that clicking a point opens the inspector.

---

## 4. Backend & Data Preparation Tasks

1. **Valkey key design** – Ensure the backend stores each signal or pattern as `sep:signal:{instrument}:{timestamp}` with fields `{price, entropy, stability, coherence, state, created_at, last_update}` (similar to the whitepaper).  Provide an index (ZSET) for chronological access.  This is required for the manifold visualisations.
2. **New API endpoints** – Implement `/api/valkey/metrics` to return a time‑series of (entropy, stability, coherence) for a given identity or instrument.  Add `/api/ruptures` to retrieve rupture counts over a window.  These endpoints will feed the graphs.
3. **WebSocket messages** – Make sure the `manifold_update`, `pin_state_change`, `signal_evolution`, and `valkey_metrics` messages deliver the fields needed for the graphs.  The current `WebSocketContext` already anticipates these handlers.

---

## 5. Educational Narrative (High‑School Level)

1. **Simple definitions** – Explain that *entropy* measures randomness (like flipping a coin), *stability* measures how much the signal stays the same from one tick to the next, and *coherence* measures how consistently the signal follows a pattern.  Use analogies (e.g. music rhythm vs noise).
2. **Clock tick metaphor** – Illustrate that each key in the database corresponds to a clock tick.  Market data arrives at each tick and gets “pinned” to that key.  The metrics summarise how “noisy” or “stable” that tick is.
3. **Visual transitions** – Show a sequence of charts where entropy drops and stability/coherence rise.  Emphasise that patterns only emerge when enough randomness fades away.
4. **Manifold picture** – Use the 3D scatter to show that stable patterns cluster together in a part of the metric space while chaotic ones spread out.  Highlight that a “long chain” corresponds to a path that stays in the coherent region.
5. **Deterministic kernel** – Discuss that the system’s kernel is like a precise calculator: given the current state and new data, it always gives the same result.  Thus, the path is reproducible and can even be reconstructed backwards.
6. **Decisions from patterns** – Briefly show how a trading signal might be generated when a pattern reaches high coherence and stability.  Make it clear that signals are decisions *about* the data, not random guesses.

---

## 6. Documentation & Presentation

1. **Update `docs/README.md`** – Include instructions on running the visualisations and a section titled “Understanding SEP Manifold Visualisations” with screenshots and descriptions.
2. **Extend the whitepaper** – Add a new section summarising how the manifold visualisations map to the theoretical constructs: time‑anchored keys, metric space, deterministic kernel.  Use diagrams created in your new components.
3. **High‑level slide deck** – Prepare slides that borrow from 3Blue1Brown‑style animations: start with a single point (random), then show paths as entropy decreases and coherence/stability increase.  Use simple language and annotate the axes.  End with the pipeline diagram showing how the SEP engine processes data.

---

### Summary of Key Repo Citations

* **Frontend API consumption** – The UI calls `/api/candles/fetch`, `/api/candles/{instrument}`, `/api/config/get`, and `/api/config/set`, confirming where data comes from.
* **TODO tasks** – The docs/TODO file lists essential tasks for the UI: rendering tabs, building each component, hooking WebSocket, adding Valkey metrics, theme support, tests, and documentation.
* **Metric definitions** – The core engine computes coherence, stability, and entropy by measuring pattern variance, consistency, and entropy of bit patterns.
* **UI/backend separation** – The external API ensures that the front-end only renders data; all heavy computation happens in the engine.

By following this plan, you will build a **compelling set of visual aids** that not only illustrate the SEP Engine's operation but also make complex concepts like entropy and coherence accessible to a broad audience.

---

