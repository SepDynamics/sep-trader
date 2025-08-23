# IO Connectors Overview

Modules under `src/io` provide external interfaces for data ingress and
language bindings.

## OANDA connector

Files: `oanda_connector.*`, `oanda_constants.h`

- Handles REST and streaming communication with the OANDA trading platform
  using libcurl.
- Features historical candle retrieval, real‑time price streaming, account
  and order queries, plus basic order placement.
- Implements caching, ATR‑based volatility assessment and simple candle
  validation.
- Helper `MarketDataConverter` transforms OANDA structures to byte streams
  and bitstreams for CUDA processing.
- **Mock/placeholder aspects:** timestamp hashing and simplified
  normalization in `MarketDataConverter`, minimal optimization in ATR and
  volatility calculations, and order tracking utilities that expose limited
  error handling.

## C API

Files: `sep_c_api.*`, `sep.pc.in`

- Exposes the DSL interpreter through a C interface for embedding or
  scripting from non‑C++ languages.
- Supports executing source strings or files, fetching variable values and
  querying interpreter state.
- `sep.pc.in` supplies pkg‑config metadata for downstream build systems.
- **Mock/placeholder aspects:** the API targets alpha functionality and does
  not currently manage asynchronous execution or advanced type conversions.

