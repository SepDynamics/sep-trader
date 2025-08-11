# Trace Logging

The trading pipeline exposes lightweight trace hooks for debugging data flow.

## Enabling Tracing
Define `SEP_ENABLE_TRACE` at compile time to activate logging:
```
cmake -DSEP_ENABLE_TRACE=ON ...
```
When enabled, calls to `sep::testbed::trace` emit messages to `std::clog`.

## Instrumented Stages
Current instrumentation covers:
- `ingest`: market data received for analysis
- `feature extraction`: quantum features computed from data
- `signal emission`: trading signal determined

These hooks aid step‑wise inspection of analysis without impacting optimized builds.
