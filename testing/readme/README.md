# Testing and Real Data Protocols

This directory describes how to run SEP's real data validation scripts and what
outputs to expect. **All procedures enforce a strict zero‑tolerance policy for
synthetic or fabricated market data.** Only authentic records from the OANDA
Practice API are acceptable.

## Shell Scripts

### `real_data_validation.sh`
Collects and verifies two weeks of historical forex data and performs a short
live‑stream test.

**Key steps**
1. Loads OANDA credentials from `OANDA.env`.
2. Fetches 14 days of candles for major pairs (`EUR_USD`, `GBP_USD`, `USD_JPY`,
   `AUD_USD`, `USD_CHF`) across multiple granularities.
3. Writes JSON files and fetch logs under `testing/real_data/historical/` and
   `testing/real_data/validation_reports/`.
4. Generates a markdown integrity report summarising file sizes, record counts
   and timestamp ranges.
5. Executes a 5‑minute live‑stream test to verify real‑time connectivity.

**Expected output**
- `testing/real_data/historical/*.json` – authentic candle data.
- `testing/real_data/validation_reports/` – logs and the integrity report.
- `testing/real_data/live_streams/` – live stream captures when enabled.

### `retail_data_validation.sh`
Validates the broader retail development kit using only real market data.

**Key steps**
1. Checks system status and confirms OANDA API connectivity.
2. Validates the existing real‑data cache and fetches fresh weekly data for a
   wider set of major pairs.
3. Runs quantum processing tests on the retrieved data.
4. Produces authenticity evidence in `validation/retail_kit_proof/`.

**Expected output**
- `validation/real_data_logs/` – system and cache validation logs.
- `validation/weekly_validation/<PAIR>/` – per‑pair weekly fetch logs.
- `validation/quantum_tests/` and `validation/pair_analysis/` – quantum
  processing results.
- `validation/retail_kit_proof/data_authenticity_report_<DATE>.md` – final report
  confirming that only genuine OANDA data was used.

## Directory Overview

### `examples/`
Contains simple `.sep` patterns (`test_basic.sep`, `test_fixed.sep`) to verify the
DSL runtime. These examples are purely demonstrative and do **not** use market
prices.

### `real_data/`
Created automatically by `real_data_validation.sh` to store authentic market
records and validation reports. This directory is not version‑controlled and
should contain **only** genuine OANDA outputs.

---
Running either script without legitimate OANDA credentials or by introducing
synthetic data violates project policy and invalidates all results.
