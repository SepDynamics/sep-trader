# Live Currency Pair Training System

## Overview

The SEP Engine now includes a live training system that optimizes quantum field harmonics parameters for specific currency pairs using real-time OANDA market data. This eliminates the need for static historical files and ensures optimization is based on current market conditions.

## Key Features

- **Live Data Integration**: Uses OANDA API to fetch real-time market data for training
- **Quantum Field Optimization**: Optimizes stability, coherence, and entropy weights
- **Threshold Tuning**: Finds optimal confidence and coherence thresholds
- **Currency Pair Specific**: Customized optimization for each trading pair
- **Production Ready**: Integrates with the live quantum_tracker system

## Usage

### Basic Usage

```bash
# Train EUR_USD with 48 hours of recent data
python train_currency_pair.py EUR_USD

# Train GBP_USD with 72 hours of recent data
python train_currency_pair.py GBP_USD --hours 72

# Quick optimization (fewer test combinations)
python train_currency_pair.py EUR_USD --quick
```

### Specialized Training

```bash
# Only optimize weights (stability, coherence, entropy)
python train_currency_pair.py EUR_USD --weights-only

# Only optimize thresholds (confidence, coherence)
python train_currency_pair.py EUR_USD --thresholds-only

# Quick threshold optimization
python train_currency_pair.py EUR_USD --thresholds-only --quick
```

## Prerequisites

### 1. OANDA Account Setup

You need OANDA API credentials:

```bash
export OANDA_API_KEY='your_oanda_api_key'
export OANDA_ACCOUNT_ID='your_oanda_account_id'
```

### 2. Build the System

```bash
./build.sh
```

## How It Works

### Phase 1: Weight Optimization

The system tests different combinations of quantum field weights:
- **Stability Weight**: Controls pattern stability detection
- **Coherence Weight**: Controls signal coherence filtering  
- **Entropy Weight**: Controls entropy-based pattern analysis

**Optimization Process**:
1. Modifies quantum signal bridge source code
2. Rebuilds the system with new weights
3. Runs live analysis on recent market data
4. Measures profitability score: `(High-Conf Accuracy - 50) × Signal Rate`
5. Finds optimal weight combination

### Phase 2: Threshold Optimization

Using the best weights, optimizes filtering thresholds:
- **Confidence Threshold**: Minimum confidence for signal generation
- **Coherence Threshold**: Minimum coherence for pattern acceptance

**Optimization Process**:
1. Uses optimal weights from Phase 1
2. Tests different threshold combinations
3. Maximizes profitability while maintaining signal quality
4. Finds optimal balance between accuracy and frequency

## Output Files

Results are saved in `/sep/training_results/{PAIR_NAME}/`:

- `weight_optimization_YYYYMMDD_HHMMSS.json` - Weight optimization results
- `threshold_optimization_YYYYMMDD_HHMMSS.json` - Threshold optimization results  
- `optimization_summary_YYYYMMDD_HHMMSS.json` - Combined final configuration

### Example Output

```json
{
  "pair": "EUR_USD",
  "hours_analyzed": 48,
  "final_configuration": {
    "weights": {
      "stability": 0.40,
      "coherence": 0.10, 
      "entropy": 0.50
    },
    "thresholds": {
      "confidence": 0.65,
      "coherence": 0.30
    },
    "performance": {
      "overall_accuracy": 41.83,
      "high_conf_accuracy": 60.73,
      "signal_rate": 19.1,
      "profitability_score": 204.94
    }
  }
}
```

## Implementation in Production

After training, implement the optimal configuration:

```cpp
// In quantum_signal_bridge.cpp
double stability_weight = 0.40;
double coherence_weight = 0.10;
double entropy_weight = 0.50;

double confidence_threshold = 0.65;
double coherence_threshold = 0.30;
```

## Supported Currency Pairs

The system works with any OANDA-supported currency pair:
- EUR_USD, GBP_USD, USD_JPY, AUD_USD
- EUR_GBP, EUR_JPY, GBP_JPY, AUD_JPY
- USD_CAD, NZD_USD, EUR_AUD, etc.

## Advanced Features

### Quick Mode
- Uses fewer test combinations for faster optimization
- Suitable for rapid iteration during development
- Trade-off between speed and thoroughness

### Live Market Integration
- Automatically handles market hours and weekend closures
- Falls back to cached data when markets are closed
- Uses real tick data for most accurate training

### Safety Features
- Always restores original source files after training
- Includes timeout protection for API calls
- Validates OANDA credentials before starting
- Comprehensive error handling and logging

## Performance Expectations

Typical training performance:
- **Full Optimization**: 30-60 minutes (both weights and thresholds)
- **Quick Optimization**: 10-20 minutes
- **Weights Only**: 15-30 minutes
- **Thresholds Only**: 10-20 minutes

Results typically achieve:
- **Overall Accuracy**: 35-45%
- **High-Confidence Accuracy**: 50-65%
- **Signal Rate**: 15-25%
- **Profitability Score**: 100-250+

## Troubleshooting

### Common Issues

1. **"OANDA credentials required"**
   - Set OANDA_API_KEY and OANDA_ACCOUNT_ID environment variables

2. **"Build failed"**
   - Run `./build.sh` manually to see detailed error messages
   - Ensure CUDA and all dependencies are installed

3. **"Market closed/No data"**
   - System automatically falls back to cached data
   - Training will still complete successfully

4. **"Timeout errors"**
   - Try reducing --hours parameter (e.g., --hours 24)
   - Check internet connection and OANDA API status

### Debug Mode

For detailed debugging, run with verbose output:

```bash
OANDA_API_KEY=your_key OANDA_ACCOUNT_ID=your_id python train_currency_pair.py EUR_USD --quick 2>&1 | tee training.log
```

This system represents the production implementation of the quantum field harmonics patent, providing adaptive optimization for real-world trading environments.
