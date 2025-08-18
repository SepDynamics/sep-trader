# Strategy Development Guide

**Build custom trading strategies with SEP DSL**

This guide teaches you how to develop sophisticated trading strategies using the SEP DSL language, from basic patterns to advanced multi-timeframe systems.

## DSL Trading Strategy Fundamentals

### Pattern Structure

Every trading strategy in SEP DSL is defined as a **pattern**:

```sep
pattern my_trading_strategy {
    // 1. Data acquisition
    // 2. Analysis
    // 3. Signal generation
    // 4. Risk management
    // 5. Trade execution
}
```

### Core Building Blocks

#### 1. Data Acquisition
```sep
// Single timeframe
m15_data = fetch_live_oanda_data("EUR_USD", "M15", 100)

// Multiple timeframes
m1_data = fetch_live_oanda_data("EUR_USD", "M1", 500)
m5_data = fetch_live_oanda_data("EUR_USD", "M5", 200) 
m15_data = fetch_live_oanda_data("EUR_USD", "M15", 100)

// Multiple currency pairs
eur_usd = fetch_live_oanda_data("EUR_USD", "M15", 100)
gbp_usd = fetch_live_oanda_data("GBP_USD", "M15", 100)
usd_jpy = fetch_live_oanda_data("USD_JPY", "M15", 100)
```

#### 2. Quantum Analysis Functions
```sep
// Core analysis functions
coherence = measure_coherence(price_data)    // 0.0-1.0, higher = more structured
entropy = measure_entropy(price_data)        // 0.0-1.0, higher = more random  
stability = measure_stability(price_data)    // 0.0-1.0, higher = more stable

// Advanced functions
bits = extract_bits(price_data)              // Binary representation
qfh_result = qfh_analyze(bits)               // Quantum field harmonics
optimized = manifold_optimize(pattern, c, s) // Pattern optimization
```

#### 3. Signal Generation
```sep
// Simple signal
signal_strength = coherence * (1.0 - entropy) * stability

// Weighted signal (production settings)
signal_strength = (0.4 * stability) + (0.1 * coherence) + (0.5 * (1.0 - entropy))

// Multi-timeframe signal
m1_signal = measure_coherence(m1_data) * (1.0 - measure_entropy(m1_data))
m5_signal = measure_coherence(m5_data) * (1.0 - measure_entropy(m5_data))
m15_signal = measure_coherence(m15_data) * (1.0 - measure_entropy(m15_data))
combined_signal = (m1_signal + m5_signal + m15_signal) / 3.0
```

## Strategy Templates

### 1. Trend Following Strategy

```sep
pattern trend_following_eur_usd {
    // Multi-timeframe data
    m5_data = fetch_live_oanda_data("EUR_USD", "M5", 200)
    m15_data = fetch_live_oanda_data("EUR_USD", "M15", 100)
    h1_data = fetch_live_oanda_data("EUR_USD", "H1", 50)
    
    // Trend analysis
    m5_coherence = measure_coherence(m5_data)
    m15_coherence = measure_coherence(m15_data)
    h1_coherence = measure_coherence(h1_data)
    
    // Trend confirmation (all timeframes aligned)
    trend_strength = (m5_coherence + m15_coherence + h1_coherence) / 3.0
    trend_confirmed = (m5_coherence > 0.6) && (m15_coherence > 0.6) && (h1_coherence > 0.6)
    
    // Entry signal
    m5_entropy = measure_entropy(m5_data)
    entry_quality = trend_strength * (1.0 - m5_entropy)
    
    if (trend_confirmed && entry_quality > 0.65) {
        direction = h1_coherence > 0.75 ? "BUY" : "SELL"
        position_size = calculate_position_size(entry_quality)
        
        print("🎯 TREND SIGNAL:", direction, "Quality:", entry_quality)
        execute_trade("EUR_USD", direction, position_size)
    }
}
```

### 2. Mean Reversion Strategy

```sep
pattern mean_reversion_gbp_usd {
    // Short-term data for reversion signals
    m1_data = fetch_live_oanda_data("GBP_USD", "M1", 300)
    m5_data = fetch_live_oanda_data("GBP_USD", "M5", 100)
    
    // Measure volatility and stability
    m1_entropy = measure_entropy(m1_data)
    m5_stability = measure_stability(m5_data)
    
    // Mean reversion conditions
    high_volatility = m1_entropy > 0.7        // Market is chaotic
    stable_mean = m5_stability > 0.6           // But stable over longer term
    
    // Reversion signal
    reversion_probability = (1.0 - m1_entropy) * m5_stability
    
    if (high_volatility && stable_mean && reversion_probability > 0.4) {
        // Counter-trend entry
        m1_coherence = measure_coherence(m1_data)
        direction = m1_coherence > 0.5 ? "SELL" : "BUY"  // Fade the move
        
        print("🔄 REVERSION SIGNAL:", direction, "Probability:", reversion_probability)
        execute_trade("GBP_USD", direction, 1500)
    }
}
```

### 3. Breakout Strategy

```sep
pattern volatility_breakout {
    // Multiple pairs for breakout detection
    eur_usd = fetch_live_oanda_data("EUR_USD", "M15", 100)
    gbp_usd = fetch_live_oanda_data("GBP_USD", "M15", 100)
    usd_jpy = fetch_live_oanda_data("USD_JPY", "M15", 100)
    
    // Analyze each pair
    eur_entropy = measure_entropy(eur_usd)
    gbp_entropy = measure_entropy(gbp_usd)
    jpy_entropy = measure_entropy(usd_jpy)
    
    eur_coherence = measure_coherence(eur_usd)
    gbp_coherence = measure_coherence(gbp_usd)
    jpy_coherence = measure_coherence(usd_jpy)
    
    // Breakout detection
    function detect_breakout(entropy, coherence) {
        // High entropy (volatility) + high coherence (direction)
        return (entropy > 0.6) && (coherence > 0.7)
    }
    
    eur_breakout = detect_breakout(eur_entropy, eur_coherence)
    gbp_breakout = detect_breakout(gbp_entropy, gbp_coherence)
    jpy_breakout = detect_breakout(jpy_entropy, jpy_coherence)
    
    // Execute on strongest breakout
    if (eur_breakout) {
        print("🚀 EUR/USD BREAKOUT! Coherence:", eur_coherence)
        execute_trade("EUR_USD", "BUY", 2000)
    }
    
    if (gbp_breakout) {
        print("🚀 GBP/USD BREAKOUT! Coherence:", gbp_coherence)
        execute_trade("GBP_USD", "BUY", 1500)
    }
    
    if (jpy_breakout) {
        print("🚀 USD/JPY BREAKOUT! Coherence:", jpy_coherence)
        execute_trade("USD_JPY", "BUY", 2500)
    }
}
```

## Advanced Patterns

### 1. Multi-Strategy Portfolio

```sep
// Define individual strategies as patterns
pattern trend_strategy { /* trend logic */ }
pattern mean_reversion_strategy { /* reversion logic */ }
pattern breakout_strategy { /* breakout logic */ }

// Portfolio coordinator
pattern portfolio_manager {
    // Market condition analysis
    market_data = fetch_live_oanda_data("EUR_USD", "H1", 100)
    market_volatility = measure_entropy(market_data)
    market_trend = measure_coherence(market_data)
    
    // Strategy selection based on market conditions
    if (market_trend > 0.7 && market_volatility < 0.5) {
        // Trending market - use trend following
        print("📈 Trending market detected - using trend strategy")
        trend_strategy
    } else if (market_volatility > 0.7 && market_trend < 0.4) {
        // Volatile, directionless - use mean reversion
        print("🔄 Volatile market detected - using mean reversion")
        mean_reversion_strategy  
    } else if (market_volatility > 0.6 && market_trend > 0.6) {
        // High volatility + direction - use breakout
        print("🚀 Breakout conditions detected")
        breakout_strategy
    } else {
        print("⏳ No clear market condition - waiting")
    }
}
```

### 2. Adaptive Risk Management

```sep
pattern adaptive_risk_management {
    // Account metrics
    account_balance = get_account_balance()
    current_drawdown = get_current_drawdown()
    open_positions = get_position_count()
    
    // Market volatility
    eur_usd = fetch_live_oanda_data("EUR_USD", "H1", 50)
    market_entropy = measure_entropy(eur_usd)
    
    // Dynamic risk calculation
    base_risk = 0.02  // 2% base risk per trade
    
    // Reduce risk during high volatility
    volatility_adjustment = market_entropy > 0.6 ? 0.5 : 1.0
    
    // Reduce risk during drawdown
    drawdown_adjustment = current_drawdown > 0.05 ? 0.3 : 1.0
    
    // Reduce risk with many positions
    position_adjustment = open_positions > 3 ? 0.7 : 1.0
    
    // Final risk per trade
    adjusted_risk = base_risk * volatility_adjustment * drawdown_adjustment * position_adjustment
    
    print("💼 Risk Management:")
    print("  Base Risk:", base_risk)
    print("  Market Volatility:", market_entropy)
    print("  Current Drawdown:", current_drawdown)
    print("  Open Positions:", open_positions)
    print("  Adjusted Risk:", adjusted_risk)
    
    // Use this risk for position sizing
    return adjusted_risk
}
```

### 3. Correlation-Based Strategy

```sep
pattern correlation_strategy {
    // Major currency pairs
    eur_usd = fetch_live_oanda_data("EUR_USD", "M15", 100)
    gbp_usd = fetch_live_oanda_data("GBP_USD", "M15", 100)
    usd_chf = fetch_live_oanda_data("USD_CHF", "M15", 100)
    usd_jpy = fetch_live_oanda_data("USD_JPY", "M15", 100)
    
    // Individual analysis
    eur_coherence = measure_coherence(eur_usd)
    gbp_coherence = measure_coherence(gbp_usd)
    chf_coherence = measure_coherence(usd_chf)
    jpy_coherence = measure_coherence(usd_jpy)
    
    // Correlation analysis
    // EUR/USD and GBP/USD usually correlate positively
    eur_gbp_agreement = (eur_coherence > 0.6 && gbp_coherence > 0.6) ||
                       (eur_coherence < 0.4 && gbp_coherence < 0.4)
    
    // USD/CHF typically correlates negatively with EUR/USD
    eur_chf_divergence = (eur_coherence > 0.6 && chf_coherence < 0.4) ||
                        (eur_coherence < 0.4 && chf_coherence > 0.6)
    
    // High-confidence multi-pair signal
    if (eur_gbp_agreement && eur_chf_divergence) {
        print("🎯 CORRELATION CONFIRMED:")
        print("  EUR/USD Coherence:", eur_coherence)
        print("  GBP/USD Coherence:", gbp_coherence) 
        print("  USD/CHF Coherence:", chf_coherence)
        
        if (eur_coherence > 0.6) {
            execute_trade("EUR_USD", "BUY", 2000)
            execute_trade("GBP_USD", "BUY", 1500)
            execute_trade("USD_CHF", "SELL", 1800)
        } else {
            execute_trade("EUR_USD", "SELL", 2000)
            execute_trade("GBP_USD", "SELL", 1500)
            execute_trade("USD_CHF", "BUY", 1800)
        }
    }
}
```

## Strategy Optimization

### 1. Parameter Tuning

```sep
pattern optimizable_strategy {
    // Configurable parameters
    stability_weight = 0.4      // Tune this
    coherence_weight = 0.1      // Tune this  
    entropy_weight = 0.5        // Tune this
    
    confidence_threshold = 0.65  // Tune this
    coherence_threshold = 0.30   // Tune this
    
    // Your strategy logic using these parameters
    price_data = fetch_live_oanda_data("EUR_USD", "M15", 100)
    
    coherence = measure_coherence(price_data)
    entropy = measure_entropy(price_data)
    stability = measure_stability(price_data)
    
    signal_strength = (stability_weight * stability) + 
                     (coherence_weight * coherence) + 
                     (entropy_weight * (1.0 - entropy))
    
    if (signal_strength > confidence_threshold && coherence > coherence_threshold) {
        execute_trade("EUR_USD", "BUY", 1000)
    }
}
```

### 2. Backtesting Framework

```sep
pattern backtest_strategy {
    // Historical data for backtesting
    historical_data = fetch_historical_oanda_data("EUR_USD", "M15", "2024-01-01", "2024-12-31")
    
    // Strategy metrics
    total_trades = 0
    winning_trades = 0
    total_profit = 0.0
    
    // Apply strategy to historical data
    for (data_point in historical_data) {
        coherence = measure_coherence(data_point)
        entropy = measure_entropy(data_point)
        signal_strength = coherence * (1.0 - entropy)
        
        if (signal_strength > 0.65) {
            total_trades = total_trades + 1
            
            // Simulate trade outcome
            trade_result = simulate_trade_outcome(data_point)
            if (trade_result > 0) {
                winning_trades = winning_trades + 1
            }
            total_profit = total_profit + trade_result
        }
    }
    
    // Performance metrics
    win_rate = winning_trades / total_trades
    avg_profit_per_trade = total_profit / total_trades
    
    print("📊 Backtest Results:")
    print("  Total Trades:", total_trades)
    print("  Win Rate:", win_rate)
    print("  Avg Profit:", avg_profit_per_trade)
    print("  Total Profit:", total_profit)
}
```

## Best Practices

### 1. Strategy Structure
- **Keep patterns focused** - one strategy per pattern
- **Use descriptive names** for variables and patterns
- **Comment your logic** for future reference
- **Test incrementally** - start simple, add complexity

### 2. Risk Management
- **Always set position sizes** based on account balance
- **Include stop-loss logic** in every strategy
- **Monitor correlation** between strategies
- **Limit concurrent positions** to manage exposure

### 3. Testing & Validation
- **Start with demo account** for all new strategies
- **Backtest thoroughly** before live deployment
- **Monitor first 10 trades** manually
- **Track performance metrics** continuously

### 4. Performance Optimization
- **Use appropriate timeframes** for your strategy type
- **Minimize unnecessary calculations** in hot paths
- **Cache repeated data fetches** when possible
- **Profile strategy execution** time

## Common Patterns Reference

### Signal Generation Patterns
```sep
// High-confidence signal
high_conf_signal = (coherence > 0.7) && (entropy < 0.3) && (stability > 0.6)

// Momentum signal  
momentum_signal = coherence * (1.0 - entropy)

// Volatility-adjusted signal
vol_adj_signal = coherence * (1.0 - entropy) * sqrt(stability)

// Multi-timeframe confirmation
confirmed_signal = (m5_signal > 0.6) && (m15_signal > 0.6) && (h1_signal > 0.6)
```

### Risk Management Patterns
```sep
// Position sizing
function calculate_position_size(signal_strength, max_risk_percent) {
    account_balance = get_account_balance()
    max_risk_amount = account_balance * max_risk_percent
    confidence_multiplier = signal_strength > 0.8 ? 1.5 : 1.0
    return max_risk_amount * confidence_multiplier
}

// Stop loss calculation
function calculate_stop_loss(entry_price, direction, atr_value) {
    stop_distance = atr_value * 2.0
    if (direction == "BUY") {
        return entry_price - stop_distance
    } else {
        return entry_price + stop_distance
    }
}
```

### Market Condition Detection
```sep
// Trending market
trending_market = (coherence > 0.6) && (entropy < 0.5)

// Volatile market
volatile_market = (entropy > 0.7) && (coherence > 0.4)

// Quiet market
quiet_market = (entropy < 0.3) && (coherence < 0.4)

// Breakout conditions
breakout_setup = (entropy > 0.6) && (coherence > 0.7)
```

## Next Steps

1. **Start Simple**: Begin with the basic templates and modify them
2. **Test Thoroughly**: Use demo accounts and backtesting extensively  
3. **Monitor Performance**: Track win rates, drawdowns, and profitability
4. **Iterate Gradually**: Make small improvements based on results
5. **Scale Carefully**: Increase position sizes only after consistent performance

For more advanced topics, see:
- [Risk Management Guide](RISK_MANAGEMENT.md)
- [Pattern Library](PATTERN_LIBRARY.md)
- [Backtesting Guide](BACKTESTING_GUIDE.md)
- [Live Deployment Guide](LIVE_DEPLOYMENT.md)