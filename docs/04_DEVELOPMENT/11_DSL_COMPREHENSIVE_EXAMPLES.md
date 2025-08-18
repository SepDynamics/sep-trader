# SEP DSL Comprehensive Examples

This document contains complete DSL examples demonstrating multi-domain pattern analysis capabilities.

## Multi-Domain Analysis Example

This demonstrates how the same DSL constructs work across different domains:

```sep
// Comprehensive Multi-Domain Analysis Example
// This demonstrates how the same DSL constructs work across different domains

// Financial Market Analysis
stream market_data from "financial/EUR_USD_M5.csv"

pattern market_volatility {
    input: market_data
    
    price_entropy = measure_entropy(market_data) > 0.65
    trend_rupture = qfh_analyze(market_data) > 0.75
    coherence_break = measure_coherence(market_data) < 0.4
    
    volatile_market = price_entropy && trend_rupture
    unstable_trend = coherence_break && trend_rupture
}

signal trading_alert {
    trigger: market_volatility.volatile_market
    confidence: market_volatility.unstable_trend
    action: TRADE_ALERT
}

// Scientific Experiment Monitoring  
stream experiment_readings from "lab/quantum_measurement.json"

pattern quantum_decoherence {
    input: experiment_readings
    
    coherence_loss = measure_coherence(experiment_readings) < 0.3
    entropy_spike = measure_entropy(experiment_readings) > 0.8
    field_disruption = qfh_analyze(experiment_readings) > 0.9
    
    decoherence_event = coherence_loss && entropy_spike
    critical_disruption = decoherence_event && field_disruption
}

signal experiment_flag {
    trigger: quantum_decoherence.decoherence_event
    confidence: quantum_decoherence.critical_disruption
    action: RECALIBRATE
}

// Medical Monitoring
stream patient_vitals from "medical/ecg_monitor.csv"

pattern cardiac_anomaly {
    input: patient_vitals
    
    irregular_rhythm = measure_entropy(patient_vitals) > 0.7
    coherence_drop = measure_coherence(patient_vitals) < 0.5
    pattern_break = qfh_analyze(patient_vitals) > 0.8
    
    arrhythmia_detected = irregular_rhythm && coherence_drop
    emergency_pattern = arrhythmia_detected && pattern_break
}

signal medical_alert {
    trigger: cardiac_anomaly.arrhythmia_detected
    confidence: cardiac_anomaly.emergency_pattern
    action: MEDICAL_ALERT
}

// IoT Sensor Network
stream sensor_network from "iot/environmental_sensors.json"

pattern environmental_anomaly {
    input: sensor_network
    
    sensor_divergence = measure_coherence(sensor_network) < 0.4
    rapid_fluctuation = qfh_analyze(sensor_network) > 0.7
    high_noise = measure_entropy(sensor_network) > 0.6
    
    sensor_malfunction = sensor_divergence && high_noise
    environmental_event = rapid_fluctuation && !sensor_malfunction
}

signal maintenance_required {
    trigger: environmental_anomaly.sensor_malfunction
    action: MAINTENANCE
}

signal environmental_alert {
    trigger: environmental_anomaly.environmental_event
    confidence: environmental_anomaly.rapid_fluctuation
    action: INVESTIGATE
}
```

## Key Concepts Demonstrated

### 1. Domain Agnostic Analysis
The same quantum analysis functions (`measure_coherence`, `measure_entropy`, `qfh_analyze`) work effectively across:
- **Financial Markets**: Detecting volatility and trend breaks
- **Scientific Experiments**: Monitoring quantum decoherence
- **Medical Monitoring**: Identifying cardiac arrhythmias  
- **IoT Networks**: Distinguishing sensor malfunctions from environmental events

### 2. Pattern Composition
Each pattern combines multiple quantum metrics to detect domain-specific conditions:
- **Boolean Logic**: Using `&&`, `||`, `!` operators for complex conditions
- **Threshold Analysis**: Comparing metrics against domain-appropriate thresholds
- **Multi-Factor Decisions**: Combining multiple signals for confident detection

### 3. Signal Generation
Signals provide actionable outputs based on pattern analysis:
- **Trigger Conditions**: When to activate the signal
- **Confidence Levels**: How certain we are about the signal
- **Action Specification**: What action to take when triggered

### 4. Real-World Applications
This example shows how AGI-level pattern recognition can be applied to:
- **Trading Systems**: Automated market analysis and alert generation
- **Laboratory Equipment**: Real-time experiment monitoring and control
- **Healthcare Devices**: Patient monitoring and emergency detection
- **Smart Cities**: Environmental monitoring and predictive maintenance

## Running the Example

```bash
# Save the example as multi_domain_analysis.sep
./build/src/dsl/sep_dsl_interpreter multi_domain_analysis.sep
```

## Expected Behavior

The interpreter will:
1. Set up data streams from various sources
2. Execute pattern analysis on each data stream
3. Generate appropriate signals based on detected patterns
4. Output action recommendations for each domain

This demonstrates the power of the SEP DSL: **one language, infinite applications**.