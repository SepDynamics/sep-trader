# SEP DSL Quick Start Tutorial

## Your First DSL Program

Let's build a simple pattern analysis program step by step.

### Step 1: Create a Data Stream

```sep
stream sensor_data from "temperature_readings.csv"
```

This tells the DSL to read data from a CSV file. The `sensor_data` identifier will be available throughout your program.

### Step 2: Define a Pattern

```sep
pattern temperature_spike {
    input: sensor_data
    
    high_temp = measure_coherence(sensor_data) > 0.8
    rapid_change = qfh_analyze(sensor_data) > 0.7
}
```

This pattern analyzes the sensor data and creates two boolean variables:
- `high_temp`: True when coherence measurement exceeds 0.8
- `rapid_change`: True when QFH analysis detects rapid changes

### Step 3: Create a Signal

```sep
signal alert_system {
    trigger: temperature_spike.high_temp
    confidence: temperature_spike.rapid_change
    action: SEND_ALERT
}
```

This signal will trigger when the pattern detects high temperature readings. The confidence is based on whether we also detect rapid changes.

### Complete Program

```sep
stream sensor_data from "temperature_readings.csv"

pattern temperature_spike {
    input: sensor_data
    
    high_temp = measure_coherence(sensor_data) > 0.8
    rapid_change = qfh_analyze(sensor_data) > 0.7
}

signal alert_system {
    trigger: temperature_spike.high_temp
    confidence: temperature_spike.rapid_change  
    action: SEND_ALERT
}
```

### Running Your Program

```bash
# Save the program as temperature_monitor.sep
./build/src/dsl/sep_dsl_interpreter temperature_monitor.sep
```

### Expected Output

```
=== SEP DSL Interpreter ===
Executing stream declaration: sensor_data from temperature_readings.csv
Executing pattern declaration: temperature_spike
Calling real measure_coherence with 1 arguments
Calling qfh_analyze with 1 arguments
Pattern temperature_spike captured 3 variables
Executing signal declaration: alert_system
Signal alert_system triggered! Action: SEND_ALERT
```

## Understanding the Output

1. **Stream Declaration**: The DSL sets up data ingestion
2. **Pattern Execution**: Built-in functions analyze the data
3. **Pattern Capture**: Variables are stored for signal access
4. **Signal Evaluation**: Trigger condition is checked and signal fires

## Next Steps

- Try changing the threshold values (0.8, 0.7) to see different behavior
- Add more patterns to analyze different aspects of your data
- Experiment with different built-in functions
- Create multiple signals from the same pattern

## Common Patterns

### Multiple Input Analysis
```sep
pattern comprehensive_check {
    input: sensor_data
    
    entropy_high = measure_entropy(sensor_data) > 0.6
    coherence_low = measure_coherence(sensor_data) < 0.4
    qfh_anomaly = qfh_analyze(sensor_data) > 0.9
    
    critical_state = entropy_high && coherence_low && qfh_anomaly
}
```

### Conditional Signals
```sep
signal emergency_alert {
    trigger: comprehensive_check.critical_state
    action: EMERGENCY
}

signal warning_alert {
    trigger: comprehensive_check.entropy_high && !comprehensive_check.critical_state
    action: WARNING
}
```

This tutorial covers the basics. The DSL is designed to be intuitive - if you can describe your analysis requirements in English, you can likely express them in SEP DSL!