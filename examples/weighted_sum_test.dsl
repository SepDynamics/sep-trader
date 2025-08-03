pattern test_weighted_sum {
    input: a, b, c
    
    result = weighted_sum {
        0.5: a,
        0.3: b,
        0.2: c
    }
}

signal test_signal {
    trigger: test_weighted_sum.result > 10.0
    confidence: 0.9
    action: "log_alert"
}