pattern base_pattern {
    a = 10
    b = 20
}

pattern child_pattern inherits base_pattern {
    c = a + b
}

signal test_signal {
    trigger: child_pattern.c == 30;
    confidence: 1.0;
    action: "print_success";
}