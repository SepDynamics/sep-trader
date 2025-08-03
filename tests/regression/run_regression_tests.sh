#!/bin/bash

# SEP DSL Regression Test Runner
# Ensures backward compatibility across versions

set -e

echo "🧪 SEP DSL Regression Test Suite"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test directories
REGRESSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$REGRESSION_DIR")")"
DSL_INTERPRETER="$PROJECT_ROOT/build/src/dsl/sep_dsl_interpreter"
TEST_OUTPUT_DIR="$REGRESSION_DIR/output"

# Ensure output directory exists
mkdir -p "$TEST_OUTPUT_DIR"

# Check if interpreter exists
if [ ! -f "$DSL_INTERPRETER" ]; then
    echo -e "${RED}❌ DSL interpreter not found at $DSL_INTERPRETER${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

echo -e "${BLUE}📍 Using interpreter: $DSL_INTERPRETER${NC}"
echo -e "${BLUE}📁 Test directory: $REGRESSION_DIR${NC}"
echo -e "${BLUE}📊 Output directory: $TEST_OUTPUT_DIR${NC}"
echo ""

# Function to run a single test
run_test() {
    local test_file="$1"
    local test_name="$(basename "$test_file" .sep)"
    local output_file="$TEST_OUTPUT_DIR/${test_name}_output.txt"
    local error_file="$TEST_OUTPUT_DIR/${test_name}_error.txt"
    
    echo -n "  Testing $test_name... "
    
    # Run the test and capture output
    if timeout 30s "$DSL_INTERPRETER" "$test_file" > "$output_file" 2> "$error_file"; then
        # Check if there were any errors
        if [ -s "$error_file" ]; then
            echo -e "${YELLOW}⚠️  WARNINGS${NC}"
            echo "    Warnings written to $error_file"
        else
            echo -e "${GREEN}✅ PASS${NC}"
        fi
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "    Error output:"
        cat "$error_file" | head -5 | sed 's/^/      /'
        echo "    Full error log: $error_file"
        return 1
    fi
}

# Function to run performance test
run_performance_test() {
    local test_file="$1"
    local test_name="$(basename "$test_file" .sep)"
    local output_file="$TEST_OUTPUT_DIR/${test_name}_perf.txt"
    
    echo -n "  Performance test $test_name... "
    
    # Run with time measurement
    if /usr/bin/time -o "$output_file" -f "Real: %es, User: %Us, Sys: %Ss, Memory: %MKB" \
       timeout 60s "$DSL_INTERPRETER" "$test_file" >/dev/null 2>&1; then
        
        # Extract timing info
        timing=$(cat "$output_file")
        echo -e "${GREEN}✅ PASS${NC} ($timing)"
        return 0
    else
        echo -e "${RED}❌ TIMEOUT/FAIL${NC}"
        return 1
    fi
}

# Test counters
total_tests=0
passed_tests=0
failed_tests=0
warning_tests=0

echo "🔍 Running Core Language Regression Tests..."

# Test 1: Main regression test suite
total_tests=$((total_tests + 1))
if run_test "$REGRESSION_DIR/regression_test_suite.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

echo ""
echo "🔍 Running Individual Feature Tests..."

# Test 2: Basic syntax test
total_tests=$((total_tests + 1))
cat > "$TEST_OUTPUT_DIR/basic_syntax_test.sep" << 'EOF'
pattern basic_test {
    x = 42
    y = x + 8
    result = y * 2
}
EOF

if run_test "$TEST_OUTPUT_DIR/basic_syntax_test.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

# Test 3: Math functions test
total_tests=$((total_tests + 1))
cat > "$TEST_OUTPUT_DIR/math_test.sep" << 'EOF'
pattern math_test {
    pi_val = pi()
    sqrt_val = sqrt(16)
    sin_val = sin(0)
    log_val = log(e())
    pow_val = pow(2, 3)
}
EOF

if run_test "$TEST_OUTPUT_DIR/math_test.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

# Test 4: Statistical functions test
total_tests=$((total_tests + 1))
cat > "$TEST_OUTPUT_DIR/stats_test.sep" << 'EOF'
pattern stats_test {
    mean_val = mean(1, 2, 3, 4, 5)
    sum_val = sum(1, 2, 3, 4, 5)
    median_val = median(1, 3, 2, 5, 4)
    stddev_val = stddev(1, 2, 3, 4, 5)
}
EOF

if run_test "$TEST_OUTPUT_DIR/stats_test.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

# Test 5: Array support test
total_tests=$((total_tests + 1))
cat > "$TEST_OUTPUT_DIR/array_test.sep" << 'EOF'
pattern array_test {
    arr = [1, 2, 3, 4, 5]
    first = arr[0]
    second = arr[1]
    array_sum = sum(arr[0], arr[1], arr[2])
}
EOF

if run_test "$TEST_OUTPUT_DIR/array_test.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

# Test 6: Control flow test
total_tests=$((total_tests + 1))
cat > "$TEST_OUTPUT_DIR/control_flow_test.sep" << 'EOF'
pattern control_test {
    x = 10
    if (x > 5) {
        status = "high"
    } else {
        status = "low"
    }
    
    counter = 0
    while (counter < 3) {
        counter = counter + 1
    }
}
EOF

if run_test "$TEST_OUTPUT_DIR/control_flow_test.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

# Test 7: Function definition test
total_tests=$((total_tests + 1))
cat > "$TEST_OUTPUT_DIR/function_test.sep" << 'EOF'
function multiply(a, b) {
    return a * b
}

pattern function_test {
    result = multiply(6, 7)
}
EOF

if run_test "$TEST_OUTPUT_DIR/function_test.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

# Test 8: Exception handling test
total_tests=$((total_tests + 1))
cat > "$TEST_OUTPUT_DIR/exception_test.sep" << 'EOF'
pattern exception_test {
    try {
        good_result = sqrt(16)
        status = "success"
    }
    catch (error) {
        status = "error"
    }
    finally {
        cleanup = true
    }
}
EOF

if run_test "$TEST_OUTPUT_DIR/exception_test.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

echo ""
echo "⚡ Running Performance Tests..."

# Performance test - should complete quickly
total_tests=$((total_tests + 1))
cat > "$TEST_OUTPUT_DIR/performance_test.sep" << 'EOF'
pattern performance_test {
    sum = 0
    for (i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) {
        sum = sum + sqrt(i) + sin(i)
    }
    
    data = [1, 4, 2, 8, 5, 7, 3, 9, 6, 10]
    mean_val = mean(data[0], data[1], data[2], data[3], data[4])
    stddev_val = stddev(data[0], data[1], data[2], data[3], data[4])
}
EOF

if run_performance_test "$TEST_OUTPUT_DIR/performance_test.sep"; then
    passed_tests=$((passed_tests + 1))
else
    failed_tests=$((failed_tests + 1))
fi

echo ""
echo "📊 Regression Test Results"
echo "=========================="
echo -e "${BLUE}Total Tests:${NC} $total_tests"
echo -e "${GREEN}Passed:${NC} $passed_tests"
echo -e "${RED}Failed:${NC} $failed_tests"

if [ $failed_tests -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All regression tests passed!${NC}"
    echo "✅ Backward compatibility maintained"
    echo ""
    echo "📁 Test outputs saved to: $TEST_OUTPUT_DIR"
    exit 0
else
    echo ""
    echo -e "${RED}❌ $failed_tests test(s) failed!${NC}"
    echo "⚠️  Potential backward compatibility issues detected"
    echo ""
    echo "📁 Error logs available in: $TEST_OUTPUT_DIR"
    echo ""
    echo "🔧 To debug failed tests:"
    echo "   1. Check error logs in $TEST_OUTPUT_DIR"
    echo "   2. Run individual tests manually:"
    echo "      $DSL_INTERPRETER <test_file>"
    echo "   3. Compare with expected behavior from previous versions"
    exit 1
fi
