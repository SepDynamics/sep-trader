#!/bin/bash

# Memory leak detection script using Valgrind and AddressSanitizer
# Integrates with existing Docker build system for consistent testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[Memory Test]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Create output directory for memory test results
mkdir -p output/memory_tests

# Function to run Valgrind tests
run_valgrind_tests() {
    print_status "Running Valgrind memory leak detection..."
    
    # Check if Valgrind is available in Docker
    if docker run --rm sep_build_env:latest which valgrind > /dev/null 2>&1; then
        print_status "Valgrind found in Docker container"
    else
        print_warning "Valgrind not found in Docker. Installing..."
        # Update Dockerfile to include Valgrind if needed
        if ! grep -q "valgrind" Dockerfile; then
            print_status "Adding Valgrind to Dockerfile..."
            sed -i '/RUN apt-get update/a\    valgrind \\' Dockerfile
            print_status "Rebuilding Docker image with Valgrind..."
            docker build -t sep-dsl-dev:latest .
        fi
    fi
    
    # Test DSL interpreter with Valgrind
    print_status "Testing DSL interpreter for memory leaks..."
    docker run --rm -v "$PWD:/workspace" sep_build_env:latest \
        valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
        --track-origins=yes --verbose --log-file=/workspace/output/memory_tests/valgrind_interpreter.log \
        /workspace/build/src/dsl/sep_dsl_interpreter /workspace/examples/agi_demo_simple.sep || true
    
    # Test DSL parser with Valgrind
    print_status "Testing DSL parser for memory leaks..."
    docker run --rm -v "$PWD:/workspace" sep_build_env:latest \
        valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
        --track-origins=yes --verbose --log-file=/workspace/output/memory_tests/valgrind_parser.log \
        /workspace/build/tests/dsl/unit/dsl_parser_test || true
    
    # Analyze Valgrind results
    analyze_valgrind_results
}

# Function to build with AddressSanitizer
build_with_asan() {
    print_status "Building with AddressSanitizer..."
    
    # Create ASAN build script
    cat > build_asan.sh << 'EOF'
#!/bin/bash
set -e

# Build with AddressSanitizer flags
export CXXFLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
export LDFLAGS="-fsanitize=address"

mkdir -p build_asan
cd build_asan

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address"

make -j$(nproc)
EOF
    
    chmod +x build_asan.sh
    
    # Build with ASAN in Docker
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest ./build_asan.sh
}

# Function to run AddressSanitizer tests
run_asan_tests() {
    print_status "Running AddressSanitizer tests..."
    
    # Test DSL interpreter with ASAN
    print_status "Testing DSL interpreter with AddressSanitizer..."
    docker run --rm -v "$PWD:/workspace" -w /workspace \
        -e ASAN_OPTIONS="detect_leaks=1:abort_on_error=0:log_path=/workspace/output/memory_tests/asan_interpreter" \
        sep_build_env:latest \
        ./build_asan/src/dsl/sep_dsl_interpreter examples/agi_demo_simple.sep || true
    
    # Test DSL parser with ASAN
    print_status "Testing DSL parser with AddressSanitizer..."
    docker run --rm -v "$PWD:/workspace" -w /workspace \
        -e ASAN_OPTIONS="detect_leaks=1:abort_on_error=0:log_path=/workspace/output/memory_tests/asan_parser" \
        sep_build_env:latest \
        ./build_asan/tests/dsl/unit/dsl_parser_test || true
    
    # Test all unit tests with ASAN
    print_status "Running all DSL tests with AddressSanitizer..."
    docker run --rm -v "$PWD:/workspace" -w /workspace \
        -e ASAN_OPTIONS="detect_leaks=1:abort_on_error=0:log_path=/workspace/output/memory_tests/asan_all_tests" \
        sep_build_env:latest \
        bash -c "cd build_asan && ctest --output-on-failure" || true
    
    # Analyze ASAN results
    analyze_asan_results
}

# Function to analyze Valgrind results
analyze_valgrind_results() {
    print_status "Analyzing Valgrind results..."
    
    local leak_count=0
    local error_count=0
    
    for log_file in output/memory_tests/valgrind_*.log; do
        if [[ -f "$log_file" ]]; then
            local leaks=$(grep -c "definitely lost\|indirectly lost\|possibly lost" "$log_file" || echo "0")
            local errors=$(grep -c "Invalid read\|Invalid write\|Conditional jump" "$log_file" || echo "0")
            
            leak_count=$((leak_count + leaks))
            error_count=$((error_count + errors))
            
            print_status "$(basename "$log_file"): $leaks leaks, $errors errors"
        fi
    done
    
    if [[ $leak_count -eq 0 && $error_count -eq 0 ]]; then
        print_success "Valgrind: No memory leaks or errors detected!"
    else
        print_error "Valgrind: Found $leak_count potential leaks and $error_count errors"
    fi
}

# Function to analyze AddressSanitizer results
analyze_asan_results() {
    print_status "Analyzing AddressSanitizer results..."
    
    local issue_count=0
    
    for log_file in output/memory_tests/asan_*; do
        if [[ -f "$log_file" ]]; then
            local issues=$(grep -c "ERROR: AddressSanitizer\|ERROR: LeakSanitizer" "$log_file" || echo "0")
            issue_count=$((issue_count + issues))
            
            if [[ $issues -gt 0 ]]; then
                print_error "$(basename "$log_file"): $issues issues found"
                # Show first few lines of issues for debugging
                grep -A 5 "ERROR: AddressSanitizer\|ERROR: LeakSanitizer" "$log_file" | head -20
            else
                print_success "$(basename "$log_file"): No issues detected"
            fi
        fi
    done
    
    if [[ $issue_count -eq 0 ]]; then
        print_success "AddressSanitizer: No memory issues detected!"
    else
        print_error "AddressSanitizer: Found $issue_count memory issues"
    fi
}

# Function to generate memory test report
generate_report() {
    print_status "Generating memory test report..."
    
    cat > output/memory_tests/report.md << EOF
# Memory Leak Detection Report

Generated: $(date)

## Test Summary

### Valgrind Results
$(if [[ -f output/memory_tests/valgrind_interpreter.log ]]; then
    echo "- DSL Interpreter: $(grep -c "definitely lost\|indirectly lost\|possibly lost" output/memory_tests/valgrind_interpreter.log || echo "0") potential leaks"
fi)
$(if [[ -f output/memory_tests/valgrind_parser.log ]]; then
    echo "- DSL Parser: $(grep -c "definitely lost\|indirectly lost\|possibly lost" output/memory_tests/valgrind_parser.log || echo "0") potential leaks"
fi)

### AddressSanitizer Results
$(for log_file in output/memory_tests/asan_*; do
    if [[ -f "$log_file" ]]; then
        local issues=$(grep -c "ERROR: AddressSanitizer\|ERROR: LeakSanitizer" "$log_file" || echo "0")
        echo "- $(basename "$log_file"): $issues issues"
    fi
done)

## Detailed Results

### Valgrind Logs
$(ls -la output/memory_tests/valgrind_*.log 2>/dev/null || echo "No Valgrind logs found")

### AddressSanitizer Logs  
$(ls -la output/memory_tests/asan_* 2>/dev/null || echo "No ASAN logs found")

## Recommendations

1. Review any detected leaks or errors in the detailed logs
2. Focus on fixing "definitely lost" leaks first
3. Investigate any AddressSanitizer errors immediately
4. Run tests regularly in CI/CD pipeline

EOF
    
    print_success "Report generated: output/memory_tests/report.md"
}

# Main execution
main() {
    print_status "🧪 Starting comprehensive memory leak detection"
    
    # Ensure Docker image exists
    if ! docker images | grep -q "sep_build_env"; then
        print_status "Building Docker image..."
        ./build.sh
    fi
    
    # Run both Valgrind and ASAN tests
    case "${1:-all}" in
        "valgrind")
            run_valgrind_tests
            ;;
        "asan")
            build_with_asan
            run_asan_tests
            ;;
        "all"|*)
            run_valgrind_tests
            build_with_asan  
            run_asan_tests
            ;;
    esac
    
    # Generate comprehensive report
    generate_report
    
    print_success "🎯 Memory leak detection completed!"
    print_status "📊 Check output/memory_tests/ for detailed results"
}

# Show usage if help requested
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [valgrind|asan|all]"
    echo ""
    echo "Options:"
    echo "  valgrind    Run only Valgrind memory leak detection"
    echo "  asan        Run only AddressSanitizer tests"
    echo "  all         Run both Valgrind and ASAN (default)"
    echo ""
    echo "Results will be saved to output/memory_tests/"
    exit 0
fi

main "$@"
