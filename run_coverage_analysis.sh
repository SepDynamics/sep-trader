#!/bin/bash

# Code coverage analysis script using gcov/lcov
# Targets >90% coverage for production quality

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
    echo -e "${BLUE}[Coverage]${NC} $1"
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

# Create output directory for coverage results
mkdir -p output/coverage

# Function to build with coverage flags
build_with_coverage() {
    print_status "Building with code coverage instrumentation..."
    
    # Create coverage build script
    cat > build_coverage.sh << 'EOF'
#!/bin/bash
set -e

# Build with coverage flags
export CXXFLAGS="-g -O0 --coverage -fprofile-arcs -ftest-coverage"
export LDFLAGS="--coverage"

mkdir -p build_coverage
cd build_coverage

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-g -O0 --coverage -fprofile-arcs -ftest-coverage" \
    -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
    -DCMAKE_SHARED_LINKER_FLAGS="--coverage"

make -j$(nproc)
EOF
    
    chmod +x build_coverage.sh
    
    # Install coverage tools in Docker if needed
    if ! docker run --rm sep_build_env:latest which lcov > /dev/null 2>&1; then
        print_status "Adding coverage tools to Dockerfile..."
        if ! grep -q "lcov" Dockerfile; then
            sed -i '/RUN apt-get update/a\    lcov \\' Dockerfile
            sed -i '/lcov \\/a\    gcovr \\' Dockerfile
            print_status "Rebuilding Docker image with coverage tools..."
            docker build -t sep_build_env:latest .
        fi
    fi
    
    # Build with coverage in Docker
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest ./build_coverage.sh
}

# Function to run tests with coverage collection
run_tests_with_coverage() {
    print_status "Running tests with coverage collection..."
    
    # Reset coverage counters
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        lcov --directory build_coverage --zerocounters || true
    
    # Run DSL unit tests
    print_status "Running DSL unit tests..."
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        bash -c "cd build_coverage && ctest --output-on-failure --verbose" || true
    
    # Run DSL interpreter tests with sample programs
    print_status "Running DSL interpreter integration tests..."
    for sep_file in examples/*.sep; do
        if [[ -f "$sep_file" ]]; then
            print_status "Testing: $(basename "$sep_file")"
            docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
                ./build_coverage/src/dsl/sep_dsl_interpreter "$sep_file" || true
        fi
    done
    
    # Run fuzz tests for additional coverage
    print_status "Running fuzz tests for additional coverage..."
    timeout 30s docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        ./build_coverage/tests/fuzzing/fuzz_parser /workspace/tests/fuzzing/corpus/basic_pattern.sep || true
    
    timeout 30s docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        ./build_coverage/tests/fuzzing/fuzz_interpreter /workspace/tests/fuzzing/corpus/basic_pattern.sep || true
}

# Function to generate coverage reports
generate_coverage_reports() {
    print_status "Generating coverage reports..."
    
    # Generate lcov coverage data
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        lcov --directory build_coverage --capture --output-file output/coverage/coverage.info || true
    
    # Filter out external libraries and test files
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        lcov --remove output/coverage/coverage.info \
        '*/extern/*' '*/third_party/*' '*/tests/*' '*/examples/*' '/usr/*' \
        --output-file output/coverage/coverage_filtered.info || true
    
    # Generate HTML report
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        genhtml output/coverage/coverage_filtered.info \
        --output-directory output/coverage/html \
        --title "SEP DSL Coverage Report" \
        --num-spaces 4 \
        --sort \
        --function-coverage \
        --branch-coverage || true
    
    # Generate text summary
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        lcov --summary output/coverage/coverage_filtered.info > output/coverage/summary.txt 2>&1 || true
    
    # Generate gcovr reports for additional formats
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        gcovr --root . \
        --exclude '.*extern.*' \
        --exclude '.*third_party.*' \
        --exclude '.*tests.*' \
        --exclude '.*examples.*' \
        --print-summary \
        --html --html-details \
        --output output/coverage/gcovr_report.html \
        build_coverage || true
    
    # Generate XML report for CI integration
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        gcovr --root . \
        --exclude '.*extern.*' \
        --exclude '.*third_party.*' \
        --exclude '.*tests.*' \
        --exclude '.*examples.*' \
        --xml \
        --output output/coverage/coverage.xml \
        build_coverage || true
}

# Function to analyze coverage results
analyze_coverage() {
    print_status "Analyzing coverage results..."
    
    local line_coverage=0
    local function_coverage=0
    local branch_coverage=0
    local target_coverage=90
    
    # Extract coverage percentages from summary
    if [[ -f output/coverage/summary.txt ]]; then
        line_coverage=$(grep "lines" output/coverage/summary.txt | grep -o '[0-9.]*%' | head -1 | sed 's/%//' || echo "0")
        function_coverage=$(grep "functions" output/coverage/summary.txt | grep -o '[0-9.]*%' | head -1 | sed 's/%//' || echo "0")
        branch_coverage=$(grep "branches" output/coverage/summary.txt | grep -o '[0-9.]*%' | head -1 | sed 's/%//' || echo "0")
    fi
    
    print_status "Coverage Results:"
    print_status "  Lines: ${line_coverage}%"
    print_status "  Functions: ${function_coverage}%"
    print_status "  Branches: ${branch_coverage}%"
    
    # Check if we meet the target
    if (( $(echo "$line_coverage >= $target_coverage" | bc -l) )); then
        print_success "Line coverage (${line_coverage}%) meets target (${target_coverage}%)"
    else
        print_warning "Line coverage (${line_coverage}%) below target (${target_coverage}%)"
    fi
    
    if (( $(echo "$function_coverage >= $target_coverage" | bc -l) )); then
        print_success "Function coverage (${function_coverage}%) meets target (${target_coverage}%)"
    else
        print_warning "Function coverage (${function_coverage}%) below target (${target_coverage}%)"
    fi
    
    # Identify uncovered files
    print_status "Identifying files with low coverage..."
    docker run --rm -v "$PWD:/workspace" -w /workspace sep_build_env:latest \
        lcov --list output/coverage/coverage_filtered.info | \
        awk '$4 < 90 && $4 != "-" {print $1 ": " $4}' > output/coverage/low_coverage_files.txt || true
    
    if [[ -s output/coverage/low_coverage_files.txt ]]; then
        print_warning "Files with <90% coverage:"
        cat output/coverage/low_coverage_files.txt
    else
        print_success "All files meet 90% coverage target!"
    fi
}

# Function to generate coverage improvement suggestions
generate_suggestions() {
    print_status "Generating coverage improvement suggestions..."
    
    cat > output/coverage/improvement_suggestions.md << EOF
# Coverage Improvement Suggestions

Generated: $(date)

## Current Coverage Status

- **Line Coverage**: ${line_coverage:-Unknown}%
- **Function Coverage**: ${function_coverage:-Unknown}%
- **Branch Coverage**: ${branch_coverage:-Unknown}%
- **Target**: 90%

## Areas for Improvement

### Files with Low Coverage
$(if [[ -s output/coverage/low_coverage_files.txt ]]; then
    cat output/coverage/low_coverage_files.txt
else
    echo "No files below 90% coverage threshold"
fi)

### Recommended Actions

1. **Add more unit tests** for uncovered functions
2. **Test error conditions** to improve branch coverage
3. **Add integration tests** for end-to-end scenarios
4. **Test edge cases** and boundary conditions
5. **Mock external dependencies** to test error paths

### Test Categories to Expand

- [ ] Parser error recovery tests
- [ ] Interpreter exception handling tests
- [ ] Memory allocation failure tests
- [ ] File I/O error tests
- [ ] CUDA/GPU error condition tests
- [ ] Network timeout/failure tests (if applicable)
- [ ] Invalid input format tests
- [ ] Resource exhaustion tests

### Tools and Reports

- **HTML Report**: output/coverage/html/index.html
- **XML Report**: output/coverage/coverage.xml (for CI)
- **Detailed Report**: output/coverage/gcovr_report.html
- **Raw Data**: output/coverage/coverage_filtered.info

EOF
    
    print_success "Suggestions generated: output/coverage/improvement_suggestions.md"
}

# Function to integrate with CI/CD
setup_ci_integration() {
    print_status "Setting up CI/CD integration..."
    
    # Create GitHub Actions workflow for coverage
    mkdir -p .github/workflows
    
    cat > .github/workflows/coverage.yml << 'EOF'
name: Code Coverage Analysis

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  coverage:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive
    
    - name: Build Docker image
      run: docker build -t sep_build_env:latest .
    
    - name: Run coverage analysis
      run: ./run_coverage_analysis.sh
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./output/coverage/coverage.xml
        flags: unittests
        name: sep-dsl-coverage
        fail_ci_if_error: true
    
    - name: Upload HTML report
      uses: actions/upload-artifact@v3
      with:
        name: coverage-report
        path: output/coverage/html/
    
    - name: Comment PR with coverage
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          if (fs.existsSync('output/coverage/summary.txt')) {
            const summary = fs.readFileSync('output/coverage/summary.txt', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 📊 Coverage Report\n\`\`\`\n${summary}\n\`\`\``
            });
          }
EOF
    
    print_success "CI/CD workflow created: .github/workflows/coverage.yml"
}

# Main execution
main() {
    print_status "🧪 Starting comprehensive code coverage analysis"
    
    # Ensure Docker image exists
    if ! docker images | grep -q "sep_build_env"; then
        print_status "Building Docker image..."
        ./build.sh
    fi
    
    case "${1:-all}" in
        "build")
            build_with_coverage
            ;;
        "test")
            run_tests_with_coverage
            ;;
        "report")
            generate_coverage_reports
            analyze_coverage
            generate_suggestions
            ;;
        "ci")
            setup_ci_integration
            ;;
        "all"|*)
            build_with_coverage
            run_tests_with_coverage
            generate_coverage_reports
            analyze_coverage
            generate_suggestions
            setup_ci_integration
            ;;
    esac
    
    print_success "🎯 Code coverage analysis completed!"
    print_status "📊 Open output/coverage/html/index.html to view detailed report"
    print_status "📄 Check output/coverage/improvement_suggestions.md for actionable items"
}

# Show usage if help requested
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [build|test|report|ci|all]"
    echo ""
    echo "Options:"
    echo "  build    Build project with coverage instrumentation only"
    echo "  test     Run tests with coverage collection only"
    echo "  report   Generate coverage reports and analysis only"
    echo "  ci       Setup CI/CD integration only"
    echo "  all      Run complete coverage analysis (default)"
    echo ""
    echo "Results will be saved to output/coverage/"
    echo "Target: >90% line and function coverage"
    exit 0
fi

main "$@"
