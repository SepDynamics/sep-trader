#!/bin/bash

# SEP DSL Fuzz Testing Runner
# Integrates with existing build system and provides comprehensive fuzz testing

set -e

echo "🧪 SEP DSL Fuzz Testing Suite"
echo "=============================="

# Function to check dependencies
check_dependencies() {
    echo "📋 Checking dependencies..."
    
    if ! command -v clang++ &> /dev/null; then
        echo "❌ clang++ not found. Please install: sudo apt install clang"
        exit 1
    fi
    
    if ! command -v cmake &> /dev/null; then
        echo "❌ cmake not found. Please install: sudo apt install cmake"
        exit 1
    fi
    
    echo "✅ Dependencies satisfied"
}

# Function to build fuzz targets
build_fuzz_targets() {
    echo ""
    echo "🔨 Building fuzz targets..."
    
    # First ensure main build exists
    if [ ! -d "build" ]; then
        echo "Main build directory not found, running main build first..."
        ./build.sh
    fi
    
    # Build fuzz targets using Docker build system
    ./build_fuzz.sh
    
    echo "✅ Fuzz targets built successfully"
}

# Function to run quick fuzz tests
run_quick_tests() {
    echo ""
    echo "⚡ Running quick fuzz tests (30 seconds each)..."
    
    if [ ! -f "build/tests/fuzzing/fuzz_parser" ]; then
        echo "❌ Fuzz targets not found. Build may have failed."
        return 1
    fi
    
    echo "  Testing parser..."
    ./run_fuzz_docker.sh parser 30 2>/dev/null || true
    
    echo "  Testing interpreter..."  
    ./run_fuzz_docker.sh interpreter 30 2>/dev/null || true
    
    echo "✅ Quick fuzz tests completed"
}

# Function to run comprehensive fuzz tests
run_comprehensive_tests() {
    echo ""
    echo "🔬 Running comprehensive fuzz tests (5 minutes each)..."
    
    if [ ! -f "build/tests/fuzzing/fuzz_parser" ]; then
        echo "❌ Fuzz targets not found. Build may have failed."
        return 1
    fi
    
    echo "  Deep testing parser (5 min)..."
    ./run_fuzz_docker.sh parser 300 || true
    
    echo "  Deep testing interpreter (5 min)..."
    ./run_fuzz_docker.sh interpreter 300 || true
    
    echo "✅ Comprehensive fuzz tests completed"
}

# Function to generate fuzz report
generate_report() {
    echo ""
    echo "📊 Fuzz Testing Report"
    echo "====================="
    echo "Parser fuzz target: $(ls -la build/tests/fuzzing/fuzz_parser 2>/dev/null | awk '{print $5}') bytes"
    echo "Interpreter fuzz target: $(ls -la build/tests/fuzzing/fuzz_interpreter 2>/dev/null | awk '{print $5}') bytes"
    echo "Corpus files: $(find tests/fuzzing/corpus/ -name '*.sep' 2>/dev/null | wc -l)"
    echo ""
    echo "📝 To run manual fuzz testing:"
    echo "   ./run_fuzz_docker.sh parser 3600"
    echo "   ./run_fuzz_docker.sh interpreter 3600"
}

# Main execution
main() {
    case "${1:-quick}" in
        "quick")
            check_dependencies
            build_fuzz_targets
            run_quick_tests
            generate_report
            ;;
        "comprehensive"|"deep")
            check_dependencies
            build_fuzz_targets
            run_comprehensive_tests
            generate_report
            ;;
        "build-only")
            check_dependencies
            build_fuzz_targets
            generate_report
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [quick|comprehensive|build-only|help]"
            echo ""
            echo "Options:"
            echo "  quick          Run quick 30-second fuzz tests (default)"
            echo "  comprehensive  Run deep 5-minute fuzz tests"
            echo "  build-only     Just build fuzz targets"
            echo "  help           Show this help message"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
