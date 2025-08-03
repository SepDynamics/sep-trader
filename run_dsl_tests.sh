#!/bin/bash

echo "Building and running DSL tests..."

# Build the project
./build.sh

echo ""
echo "Running DSL Parser Tests..."
echo "=========================="
./build/tests/dsl_parser_test

echo ""
echo "Running DSL Interpreter Tests..."
echo "================================"
./build/tests/dsl_interpreter_test

echo ""
echo "DSL Test Suite Complete!"
