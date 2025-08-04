#!/bin/bash
# SEP Dynamics Professional Packaging Script

set -e

VERSION=${1:-"v1.0.0"}
PACKAGE_DIR="release"
ENGINE_PACKAGE="sep-engine-${VERSION}"
WEBSITE_PACKAGE="sepdynamics-website-${VERSION}"

echo "🚀 SEP Dynamics Professional Packaging"
echo "Version: ${VERSION}"
echo "========================================"

# Clean previous builds
rm -rf ${PACKAGE_DIR}
mkdir -p ${PACKAGE_DIR}

echo "📦 Building SEP Engine..."
./build.sh

echo "🧪 Running Test Suite..."
./build/tests/test_forward_window_metrics
./build/tests/trajectory_metrics_test  
./build/tests/pattern_metrics_test
./build/tests/quantum_signal_bridge_test
./build/src/apps/oanda_trader/quantum_tracker --test

echo "✅ All tests passed!"

echo "📁 Packaging SEP Engine..."
mkdir -p ${PACKAGE_DIR}/${ENGINE_PACKAGE}

# Core engine files
cp -r build/src ${PACKAGE_DIR}/${ENGINE_PACKAGE}/
cp -r build/examples ${PACKAGE_DIR}/${ENGINE_PACKAGE}/
cp -r build/tests ${PACKAGE_DIR}/${ENGINE_PACKAGE}/

# Documentation
cp -r docs ${PACKAGE_DIR}/${ENGINE_PACKAGE}/
cp README.md ${PACKAGE_DIR}/${ENGINE_PACKAGE}/
cp COMMERCIAL.md ${PACKAGE_DIR}/${ENGINE_PACKAGE}/
cp LICENSE ${PACKAGE_DIR}/${ENGINE_PACKAGE}/

# Configuration and scripts
cp build.sh ${PACKAGE_DIR}/${ENGINE_PACKAGE}/
cp install.sh ${PACKAGE_DIR}/${ENGINE_PACKAGE}/
cp CMakeLists.txt ${PACKAGE_DIR}/${ENGINE_PACKAGE}/
cp -r cmake ${PACKAGE_DIR}/${ENGINE_PACKAGE}/

# Create engine tarball
cd ${PACKAGE_DIR}
tar -czf ${ENGINE_PACKAGE}.tar.gz ${ENGINE_PACKAGE}/
cd ..

echo "🌐 Building Website..."
cd website
npm ci
npm run build
cd ..

echo "📁 Packaging Website..."
mkdir -p ${PACKAGE_DIR}/${WEBSITE_PACKAGE}
cp -r website/dist/* ${PACKAGE_DIR}/${WEBSITE_PACKAGE}/

# Create website tarball
cd ${PACKAGE_DIR}
tar -czf ${WEBSITE_PACKAGE}.tar.gz ${WEBSITE_PACKAGE}/
cd ..

echo "📊 Package Summary:"
echo "==================="
ls -lh ${PACKAGE_DIR}/*.tar.gz

echo ""
echo "✅ Professional packaging complete!"
echo "🎯 Ready for commercial deployment"
echo ""
echo "Engine Package: ${PACKAGE_DIR}/${ENGINE_PACKAGE}.tar.gz"
echo "Website Package: ${PACKAGE_DIR}/${WEBSITE_PACKAGE}.tar.gz"
echo ""
echo "🚀 SEP Dynamics - Quantum-Inspired Financial Intelligence"
echo "Patent Application: 584961162ABX"
echo "Contact: alex@sepdynamics.com"
