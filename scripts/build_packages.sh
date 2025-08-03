#!/bin/bash

# SEP DSL Package Build Script
# Builds and prepares packages for distribution

set -e

echo "🚀 Building SEP DSL packages for distribution..."

# Build core library first
echo "📦 Building core SEP DSL library..."
cd /sep
./build.sh

# Build Python package
echo "🐍 Building Python package..."
cd /sep/bindings/python
python setup.py sdist bdist_wheel
echo "✅ Python package built in dist/"

# Build JavaScript package
echo "📦 Building JavaScript package..."
cd /sep/bindings/javascript
npm install
npm run build
npm pack
echo "✅ JavaScript package built (sep-dsl-*.tgz)"

# Build Language Server
echo "🔧 Building Language Server..."
cd /sep/tools/lsp
npm install
npm run build
npm pack
echo "✅ Language Server built (sep-dsl-language-server-*.tgz)"

# Build VSCode Extension
echo "🎨 Building VSCode Extension..."
cd /sep/tools/vscode-extension
npm install
npm run compile
# npm run package  # Requires vsce to be installed
echo "✅ VSCode Extension compiled"

# Build Ruby gem
echo "💎 Building Ruby gem..."
cd /sep/bindings/ruby
gem build sep_dsl.gemspec
echo "✅ Ruby gem built (sep_dsl-*.gem)"

echo ""
echo "🎉 All packages built successfully!"
echo ""
echo "📂 Package locations:"
echo "  Python:     /sep/bindings/python/dist/"
echo "  JavaScript: /sep/bindings/javascript/sep-dsl-*.tgz"
echo "  Ruby:       /sep/bindings/ruby/sep_dsl-*.gem"
echo "  LSP:        /sep/tools/lsp/sep-dsl-language-server-*.tgz"
echo "  VSCode:     /sep/tools/vscode-extension/ (compiled)"
echo ""
echo "📋 Next steps:"
echo "  - Test packages locally"
echo "  - Upload to package registries (npm, PyPI, RubyGems)"
echo "  - Publish VSCode extension to marketplace"
