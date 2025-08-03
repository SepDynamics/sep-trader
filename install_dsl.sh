#!/bin/bash
echo "Installing SEP DSL Interpreter..."

# First, build the project to ensure the executable is up-to-date
./build.sh

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Cannot install."
    exit 1
fi

# Check if the executable exists
if [ ! -f "./build/src/dsl/sep_dsl_interpreter" ]; then
    echo "❌ DSL interpreter executable not found."
    exit 1
fi

# Copy the interpreter to a system-wide location
echo "Installing DSL interpreter to /usr/local/bin/sep..."
sudo cp ./build/src/dsl/sep_dsl_interpreter /usr/local/bin/sep

if [ $? -eq 0 ]; then
    echo "✅ SEP DSL Interpreter installed as 'sep'."
    echo ""
    echo "Usage:"
    echo "  sep your_script.sep             # Run a script"
    echo "  ./your_script.sep               # Run executable script with shebang"
    echo ""
    echo "To make scripts executable, add this shebang line at the top:"
    echo "  #!/usr/bin/env sep"
    echo ""
    echo "Then make them executable with:"
    echo "  chmod +x your_script.sep"
else
    echo "❌ Installation failed. You may need sudo privileges."
    exit 1
fi
