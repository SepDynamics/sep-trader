#!/bin/bash

# Find all files that use nlohmann but don't include our safe header
FILES=$(grep -l "nlohmann::" src/ -r --include="*.cpp" --include="*.h" --include="*.hpp" | \
        xargs grep -L "nlohmann_json_safe.h")

echo "Fixing nlohmann includes in files:"
for file in $FILES; do
    echo "  $file"
    # Add safe include at the top after any existing includes
    if grep -q "#include.*nlohmann" "$file"; then
        # Replace direct nlohmann includes with our safe wrapper
        sed -i 's|#include <nlohmann/json.hpp>|#include "nlohmann_json_safe.h"|g' "$file"
    else
        # Add safe include before first #include line
        sed -i '1i #include "nlohmann_json_safe.h"' "$file"
    fi
done
