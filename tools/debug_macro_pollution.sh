#!/usr/bin/env bash
# Trace macro definitions and header includes to identify std namespace pollution.
set -euo pipefail

SRC=${1:-src/app/cli_main.cpp}
TMP_DIR=$(mktemp -d)

echo "[macro] generating macro dump for $SRC"
gcc -E -dM "$SRC" > "$TMP_DIR/macros.txt"
rg -n "#define\s+(std|string|cout|ostream)\b" "$TMP_DIR/macros.txt" || echo "no std macro pollution detected"

echo "[include] generating include trace"
gcc -E -H "$SRC" > /dev/null 2> "$TMP_DIR/include_trace.txt"

echo "macro dump: $TMP_DIR/macros.txt"
echo "include trace: $TMP_DIR/include_trace.txt"
