#!/usr/bin/env bash
# Check that committed generated WGSL files match what would be generated

set -e

echo "Checking that generated WGSL files are up to date..."

# Store current state
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy current generated files
cp -r src/solver/gpu/shaders/generated "$TEMP_DIR/original"

# Regenerate files via build.rs
echo "Regenerating WGSL files..."
cargo build --quiet 2>&1 | grep -v "warning:" || true

# Compare
echo "Comparing generated files..."
DIFF_OUTPUT=$(diff -r "$TEMP_DIR/original" src/solver/gpu/shaders/generated || true)

if [ -n "$DIFF_OUTPUT" ]; then
    echo "ERROR: Generated WGSL files are out of date!"
    echo "The following differences were found:"
    echo "$DIFF_OUTPUT"
    echo ""
    echo "To fix this, run 'cargo build' and commit the updated files."
    exit 1
else
    echo "✓ All generated WGSL files are up to date"
fi
