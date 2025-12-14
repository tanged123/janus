#!/usr/bin/env bash
set -e

# Setup directories
BUILD_DIR="build/coverage"
REPORT_DIR="$BUILD_DIR/html"
mkdir -p "$BUILD_DIR"

echo "=== Code Coverage Generation ==="
echo "Build Directory: $BUILD_DIR"

# 1. Configure with Coverage Enabled
cmake -B "$BUILD_DIR" -S . -DENABLE_COVERAGE=ON -G Ninja

# 2. Build and Test
cmake --build "$BUILD_DIR"
CTEST_OUTPUT_ON_FAILURE=1 cmake --build "$BUILD_DIR" --target test

# Determine GCOV tool
if command -v gcov &> /dev/null; then
    GCOV_TOOL="gcov"
elif command -v llvm-cov &> /dev/null; then
    # Create wrapper script for llvm-cov gcov
    GCOV_WRAPPER="$BUILD_DIR/gcov_wrapper.sh"
    echo '#!/bin/sh' > "$GCOV_WRAPPER"
    echo 'exec llvm-cov gcov "$@"' >> "$GCOV_WRAPPER"
    chmod +x "$GCOV_WRAPPER"
    GCOV_TOOL="$GCOV_WRAPPER"
else
    echo "Error: Neither gcov nor llvm-cov found."
    exit 1
fi

echo "Capturing coverage data using $GCOV_TOOL..."
lcov --capture --directory "$BUILD_DIR" --output-file "$BUILD_DIR/coverage.info" --gcov-tool "$GCOV_TOOL" --ignore-errors mismatch,inconsistent,unsupported,format

# 4. Filter Usage
# Remove external libraries (Eigen, CasADi) and test files from potential coverage
echo "Filtering coverage data..."
lcov --remove "$BUILD_DIR/coverage.info" \
    '/nix/*' \
    '/usr/*' \
    '*/tests/*' \
    '*/examples/*' \
    '*/examples/*' \
    '*/build/*' \
    --output-file "$BUILD_DIR/coverage_clean.info" \
    --ignore-errors mismatch,inconsistent,unsupported,format,unused

# 5. Generate Report
echo "Generating HTML report..."
genhtml "$BUILD_DIR/coverage_clean.info" --output-directory "$REPORT_DIR" --ignore-errors inconsistent,corrupt,unsupported,category

echo "Coverage report generated at: $REPORT_DIR/index.html"
