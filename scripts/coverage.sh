#!/usr/bin/env bash
set -e

# Ensure we are in a Nix environment
if [ -z "$IN_NIX_SHELL" ]; then
    echo "Not in Nix environment. Re-running inside Nix..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    "$SCRIPT_DIR/dev.sh" "$0" "$@"
    exit $?
fi

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

# 2b. Run Examples (Treating them as Integration Tests)
echo "Running examples to capture integration coverage..."
find "$BUILD_DIR/examples" -maxdepth 1 -type f -executable | while read -r example; do
    echo "Running $example..."
    "$example" > /dev/null 2>&1 || echo "Warning: $example failed"
done

# Determine GCOV tool
GCOV_TOOL=""
# Prioritize llvm-cov for Clang builds (common in Nix/LLVM environments)
if command -v llvm-cov &> /dev/null; then
    # Create wrapper script for llvm-cov gcov
    GCOV_WRAPPER="$BUILD_DIR/gcov_wrapper.sh"
    echo '#!/bin/sh' > "$GCOV_WRAPPER"
    echo 'exec llvm-cov gcov "$@"' >> "$GCOV_WRAPPER"
    chmod +x "$GCOV_WRAPPER"
    GCOV_TOOL="$GCOV_WRAPPER"
elif command -v gcov &> /dev/null; then
    GCOV_TOOL="gcov"
else
    echo "Error: Neither gcov nor llvm-cov found."
    exit 1
fi

echo "Capturing coverage data using $GCOV_TOOL..."
lcov --capture --directory "$BUILD_DIR" --output-file "$BUILD_DIR/coverage.info" --gcov-tool "$GCOV_TOOL" --ignore-errors mismatch,inconsistent,unsupported,format

# 4. Filter Usage
# Remove external libraries (Eigen, CasADi) and test files from potential coverage
# NOTE: Kept examples in the report as requested!
echo "Filtering coverage data..."
lcov --remove "$BUILD_DIR/coverage.info" \
    '/nix/*' \
    '/usr/*' \
    '*/tests/*' \
    '*/build/*' \
    --output-file "$BUILD_DIR/coverage_clean.info" \
    --ignore-errors mismatch,inconsistent,unsupported,format,unused

# 5. Generate Report
echo "Generating HTML report..."
genhtml "$BUILD_DIR/coverage_clean.info" --output-directory "$REPORT_DIR" --ignore-errors inconsistent,corrupt,unsupported,category

echo "Coverage report generated at: $REPORT_DIR/index.html"
