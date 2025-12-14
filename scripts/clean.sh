#!/usr/bin/env bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_ROOT/build"
LOGS_DIR="$PROJECT_ROOT/logs"

if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    echo "Done."
else
    echo "Build directory does not exist: $BUILD_DIR"
fi

if [ -d "$LOGS_DIR" ]; then
    echo "Cleaning logs directory: $LOGS_DIR"
    rm -rf "$LOGS_DIR"
    echo "Done."
else
    echo "Logs directory does not exist: $LOGS_DIR"
fi
