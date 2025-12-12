#!/usr/bin/env bash
set -e

# Create build directory if it doesn't exist or reconfigure
cmake -B build -G Ninja

# Build the project
ninja -C build
