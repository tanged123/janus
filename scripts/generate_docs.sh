#!/bin/bash
set -e

# Ensure doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "Doxygen could not be found. Please install it."
    exit 1
fi

echo "Generating documentation..."
mkdir -p build/docs
doxygen Doxyfile

# Copy interactive HTML examples and user guides
echo "Copying interactive examples..."
cp -r docs/examples build/docs/html/examples 2>/dev/null || true
cp -r docs/user_guides build/docs/html/user_guides 2>/dev/null || true

echo "Documentation generated in build/docs/html"
