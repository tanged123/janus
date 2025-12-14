#!/usr/bin/env bash
set -e

# Ensure logs directory exists
mkdir -p logs

# Run build and test scripts inside the nix environment
echo "Running CI under Nix..."
nix develop --command bash -c "./scripts/build.sh --clean && ./scripts/test.sh" 2>&1 | tee logs/ci.log
echo "CI Complete. Logs available at logs/ci.log"
