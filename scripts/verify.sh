#!/usr/bin/env bash
set -e

# Ensure logs directory exists
mkdir -p logs

# Run full verification pipeline: Build -> Test -> Examples
echo "Running Full Verification (Build + Tests + Examples) under Nix..."
nix develop --command bash -c "./scripts/build.sh --clean && ./scripts/test.sh && ./scripts/run_examples.sh" 2>&1 | tee logs/verify.log
echo "Verification Complete. Logs available at logs/verify.log"
