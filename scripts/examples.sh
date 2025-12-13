#!/usr/bin/env bash
set -e

# Ensure logs directory exists
mkdir -p logs

# Run examples inside the nix environment
echo "Running Examples under Nix..."
nix develop --command bash -c "./scripts/run_examples.sh" 2>&1 | tee logs/examples.log
echo "Examples Complete. Logs available at logs/examples.log"
