#!/usr/bin/env bash
set -e

# Ensure we are in a Nix environment
if [ -z "$IN_NIX_SHELL" ]; then
    echo "Not in Nix environment. Re-running inside Nix..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    "$SCRIPT_DIR/dev.sh" "$0" "$@"
    exit $?
fi

# Ensure logs directory exists
mkdir -p logs

./scripts/run_examples.sh 2>&1 | tee logs/examples.log
echo "Examples Complete. Logs available at logs/examples.log"
