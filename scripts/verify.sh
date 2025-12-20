#!/usr/bin/env bash
set -e

# Ensure we are in a Nix environment
if [ -z "$IN_NIX_SHELL" ]; then
    echo "Not in Nix environment. Re-running inside Nix..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    "$SCRIPT_DIR/dev.sh" "$0" "$@"
    exit $?
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure logs directory exists
mkdir -p "$PROJECT_ROOT/logs"

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/verify_${TIMESTAMP}.log"

./scripts/build.sh --clean && ./scripts/test.sh && ./scripts/run_examples.sh 2>&1 | tee "$LOG_FILE"

# Create symlink to latest
ln -sf "verify_${TIMESTAMP}.log" "$PROJECT_ROOT/logs/verify.log"

echo "Verification Complete. Logs available at logs/verify_${TIMESTAMP}.log (symlinked to logs/verify.log)"
