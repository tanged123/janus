#!/usr/bin/env bash
# Install git hooks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_DIR="$(git rev-parse --git-dir)"

echo "Installing git hooks..."

# Copy pre-commit hook
cp "$SCRIPT_DIR/../.github/hooks/pre-commit" "$GIT_DIR/hooks/pre-commit"
chmod +x "$GIT_DIR/hooks/pre-commit"

echo "✅ Git hooks installed successfully!"
echo "   Pre-commit hook will auto-format code with 'nix fmt'"
