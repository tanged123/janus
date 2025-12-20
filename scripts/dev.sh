#!/usr/bin/env bash
# Enter the Janus development environment
if [ $# -eq 0 ]; then
    nix develop
else
    nix develop --command "$@"
fi
