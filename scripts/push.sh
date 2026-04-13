#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [LOCAL] [REMOTE]"
  echo "Example: $0 ~/workplaces/corrupted-jepa/ goat@turing.wpi.edu:~/corrupted-jepa"
  exit 1
fi

rsync -aPv \
  --exclude=".git" \
  --filter=":- .gitignore" \
  --timeout=300 \
  --protocol=31 \
  --partial \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o ConnectTimeout=60" \
  "$1" \
  "$2"
