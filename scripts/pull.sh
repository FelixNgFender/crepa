#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [REMOTE] [LOCAL]"
  echo "Example: $0 goat@turing.wpi.edu:~/crepa ~/workplaces/crepa"
  exit 1
fi

rsync -aPvz \
  --exclude=".git" \
  --filter=":- .gitignore" "$1" "$2"
