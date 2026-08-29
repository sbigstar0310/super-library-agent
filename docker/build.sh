#!/bin/bash
# Build a benchmark agent image.
#
# Usage: bash docker/build.sh <image-tag>
#   image-tag must match a subdirectory under docker/ that contains a Dockerfile.
#
# Example: bash docker/build.sh sla-base

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <image-tag>"
    echo "Available:"
    find "$(dirname "$0")" -maxdepth 2 -name Dockerfile -printf '  %h\n' | sed "s|$(dirname "$0")/||"
    exit 1
fi

tag="$1"
ctx_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/$tag" && pwd)"

if [[ ! -f "$ctx_dir/Dockerfile" ]]; then
    echo "No Dockerfile at $ctx_dir/Dockerfile"
    exit 1
fi

echo "[build.sh] building $tag from $ctx_dir"
exec docker build -t "$tag" "$ctx_dir"
