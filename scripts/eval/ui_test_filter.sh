#!/bin/bash
# UI Test Results Analyzer Wrapper
# Usage: ./ui_test_filter.sh <ui_test_results_dir>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/ui_test_filter.py" "$@"
