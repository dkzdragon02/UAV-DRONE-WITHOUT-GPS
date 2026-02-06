#!/bin/bash
# Test runner script for uav_vision package

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Running Tests for uav_vision ==="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest not found. Install with: pip3 install pytest pytest-cov"
    exit 1
fi

# Run tests
echo "Running unit tests..."
pytest test/ -v --tb=short

echo ""
echo "=== Tests Complete ==="

