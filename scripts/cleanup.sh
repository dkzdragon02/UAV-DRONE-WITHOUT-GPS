#!/bin/bash
# Cleanup script - Remove unnecessary files

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=== Cleaning Up Unnecessary Files ==="
echo ""

cd "$PROJECT_DIR"

# Remove Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pytest_cache" -delete 2>/dev/null || true

# Remove test artifacts
echo "Removing test artifacts..."
find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true
find . -type f -name ".coverage" -delete 2>/dev/null || true
find . -type f -name "coverage.xml" -delete 2>/dev/null || true

# Remove Python package artifacts
echo "Removing Python package artifacts..."
find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
find . -type d -name "dist" -exec rm -r {} + 2>/dev/null || true
find . -type d -name "build" -path "*/ros2_ws/src/*" -exec rm -r {} + 2>/dev/null || true

# Remove editor files
echo "Removing editor files..."
find . -type f -name "*.swp" -delete 2>/dev/null || true
find . -type f -name "*.swo" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
find . -type f -name "*.tmp" -delete 2>/dev/null || true
find . -type f -name "*.log" -path "*/test/*" -delete 2>/dev/null || true

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Note: ROS2 build/ and install/ directories are kept (needed for ROS2)"
echo "To clean build artifacts, run: cd ros2_ws && rm -rf build install log"

