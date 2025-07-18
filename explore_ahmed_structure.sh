#!/bin/bash
echo "🔍 Exploring Ahmed repository structure..."

TEMP_DIR="/data/ahmed_explore"
rm -rf "$TEMP_DIR"

# Clone and explore structure without downloading large files
git clone --depth 1 --no-checkout https://huggingface.co/datasets/neashton/ahmedml "$TEMP_DIR"
cd "$TEMP_DIR"

echo "📁 Repository contents:"
git ls-tree -r --name-only HEAD | head -20

echo ""
echo "📁 Run directories:"
git ls-tree -r --name-only HEAD | grep "run_" | cut -d'/' -f1 | sort -u | head -10

echo ""
echo "📄 STL files (first 5):"
git ls-tree -r --name-only HEAD | grep "\.stl" | head -5

echo ""
echo "📄 VTP files (first 5):"
git ls-tree -r --name-only HEAD | grep "\.vtp" | head -5

echo ""
echo "📄 Files in run_1:"
git ls-tree -r --name-only HEAD | grep "^run_1/" | head -10

# Clean up
cd /
rm -rf "$TEMP_DIR"
