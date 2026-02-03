#!/bin/bash
# Export all marimo notebooks to HTML for GitHub Pages
#
# Usage: ./scripts/export_html.sh
#
# Outputs HTML files to docs/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NOTEBOOKS_DIR="$PROJECT_DIR/notebooks"
OUTPUT_DIR="$PROJECT_DIR/docs"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Exporting marimo notebooks to HTML..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Find all .py files in notebooks directory
for notebook in "$NOTEBOOKS_DIR"/*.py; do
    if [ -f "$notebook" ]; then
        # Get basename without extension
        basename=$(basename "$notebook" .py)
        output_file="$OUTPUT_DIR/${basename}.html"

        echo "Exporting: $basename.py -> $basename.html"

        # Run marimo export
        # Using --include-code to show source code in the exported HTML
        uv run marimo export html "$notebook" -o "$output_file" --include-code 2>&1 || {
            echo "  WARNING: Failed to export $basename.py (might have execution errors)"
            continue
        }

        echo "  Done: $output_file"
    fi
done

echo ""
echo "Export complete!"
echo "Files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/*.html 2>/dev/null || echo "  (no HTML files found)"
