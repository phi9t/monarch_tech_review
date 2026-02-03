#!/bin/bash
# Export all marimo notebooks to HTML for GitHub Pages
#
# Usage: ./scripts/export_html.sh
#
# Outputs HTML files to docs/ directory
# Then commit and push docs/ to have GitHub Pages serve them

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

        # Run marimo export (runs the notebook and captures output)
        uv run marimo export html "$notebook" -o "$output_file" --include-code 2>&1 || {
            echo "  WARNING: Failed to export $basename.py (might have execution errors)"
            continue
        }

        echo "  Done: $output_file"
    fi
done

# Create index page
echo ""
echo "Creating index.html..."

cat > "$OUTPUT_DIR/index.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
  <title>Monarch GPU Mode Notebooks</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
    h1 { color: #333; }
    ul { list-style: none; padding: 0; }
    li { margin: 15px 0; }
    a { color: #0066cc; text-decoration: none; font-size: 1.1em; }
    a:hover { text-decoration: underline; }
    .desc { color: #666; font-size: 0.9em; margin-left: 20px; }
  </style>
</head>
<body>
  <h1>Monarch GPU Mode Notebooks</h1>
  <p>Interactive notebooks for the GPU Mode presentation on Monarch.</p>
  <ul>
EOF

for f in "$OUTPUT_DIR"/*.html; do
    if [ "$(basename "$f")" != "index.html" ]; then
        name=$(basename "$f" .html)
        echo "    <li><a href=\"${name}.html\">${name}</a></li>" >> "$OUTPUT_DIR/index.html"
    fi
done

cat >> "$OUTPUT_DIR/index.html" << 'EOF'
  </ul>
</body>
</html>
EOF

echo ""
echo "Export complete!"
echo ""
echo "Files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/*.html 2>/dev/null || echo "  (no HTML files found)"
echo ""
echo "Next steps:"
echo "  1. git add docs/"
echo "  2. git commit -m 'Update notebook exports'"
echo "  3. git push"
echo "  4. Enable GitHub Pages: Settings → Pages → Source: 'Deploy from branch' → 'main' → '/docs'"
