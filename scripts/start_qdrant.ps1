# Start Qdrant server — run this once before starting the app
# It serves both rufus_products and rufus_clip from existing local storage.
# The server manages memory properly; local file mode loads ~8 GB into RAM.

$bin = Join-Path $PSScriptRoot "..\data\qdrant_bin\qdrant.exe"
$cfg = Join-Path $PSScriptRoot "..\data\qdrant_bin\config.yaml"

if (-not (Test-Path $bin)) {
    Write-Error "Qdrant binary not found: $bin"
    Write-Host "Run from the repo root: uv run python scripts/download_datasets.py (or re-run setup)"
    exit 1
}

Write-Host "Starting Qdrant server on http://localhost:6333 ..."
Write-Host "Press Ctrl+C to stop."
& $bin --config-path $cfg
