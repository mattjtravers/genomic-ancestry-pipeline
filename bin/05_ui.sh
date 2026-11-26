#!/usr/bin/env bash
# Local preview of the static site at http://localhost:8000.
#
# Assembles site/ + samples/ with bin/assemble_site.py (the same build the CI
# deploy job publishes) into a temporary directory, using PHENO_API_ENDPOINT
# from the environment for config.js, and serves it until you press Ctrl-C.
#
# Port 8000 is fixed: http://localhost:8000 is the only local origin the API's
# CORS configuration allows (docs/llds/06_deployment.md § CORS). Without
# PHENO_API_ENDPOINT the preview is samples-only.
# @spec UI-UI-050
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP_DIR=$(mktemp -d)
SITE_DIR="${TMP_DIR}/site"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ -z "${PHENO_API_ENDPOINT:-}" ]]; then
    echo "WARNING: PHENO_API_ENDPOINT is not set; serving a samples-only preview" \
         "(live predictions will report the service as not configured)." >&2
fi

uv run python "${REPO}/bin/assemble_site.py" --out "$SITE_DIR" --endpoint "${PHENO_API_ENDPOINT:-}"

echo "Serving http://localhost:8000 (Ctrl-C to stop)"
uv run python -m http.server 8000 --directory "$SITE_DIR"
