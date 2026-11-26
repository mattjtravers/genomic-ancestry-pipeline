#!/usr/bin/env bash
# Precomputes the bundled samples' results against the deployed API.
#
# Posts each samples/sample_snp_N.vcf to POST ${PHENO_API_ENDPOINT}/predict and
# writes the response body, unchanged, to samples/sample_snp_N.result.json. The
# static site renders these instantly when a visitor picks a sample, so they
# must come from the model that is actually deployed: re-run this after every
# 04_deploy.sh that binds a new ModelRunId, then commit the result files.
#
# All-or-nothing: responses are collected in a temp directory and the result
# files are replaced only if every sample returned 200, so a failure partway
# through never leaves results from two different models side by side.
#
# SAMPLES_DIR overrides the samples directory (used by the tests).
# @spec UI-UI-040, UI-UI-058
set -euo pipefail

: "${PHENO_API_ENDPOINT:?Set PHENO_API_ENDPOINT (printed by 04_deploy.sh) before running this script}"

SAMPLES_DIR="${SAMPLES_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/samples}"
URL="${PHENO_API_ENDPOINT%/}/predict"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

shopt -s nullglob
VCFS=()
for vcf in "$SAMPLES_DIR"/sample_snp_*.vcf; do
    [[ "$(basename "$vcf")" =~ ^sample_snp_[0-9]+\.vcf$ ]] && VCFS+=("$vcf")
done
if (( ${#VCFS[@]} == 0 )); then
    echo "ERROR: no sample_snp_N.vcf files in ${SAMPLES_DIR}" >&2
    exit 1
fi

for vcf in "${VCFS[@]}"; do
    name=$(basename "$vcf" .vcf)
    python3 -c 'import json, sys; json.dump({"vcf": open(sys.argv[1], encoding="utf-8").read()}, sys.stdout)' \
        "$vcf" > "$TMP_DIR/${name}.request.json"

    status=$(curl -s -o "$TMP_DIR/${name}.result.json" -w '%{http_code}' --max-time 35 \
        -X POST "$URL" -H 'Content-Type: application/json' \
        --data-binary @"$TMP_DIR/${name}.request.json" || true)

    if [[ "$status" != "200" ]]; then
        echo "ERROR: ${name} -> HTTP ${status:-000}; no result files were changed." >&2
        [[ -s "$TMP_DIR/${name}.result.json" ]] && cat "$TMP_DIR/${name}.result.json" >&2 && echo >&2
        exit 1
    fi
    python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); print(sys.argv[2] + ":", r.get("predicted_phenotype", "?"), "(" + str(r.get("model_artifact_version")) + ")")' \
        "$TMP_DIR/${name}.result.json" "$name"
done

for vcf in "${VCFS[@]}"; do
    name=$(basename "$vcf" .vcf)
    mv "$TMP_DIR/${name}.result.json" "$SAMPLES_DIR/${name}.result.json"
done
echo "Wrote ${#VCFS[@]} result file(s) to ${SAMPLES_DIR}. Commit them."
