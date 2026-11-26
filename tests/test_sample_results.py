"""Checks on the committed precomputed sample results — UI-UI-057, UI-UI-040.

These read the real ``samples/`` directory: the result files are committed
artifacts produced by ``bin/06_precompute_samples.sh`` against the deployed API.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from genomic_ancestry_pipeline.models import PredictionResult

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _numbered_vcfs() -> list[Path]:
    return sorted(p for p in SAMPLES.glob("sample_snp_*.vcf")
                  if re.fullmatch(r"sample_snp_[0-9]+\.vcf", p.name))


def _results() -> dict[str, dict]:
    return {p.name: json.loads(p.read_text()) for p in SAMPLES.glob("sample_snp_*.result.json")}


# @spec UI-UI-057
def test_all_result_files_share_one_model_artifact_version():
    results = _results()
    assert results, "no samples/sample_snp_N.result.json files are committed"
    versions = {name: r.get("model_artifact_version") for name, r in results.items()}
    assert len(set(versions.values())) == 1, (
        f"result files come from different models: {versions}"
    )


# @spec UI-UI-040
def test_every_numbered_sample_has_a_valid_prediction_result():
    results = _results()
    for vcf in _numbered_vcfs():
        name = vcf.name.replace(".vcf", ".result.json")
        assert name in results, f"{vcf.name} has no {name}"
        PredictionResult.model_validate(results[name])


# @spec PRED-PROC-002, UI-UI-057
def test_result_files_record_the_bundle_prefix_form():
    """model_artifact_version is ``models/{run_id}/``, not the bare run id."""
    for name, r in _results().items():
        assert re.fullmatch(r"models/[0-9]{8}-[0-9a-f]{6}/", r["model_artifact_version"]), (
            f"{name}: {r['model_artifact_version']!r} is not of the form models/{{run_id}}/"
        )
