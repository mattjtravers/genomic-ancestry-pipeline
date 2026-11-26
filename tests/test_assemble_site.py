"""Tests for bin/assemble_site.py — the site build shared by CI, 05_ui.sh and tests.

Covers the file-level behaviour of UI-UI-041 (manifest + copied sample files),
UI-UI-042 (config.js) and UI-UI-055 (missing result fails the build). The CI
deploy job that calls the script is tested separately.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "bin" / "assemble_site.py"


def _samples(tmp_path: Path, numbers=(1, 2, 10), with_results=True) -> Path:
    d = tmp_path / "samples"
    d.mkdir()
    src = (REPO / "samples" / "sample_snp_1.vcf").read_text()
    for n in numbers:
        (d / f"sample_snp_{n}.vcf").write_text(src)
        if with_results:
            (d / f"sample_snp_{n}.result.json").write_text(json.dumps({"n": n}))
    (d / "sample_snp_upload.vcf").write_text(src)
    (d / "README.md").write_text("not a sample")
    return d


def _assemble(out: Path, samples: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--samples", str(samples), *extra],
        capture_output=True, text=True, timeout=60,
    )


# @spec UI-UI-041
def test_manifest_lists_numbered_samples_in_numeric_order(tmp_path):
    out = tmp_path / "site"
    proc = _assemble(out, _samples(tmp_path), "--endpoint", "https://api.example")
    assert proc.returncode == 0, proc.stderr

    manifest = json.loads((out / "samples" / "manifest.json").read_text())
    assert manifest == [
        {"label": f"Sample SNP {n}", "vcf": f"sample_snp_{n}.vcf",
         "result": f"sample_snp_{n}.result.json"}
        for n in (1, 2, 10)
    ]


# @spec UI-UI-041
def test_listed_vcf_and_result_files_are_copied_and_nothing_else(tmp_path):
    out = tmp_path / "site"
    assert _assemble(out, _samples(tmp_path)).returncode == 0
    copied = sorted(p.name for p in (out / "samples").iterdir())
    assert copied == sorted(
        ["manifest.json"]
        + [f"sample_snp_{n}.vcf" for n in (1, 2, 10)]
        + [f"sample_snp_{n}.result.json" for n in (1, 2, 10)]
    )


def test_site_source_files_are_copied(tmp_path):
    out = tmp_path / "site"
    assert _assemble(out, _samples(tmp_path)).returncode == 0
    for name in ("index.html", "app.js", "vcf.js", "styles.css"):
        assert (out / name).read_bytes() == (REPO / "site" / name).read_bytes()


# @spec UI-UI-042
def test_config_js_defines_endpoint_global(tmp_path):
    out = tmp_path / "site"
    assert _assemble(out, _samples(tmp_path), "--endpoint", "https://api.example").returncode == 0
    assert (out / "config.js").read_text().strip() == (
        'window.PHENO_API_ENDPOINT = "https://api.example";'
    )


# @spec UI-UI-042
def test_config_js_endpoint_is_empty_when_not_given(tmp_path):
    out = tmp_path / "site"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--samples", str(_samples(tmp_path))],
        capture_output=True, text=True, timeout=60,
        env={k: v for k, v in os.environ.items() if k != "PHENO_API_ENDPOINT"},
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "config.js").read_text().strip() == 'window.PHENO_API_ENDPOINT = "";'


# @spec UI-UI-055
def test_missing_result_file_fails_without_writing_a_site(tmp_path):
    samples = _samples(tmp_path)
    (samples / "sample_snp_2.result.json").unlink()
    out = tmp_path / "site"

    proc = _assemble(out, samples)

    assert proc.returncode != 0
    assert "sample_snp_2" in proc.stderr
    assert not out.exists(), "a failed assembly must not leave a partial site behind"


def test_existing_output_directory_is_replaced(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    assert _assemble(out, _samples(tmp_path)).returncode == 0
    assert not (out / "stale.txt").exists()


def test_repo_samples_assemble_cleanly(tmp_path):
    """The committed samples/ directory itself assembles (every sample has a result)."""
    out = tmp_path / "site"
    proc = _assemble(out, REPO / "samples")
    assert proc.returncode == 0, proc.stderr
    shutil.rmtree(out)
