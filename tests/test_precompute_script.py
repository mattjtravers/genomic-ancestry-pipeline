"""Tests for bin/06_precompute_samples.sh — UI-UI-040, UI-UI-058.

The real script runs against a temporary samples directory (``SAMPLES_DIR``)
with a stub ``curl`` first on PATH. The stub records each call, writes a canned
``PredictionResult``-shaped body to curl's ``-o`` path, and prints a status
code (503 for the sample named in ``FAKE_FAIL_SAMPLE``). No network is used.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "bin" / "06_precompute_samples.sh"
REPO_SAMPLES = Path(__file__).resolve().parent.parent / "samples"
ENDPOINT = "https://abc123.execute-api.us-east-1.amazonaws.com"

_CURL_STUB = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
out = args[args.index("-o") + 1]
payload_path = next(a for a in args if a.startswith("@"))[1:]
url = next(a for a in args if a.startswith("http"))
payload = json.load(open(payload_path))
sample = os.path.basename(payload_path).split(".")[0]
with open(os.environ["CALL_LOG"], "a") as log:
    log.write(json.dumps({"url": url, "sample": sample, "vcf": payload["vcf"],
                          "keys": sorted(payload)}) + "\n")
failing = os.environ.get("FAKE_FAIL_SAMPLE") == sample
body = {"error": "MODEL_UNAVAILABLE", "detail": "x"} if failing else {
    "sample_id": sample, "model_artifact_version": "models/20260520-3c1123/"}
open(out, "w").write(json.dumps(body))
sys.stdout.write("503" if failing else "200")
"""


def _samples(tmp_path: Path) -> Path:
    d = tmp_path / "samples"
    d.mkdir()
    for name in ("sample_snp_1.vcf", "sample_snp_2.vcf", "sample_snp_upload.vcf"):
        shutil.copy(REPO_SAMPLES / name, d / name)
    return d


def _run(tmp_path: Path, samples: Path, **env_overrides: str | None):
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(exist_ok=True)
    curl = stub_dir / "curl"
    curl.write_text(_CURL_STUB)
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
    call_log = tmp_path / "calls.jsonl"
    call_log.write_text("")
    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "SAMPLES_DIR": str(samples),
        "PHENO_API_ENDPOINT": ENDPOINT,
    }
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                          timeout=60)
    calls = [json.loads(line) for line in call_log.read_text().splitlines()]
    return proc, calls


# @spec UI-UI-040
def test_posts_each_numbered_sample_and_writes_response_body_unchanged(tmp_path):
    samples = _samples(tmp_path)
    proc, calls = _run(tmp_path, samples)

    assert proc.returncode == 0, proc.stderr
    assert sorted(c["sample"] for c in calls) == ["sample_snp_1", "sample_snp_2"], (
        "only sample_snp_[0-9]*.vcf files are sent (not sample_snp_upload.vcf)"
    )
    for c in calls:
        assert c["url"] == f"{ENDPOINT}/predict"
        assert c["keys"] == ["vcf"], "request body is {'vcf': <contents>} only"
        assert c["vcf"] == (samples / f"{c['sample']}.vcf").read_text()
    for n in (1, 2):
        written = (samples / f"sample_snp_{n}.result.json").read_text()
        assert json.loads(written) == {
            "sample_id": f"sample_snp_{n}",
            "model_artifact_version": "models/20260520-3c1123/",
        }
    assert not (samples / "sample_snp_upload.result.json").exists()


# @spec UI-UI-058
def test_any_non_200_leaves_every_result_file_untouched(tmp_path):
    samples = _samples(tmp_path)
    stale = samples / "sample_snp_1.result.json"
    stale.write_text('{"model_artifact_version": "models/old/"}')

    proc, _ = _run(tmp_path, samples, FAKE_FAIL_SAMPLE="sample_snp_2")

    assert proc.returncode != 0
    assert stale.read_text() == '{"model_artifact_version": "models/old/"}', (
        "a failure must not replace results that came from the previous model"
    )
    assert not (samples / "sample_snp_2.result.json").exists()


# @spec UI-UI-058
def test_missing_endpoint_exits_non_zero_without_requests(tmp_path):
    samples = _samples(tmp_path)
    proc, calls = _run(tmp_path, samples, PHENO_API_ENDPOINT=None)

    assert proc.returncode != 0
    assert calls == []
    assert not list(samples.glob("*.result.json"))
