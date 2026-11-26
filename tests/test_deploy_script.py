"""Tests for bin/04_deploy.sh's post-deploy warm-up — DEPLOY-BE-044/045/046.

The real script runs with stub ``aws``, ``docker``, ``uv`` and ``curl``
executables placed first on PATH. Each stub appends its argv to a shared call
log, so tests can assert which commands ran and in what order. No AWS, Docker
or network access is used.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "bin" / "04_deploy.sh"
ENDPOINT = "https://abc123.execute-api.us-east-1.amazonaws.com"

_STUBS = {
    # aws: answer the calls the script makes; everything else succeeds silently.
    "aws": f"""#!/usr/bin/env bash
echo "aws $*" >> "$CALL_LOG"
case "$*" in
  *"sts get-caller-identity"*) echo 123456789012 ;;
  *"ecr get-login-password"*) echo fake-password ;;
  *"cloudformation describe-stacks"*) echo "{ENDPOINT}" ;;
esac
exit 0
""",
    "docker": """#!/usr/bin/env bash
echo "docker $*" >> "$CALL_LOG"
cat > /dev/null
exit 0
""",
    # uv run sam build / uv run sam deploy; deploy fails when FAKE_DEPLOY_FAIL=1.
    "uv": """#!/usr/bin/env bash
echo "uv $*" >> "$CALL_LOG"
if [[ "$*" == *"sam deploy"* && "${FAKE_DEPLOY_FAIL:-0}" == "1" ]]; then exit 1; fi
exit 0
""",
    # curl: emulate `-w '%{http_code} %{time_total}'` output with a canned status.
    "curl": """#!/usr/bin/env bash
echo "curl $*" >> "$CALL_LOG"
printf '%s %s' "${FAKE_HEALTH_STATUS:-200}" "1.234"
exit 0
""",
}


def _run(tmp_path: Path, **env_overrides: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    for name, body in _STUBS.items():
        path = stub_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    call_log = tmp_path / "calls.log"
    call_log.touch()

    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "RUN_ID": "20260520-3c1123",
        "PHENO_S3_BUCKET": "test-bucket",
        **env_overrides,
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60
    )
    return proc, call_log.read_text().splitlines()


def _health_calls(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("curl ") and f"{ENDPOINT}/health" in c]


# @spec DEPLOY-BE-044
def test_successful_deploy_sends_one_health_request_after_sam_deploy(tmp_path):
    proc, calls = _run(tmp_path)

    assert proc.returncode == 0, proc.stderr
    health = _health_calls(calls)
    assert len(health) == 1, f"expected exactly one GET /health; calls were:\n{calls}"
    deploy_idx = next(i for i, c in enumerate(calls) if "sam deploy" in c)
    assert calls.index(health[0]) > deploy_idx, "GET /health must come after sam deploy"


# @spec DEPLOY-BE-044
def test_health_request_uses_35_second_client_timeout(tmp_path):
    _, calls = _run(tmp_path)
    (health,) = _health_calls(calls)
    assert "--max-time 35" in health, health


# @spec DEPLOY-BE-044
def test_health_status_and_elapsed_time_are_printed(tmp_path):
    proc, _ = _run(tmp_path)
    assert "200" in proc.stdout and "1.234" in proc.stdout, proc.stdout


# @spec DEPLOY-BE-045
@pytest.mark.parametrize("status", ["503", "504", "000"])
def test_non_200_health_exits_non_zero(tmp_path, status):
    proc, calls = _run(tmp_path, FAKE_HEALTH_STATUS=status)
    assert _health_calls(calls), "GET /health must still be attempted"
    assert proc.returncode != 0, f"status {status} must fail the script"


# @spec DEPLOY-BE-046
def test_failed_sam_deploy_exits_non_zero_without_health_request(tmp_path):
    proc, calls = _run(tmp_path, FAKE_DEPLOY_FAIL="1")
    assert proc.returncode != 0
    assert not _health_calls(calls), f"GET /health must not run after a failed deploy:\n{calls}"
