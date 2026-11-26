"""Tests for bin/05_ui.sh — the local static-site preview (UI-UI-050).

The real script runs with a stub ``uv`` first on PATH. The stub runs
``uv run python bin/assemble_site.py ...`` for real (so the assembled output
can be inspected) and records, instead of starting, the ``http.server`` call.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "bin" / "05_ui.sh"

_UV_STUB = f"""#!/usr/bin/env bash
echo "uv $*" >> "$CALL_LOG"
[[ "$1" == "run" ]] || exit 2
shift
[[ "$1" == "python" ]] || exit 2
shift
if [[ "$*" == *"http.server"* ]]; then
  # Record what would be served, then return as if the server was stopped.
  dir=""
  args=("$@")
  for ((i=0; i<${{#args[@]}}; i++)); do
    [[ "${{args[i]}}" == "--directory" ]] && dir="${{args[i+1]}}"
  done
  ls "$dir" > "$SERVED_LISTING"
  cp "$dir/config.js" "$SERVED_CONFIG"
  exit 0
fi
exec "{sys.executable}" "$@"
"""


def _run(tmp_path: Path, endpoint: str | None):
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    uv = stub_dir / "uv"
    uv.write_text(_UV_STUB)
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    env = {k: v for k, v in os.environ.items() if k != "PHENO_API_ENDPOINT"}
    env.update({
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "CALL_LOG": str(tmp_path / "calls.log"),
        "SERVED_LISTING": str(tmp_path / "listing.txt"),
        "SERVED_CONFIG": str(tmp_path / "config.js"),
    })
    if endpoint is not None:
        env["PHENO_API_ENDPOINT"] = endpoint
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                          timeout=60, cwd=tmp_path)
    calls = (tmp_path / "calls.log").read_text().splitlines()
    return proc, calls


def _served_dir(calls: list[str]) -> str:
    (serve,) = [c for c in calls if "http.server" in c]
    parts = serve.split()
    return parts[parts.index("--directory") + 1]


# @spec UI-UI-050
def test_assembles_site_with_env_endpoint_and_serves_it_on_8000(tmp_path):
    proc, calls = _run(tmp_path, "https://api.example")
    assert proc.returncode == 0, proc.stderr

    assert any("bin/assemble_site.py" in c and "--endpoint https://api.example" in c
               for c in calls), calls
    (serve,) = [c for c in calls if "http.server" in c]
    assert "python -m http.server 8000" in serve
    assert calls.index(serve) > next(i for i, c in enumerate(calls) if "assemble_site" in c)

    listing = (tmp_path / "listing.txt").read_text().split()
    assert {"index.html", "app.js", "vcf.js", "styles.css", "config.js", "samples"} <= set(listing)
    assert (tmp_path / "config.js").read_text().strip() == (
        f"window.PHENO_API_ENDPOINT = {json.dumps('https://api.example')};"
    )


# @spec UI-UI-050
def test_served_directory_is_removed_when_server_stops(tmp_path):
    _, calls = _run(tmp_path, "https://api.example")
    assert not Path(_served_dir(calls)).exists()


# @spec UI-UI-050
def test_unset_endpoint_warns_and_serves_samples_only(tmp_path):
    proc, calls = _run(tmp_path, None)
    assert proc.returncode == 0, proc.stderr
    assert "PHENO_API_ENDPOINT" in proc.stderr
    assert (tmp_path / "config.js").read_text().strip() == 'window.PHENO_API_ENDPOINT = "";'
    assert any("http.server 8000" in c for c in calls)
