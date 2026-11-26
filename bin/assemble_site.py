#!/usr/bin/env python3
"""Assemble the static site into a servable directory.

Copies ``site/`` (the committed HTML/JS/CSS), writes ``config.js`` defining
``window.PHENO_API_ENDPOINT``, and copies each ``samples/sample_snp_N.vcf`` with
its ``sample_snp_N.result.json`` into ``<out>/samples/``, listed in
``<out>/samples/manifest.json`` in ascending N order.

Fails (exit 1) without touching ``--out`` if any numbered sample has no result
file: a missing precomputed result is a deploy mistake, caught here.

Used by the CI deploy job, ``bin/05_ui.sh``, and the browser tests, so the
published, previewed, and tested sites are built identically. Standard library
only. See ``docs/llds/07_ui.md § Site Structure``.

Usage:
    python bin/assemble_site.py --out DIR [--samples DIR] [--endpoint URL]

``--endpoint`` defaults to the ``PHENO_API_ENDPOINT`` environment variable, or
an empty string (the page then runs samples-only and reports itself
unconfigured).
"""
# @spec UI-UI-041, UI-UI-042, UI-UI-055
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE_SRC = REPO / "site"
_SAMPLE_RE = re.compile(r"sample_snp_([0-9]+)\.vcf")
_BUILD_OUTPUTS = {"config.js", "samples"}


def _numbered_samples(samples_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in samples_dir.glob("sample_snp_*.vcf"):
        match = _SAMPLE_RE.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def assemble(out: Path, samples_dir: Path, endpoint: str) -> None:
    """Build the site in a temp directory, then move it into place at ``out``.

    Args:
        out: Destination directory; replaced wholesale on success.
        samples_dir: Directory holding ``sample_snp_N.vcf`` and results.
        endpoint: API base URL written into ``config.js`` (may be empty).

    Raises:
        SystemExit: If a numbered sample has no ``.result.json`` beside it.
    """
    samples = _numbered_samples(samples_dir)
    missing = [vcf.name for _, vcf in samples
               if not vcf.with_name(vcf.stem + ".result.json").exists()]
    if missing:
        sys.stderr.write(
            "ERROR: no precomputed result for " + ", ".join(missing)
            + "; run bin/06_precompute_samples.sh and commit the results.\n"
        )
        raise SystemExit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=".site-", dir=out.parent))
    try:
        for item in SITE_SRC.iterdir():
            if item.name in _BUILD_OUTPUTS:
                continue
            dest = build / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        (build / "config.js").write_text(f"window.PHENO_API_ENDPOINT = {json.dumps(endpoint)};\n")

        site_samples = build / "samples"
        site_samples.mkdir()
        manifest = []
        for n, vcf in samples:
            result = vcf.with_name(vcf.stem + ".result.json")
            shutil.copy2(vcf, site_samples / vcf.name)
            shutil.copy2(result, site_samples / result.name)
            manifest.append({"label": f"Sample SNP {n}", "vcf": vcf.name, "result": result.name})
        (site_samples / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        if out.exists():
            shutil.rmtree(out)
        build.rename(out)
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and assemble the site.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=Path, default=REPO / "samples")
    parser.add_argument("--endpoint", default=os.environ.get("PHENO_API_ENDPOINT", ""))
    args = parser.parse_args(argv)
    assemble(args.out.resolve(), args.samples.resolve(), args.endpoint)
    print(f"Assembled site in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
