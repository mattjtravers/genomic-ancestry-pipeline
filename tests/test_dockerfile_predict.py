"""Tests for the Lambda inference image definition — DEPLOY-BE-023.

Parses ``Dockerfile.predict`` as text; does not build the image.
"""
from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile.predict"


def _pip_install_packages() -> list[str]:
    """Package names (lowercased, version pins stripped) from every ``pip install``."""
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    names: list[str] = []
    for line in text.splitlines():
        match = re.search(r"\bpip install\b(.*)", line)
        if not match:
            continue
        for token in match.group(1).split():
            if token.startswith("-"):
                continue
            names.append(re.split(r"[=<>!~\[]", token, maxsplit=1)[0].lower())
    return names


# @spec DEPLOY-BE-023
def test_dockerfile_predict_uses_lambda_python_312_base():
    assert re.search(
        r"^FROM\s+public\.ecr\.aws/lambda/python:3\.12\b", DOCKERFILE.read_text(), re.M
    )


# @spec DEPLOY-BE-023
def test_dockerfile_predict_installs_exactly_the_inference_dependencies():
    assert sorted(_pip_install_packages()) == ["boto3", "numpy", "pydantic", "xgboost-cpu"]
