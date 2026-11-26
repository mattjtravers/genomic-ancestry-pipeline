"""Child-process helper: run inference code with training-only libraries blocked.

Invoked by tests as ``python _inference_without_training_libs.py <mode> [bundle_dir]``.
Blocks ``sklearn``, ``pandas`` and ``requests`` before anything else is imported,
so ``xgboost`` sees them as not installed — as in the Lambda inference image.

Modes:
    import-deployment: import the Lambda handler module with every boto3 entry
        point replaced by a function that fails, then check no model bundle
        was loaded at import (DEPLOY-BE-038, DEPLOY-BE-039).
    predict <bundle_dir>: run ``load_artifact()`` against a local bundle served
        by a fake S3 client, then ``predict()`` on ``<bundle_dir>/sample.vcf``
        (PRED-PROC-015).

Prints a line starting with ``OK`` on success; any failure raises and exits non-zero.
"""
from __future__ import annotations

import importlib.abc
import io
import sys
from pathlib import Path

BLOCKED = {"sklearn", "pandas", "requests"}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked for test: {name}", name=name)
        return None


sys.meta_path.insert(0, _Blocker())


def _import_deployment() -> None:
    import boto3

    def _no_aws(*args, **kwargs):
        raise AssertionError("AWS client created at import time")

    boto3.client = boto3.resource = boto3.Session = _no_aws

    import genomic_ancestry_pipeline.deployment as deployment

    assert deployment._artifact_cache == {}, "model bundle loaded at import time"
    print("OK import-deployment")


def _predict(bundle_dir: Path) -> None:
    from unittest.mock import patch

    import genomic_ancestry_pipeline.prediction as prediction

    class _FakeS3:
        def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg names
            return {"Body": io.BytesIO((bundle_dir / Key.rsplit("/", 1)[1]).read_bytes())}

    with patch.object(prediction.boto3, "client", return_value=_FakeS3()):
        artifact = prediction.load_artifact(bucket="test-bucket", run_id="20240115-a3f2c1")
    result = prediction.predict(
        (bundle_dir / "sample.vcf").read_bytes(),
        artifact=artifact,
        model_artifact_version="models/20240115-a3f2c1/",
    )
    for mod in BLOCKED:
        assert mod not in sys.modules, f"{mod} was imported"
    print("OK predict", result.predicted_phenotype)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "import-deployment":
        _import_deployment()
    elif mode == "predict":
        _predict(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
