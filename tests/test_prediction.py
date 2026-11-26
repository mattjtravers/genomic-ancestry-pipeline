"""Tests for the prediction component — PRED-* specs."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xgboost as xgb

from genomic_ancestry_pipeline import prediction as prediction_module
from genomic_ancestry_pipeline.models import PredictionResult
from genomic_ancestry_pipeline.prediction import PredictionError, load_artifact, predict

CHILD_SCRIPT = Path(__file__).parent / "_inference_without_training_libs.py"

# ── Shared artifact fixture ────────────────────────────────────────────────────


def _mock_artifact(feature_registry):
    """Returns a minimal in-memory artifact bundle dict for testing."""
    return {
        "booster": MagicMock(),
        "feature_registry": feature_registry,
        "imputation_medians": {e.variant_id: 1.0 for e in feature_registry.features},
        "label_encoder": {0: "blue", 1: "brown", 2: "green"},
    }


def _mock_booster_proba(proba: list[float]):
    """Returns an ``xgb.Booster``-shaped mock producing the given class probabilities.

    ``spec=xgb.Booster`` makes the mock pass ``isinstance`` checks and refuse
    attributes a real Booster lacks (``predict_proba``, ``get_booster``).
    ``predict(DMatrix)`` returns the probability row; ``predict(DMatrix,
    pred_contribs=True)`` returns per-feature SHAP contributions plus a bias column.
    """
    n_features = 5
    contribs = np.zeros((1, n_features + 1))
    contribs[0, :n_features] = [0.34, 0.21, 0.10, 0.05, 0.03]

    def _predict(data, *args, pred_contribs=False, **kwargs):
        return contribs if pred_contribs else np.array([proba])

    booster = MagicMock(spec=xgb.Booster)
    booster.predict.side_effect = _predict
    return booster


def _probability_calls(booster):
    """The booster.predict calls that requested class probabilities (not SHAP)."""
    return [c for c in booster.predict.call_args_list if not c.kwargs.get("pred_contribs")]


# ── Artifact loading ───────────────────────────────────────────────────────────


# @spec PRED-PROC-001
def test_predict_uses_caller_supplied_artifact(minimal_vcf_bytes, feature_registry):
    """predict() uses the artifact dict supplied by the caller; it does not load from S3."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("genomic_ancestry_pipeline.prediction.load_artifact") as mock_load:
        result = predict(
            minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/"
        )
        mock_load.assert_not_called()

    assert isinstance(result, PredictionResult)


# @spec PRED-PROC-002
def test_prediction_result_records_artifact_version(minimal_vcf_bytes, feature_registry):
    """PredictionResult.model_artifact_version contains the version string passed by the caller."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    result = predict(
        minimal_vcf_bytes,
        artifact=artifact,
        model_artifact_version="models/20240115-a3f2c1/",
    )

    assert result.model_artifact_version == "models/20240115-a3f2c1/"


# ── Input validation ───────────────────────────────────────────────────────────


# @spec PRED-PROC-012
def test_multi_sample_vcf_raises_prediction_error(multi_sample_vcf_bytes, feature_registry):
    """VCF with more than one sample raises PredictionError; no prediction is produced."""
    artifact = _mock_artifact(feature_registry)

    with pytest.raises(PredictionError):
        predict(multi_sample_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")


# ── Inference preprocessing ────────────────────────────────────────────────────


# @spec PRED-PROC-003
def test_inference_uses_stored_medians_without_refitting(minimal_vcf_bytes, feature_registry):
    """Inference imputes using artifact medians; no new median is computed from the input sample."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("genomic_ancestry_pipeline.prediction.np") as mock_np:
        mock_np.median = MagicMock(
            side_effect=AssertionError("median must not be recomputed at inference")
        )
        mock_np.nanmedian = MagicMock(
            side_effect=AssertionError("nanmedian must not be recomputed at inference")
        )
        mock_np.array = np.array
        mock_np.zeros = np.zeros

        try:
            predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")
        except AssertionError:
            pytest.fail("predict() recomputed imputation medians from the input sample")
        except Exception:
            pass


# @spec PRED-PROC-013
def test_input_variants_absent_from_registry_are_silently_dropped(feature_registry):
    """Variants in the input VCF but absent from the FeatureRegistry are dropped without error."""
    vcf = (
        b"##fileformat=VCFv4.1\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\n"
        b"15\t28513871\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
        b"15\t99999999\t.\tA\tG\t.\tPASS\t.\tGT\t1|1\n"
    )
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    result = predict(vcf, artifact=artifact, model_artifact_version="models/run1/")

    dmatrix = _probability_calls(booster)[0].args[0]
    assert dmatrix.num_col() == len(feature_registry.features)
    assert isinstance(result, PredictionResult)


# @spec PRED-PROC-004
def test_inference_applies_stored_feature_registry(minimal_vcf_bytes, feature_registry):
    """Inference selects and orders features using the artifact FeatureRegistry."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.7, 0.2, 0.1])
    artifact["booster"] = booster

    predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    dmatrix = _probability_calls(booster)[0].args[0]
    assert dmatrix.num_col() == len(feature_registry.features)


# ── Prediction output ──────────────────────────────────────────────────────────


# @spec PRED-PROC-005
def test_predict_returns_population_label(minimal_vcf_bytes, feature_registry):
    """PredictionResult contains a predicted_phenotype string (ancestral population label)."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert isinstance(result.predicted_phenotype, str)
    assert result.predicted_phenotype in {"blue", "brown", "green"}


# @spec PRED-PROC-006
def test_confidence_score_is_max_class_probability(minimal_vcf_bytes, feature_registry):
    """confidence_score equals the maximum class probability from booster.predict(DMatrix)."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.75, 0.15, 0.10])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert abs(result.confidence_score - 0.75) < 1e-6
    assert 0.0 <= result.confidence_score <= 1.0


# @spec PRED-DATA-001
def test_prediction_result_includes_full_class_probabilities(minimal_vcf_bytes, feature_registry):
    """class_probabilities contains all classes mapped to human-readable labels."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.75, 0.15, 0.10])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert set(result.class_probabilities.keys()) == {"blue", "brown", "green"}
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6


# ── Marker traceability ────────────────────────────────────────────────────────


# @spec PRED-PROC-007
def test_shap_computed_via_pred_contribs(minimal_vcf_bytes, feature_registry):
    """Marker contributions use XGBoost pred_contribs=True, not the shap package."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    contribs_calls = [
        c for c in booster.predict.call_args_list if c.kwargs.get("pred_contribs") is True
    ]
    assert len(contribs_calls) >= 1, "booster.predict(pred_contribs=True) was not called"


# @spec PRED-PROC-008
def test_top_n_markers_returned_by_absolute_shap(minimal_vcf_bytes, feature_registry):
    """top_markers contains the top-N features by absolute SHAP value."""
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    artifact["booster"] = booster

    result = predict(
        minimal_vcf_bytes,
        artifact=artifact,
        model_artifact_version="models/run1/",
        top_n_markers=3,
    )

    assert len(result.top_markers) <= 3
    abs_contribs = [abs(m.shap_contribution) for m in result.top_markers]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


# @spec PRED-DATA-002
def test_marker_contribution_fields_complete(minimal_vcf_bytes, feature_registry):
    """Each MarkerContribution has variant_id, chrom, pos, ref, alt, shap_contribution, rank."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert len(result.top_markers) > 0
    for marker in result.top_markers:
        assert marker.variant_id
        assert marker.chrom
        assert isinstance(marker.pos, int)
        assert marker.ref
        assert marker.alt
        assert isinstance(marker.shap_contribution, float)
        assert isinstance(marker.rank, int)
        assert marker.rank >= 1


# ── Output validation ──────────────────────────────────────────────────────────


# @spec PRED-PROC-009
def test_prediction_result_is_pydantic_validated(minimal_vcf_bytes, feature_registry):
    """PredictionResult is an instance of the Pydantic model (validated before return)."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    result = predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")

    assert isinstance(result, PredictionResult)


# @spec PRED-PROC-010
def test_pydantic_validation_failure_raises_not_partial_result(minimal_vcf_bytes, feature_registry):
    """If PredictionResult validation fails, an error is raised; no partial result returned."""
    artifact = _mock_artifact(feature_registry)
    artifact["booster"] = _mock_booster_proba([0.8, 0.1, 0.1])

    with patch("genomic_ancestry_pipeline.prediction.PredictionResult") as mock_model:
        mock_model.side_effect = ValueError("validation failure")

        with pytest.raises((PredictionError, ValueError)):
            predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")


# ── Plain Booster model handle — PRED-PROC-006 / 014 / 015 / 016 / 017 ─────────
#
# These tests use a real trained model, not mocks: a mock cannot show that the
# Booster loaded from model.json returns the same probabilities as the
# XGBClassifier that saved it, nor that inference runs without scikit-learn.

LABELS = {0: "blue", 1: "brown", 2: "green"}


def _sample_vcf(feature_registry) -> bytes:
    genotypes = ["0|1", "1|1", "0|0", "0|1", "1|0"]
    rows = [
        f"{e.chrom}\t{e.pos}\t.\t{e.ref}\t{e.alt}\t.\tPASS\t.\tGT\t{gt}\n"
        for e, gt in zip(feature_registry.features, genotypes, strict=True)
    ]
    header = (
        "##fileformat=VCFv4.1\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\n"
    )
    return (header + "".join(rows)).encode()


@pytest.fixture
def trained_bundle(tmp_path, feature_registry):
    """Train a tiny 3-class XGBClassifier and write a full artifact bundle to disk."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(30, len(feature_registry.features))).astype(float)
    y = np.array([0, 1, 2] * 10)
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
    clf.fit(X, y)

    clf.save_model(str(tmp_path / "model.json"))
    (tmp_path / "feature_registry.json").write_text(feature_registry.model_dump_json())
    (tmp_path / "imputation_medians.json").write_text(
        json.dumps({e.variant_id: 1.0 for e in feature_registry.features})
    )
    (tmp_path / "label_encoder.json").write_text(
        json.dumps({str(k): v for k, v in LABELS.items()})
    )
    (tmp_path / "evaluation_report.json").write_text("{}")
    (tmp_path / "sample.vcf").write_bytes(_sample_vcf(feature_registry))
    return tmp_path, clf


class _FakeS3:
    """Serves ``models/{run_id}/<file>`` keys from a local bundle directory."""

    def __init__(self, bundle_dir: Path):
        self._dir = bundle_dir

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803 — boto3 kwarg names
        return {"Body": io.BytesIO((self._dir / Key.rsplit("/", 1)[1]).read_bytes())}


def _load(bundle_dir: Path) -> dict:
    with patch.object(prediction_module.boto3, "client", return_value=_FakeS3(bundle_dir)):
        return load_artifact(bucket="test-bucket", run_id="20240115-a3f2c1")


# @spec PRED-PROC-014
def test_load_artifact_returns_plain_booster(trained_bundle):
    bundle_dir, _ = trained_bundle
    artifact = _load(bundle_dir)
    assert type(artifact["booster"]) is xgb.Booster, (
        f"booster must be a plain xgb.Booster, got {type(artifact['booster']).__name__}"
    )


# @spec PRED-PROC-006, PRED-PROC-014
def test_booster_probabilities_match_xgbclassifier_predict_proba(trained_bundle, feature_registry):
    bundle_dir, clf = trained_bundle
    artifact = _load(bundle_dir)
    vcf = (bundle_dir / "sample.vcf").read_bytes()

    result = predict(vcf, artifact=artifact, model_artifact_version="models/20240115-a3f2c1/")

    _, genotypes = prediction_module._parse_single_sample_vcf(vcf)
    X = prediction_module._build_feature_vector(
        genotypes, feature_registry, artifact["imputation_medians"]
    )
    expected = clf.predict_proba(X)[0]
    got = [result.class_probabilities[LABELS[i]] for i in range(3)]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
    assert result.confidence_score == pytest.approx(float(expected.max()), abs=1e-6)
    assert result.predicted_phenotype == LABELS[int(expected.argmax())]


# @spec PRED-PROC-016
def test_predict_rejects_xgbclassifier_in_booster_slot(trained_bundle, feature_registry):
    bundle_dir, clf = trained_bundle
    artifact = _mock_artifact(feature_registry)
    wrapper = MagicMock(spec=xgb.XGBClassifier)
    artifact["booster"] = wrapper

    with pytest.raises(TypeError):
        predict(
            (bundle_dir / "sample.vcf").read_bytes(),
            artifact=artifact,
            model_artifact_version="models/run1/",
        )
    wrapper.predict_proba.assert_not_called()
    wrapper.get_booster.assert_not_called()


# @spec PRED-PROC-017
@pytest.mark.parametrize(
    "bad_output",
    [
        np.array([1.0]),  # multi:softmax-style class index
        np.array([[0.5, 0.5]]),  # two values for three classes
        np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),  # two rows for one sample
    ],
    ids=["class-index", "too-few-classes", "two-rows"],
)
def test_predict_raises_when_booster_output_is_not_one_probability_row(
    minimal_vcf_bytes, feature_registry, bad_output
):
    artifact = _mock_artifact(feature_registry)
    booster = _mock_booster_proba([0.8, 0.1, 0.1])
    contribs_fn = booster.predict.side_effect
    booster.predict.side_effect = lambda data, *a, pred_contribs=False, **k: (
        contribs_fn(data, pred_contribs=True) if pred_contribs else bad_output
    )
    artifact["booster"] = booster

    with pytest.raises(PredictionError):
        predict(minimal_vcf_bytes, artifact=artifact, model_artifact_version="models/run1/")


# @spec PRED-PROC-015
def test_inference_runs_without_sklearn_pandas_requests(trained_bundle):
    """load_artifact() + predict() in a fresh interpreter where those imports are blocked.

    A fresh process is required: xgboost probes for scikit-learn once, when it is
    first imported, and this test process has already imported it.
    """
    bundle_dir, _ = trained_bundle
    proc = subprocess.run(
        [sys.executable, str(CHILD_SCRIPT), "predict", str(bundle_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.startswith("OK"), proc.stdout
