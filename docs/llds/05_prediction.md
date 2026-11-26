# Prediction

## Context and Design Philosophy

This component loads a trained model artifact and runs inference on new SNP data. It produces three outputs per sample: a predicted phenotype label, a confidence score, and a list of contributing genomic markers. Marker traceability is the key differentiator from a bare XGBoost `predict` call — it is a first-class output, not an afterthought.

The prediction component is the component most directly serving the Business Analyst persona. Every output must be interpretable and auditable.

## Inference Pipeline

1. Accept a pre-loaded artifact bundle dict from the caller — loading from S3 is the caller's responsibility
2. Accept raw SNP input — genotypes are read as 0/1/2 alternate-allele dosages from phased (`0|1`) or unphased (`0/1`) calls alike, since the model uses dosage only, not phase; a leading `chr` on the chromosome name is removed (case-insensitively), because the `FeatureRegistry` uses bare names (`15`, from the GRCh37 1000 Genomes release) while many VCFs write `chr15`; VCF must contain exactly one sample; reject multi-sample VCFs, VCFs with no sample columns, and VCFs sharing no variants with the `FeatureRegistry` by raising `PredictionError`, inference's input-error class (callers treat it as the client's fault)
3. Apply preprocessing using stored `imputation_medians` (no re-fitting):
   - Variants present in the registry but absent from the input VCF are imputed to their training median
   - Variants present in the input VCF but absent from the registry are silently dropped — they cannot influence the prediction because the model has no weight for them
   - A registry variant counts as present only when its row carries a genotype the parser can read; a row whose genotype is missing (`./.`) or unrecognised is imputed exactly like an absent row
   - If no registry variant is present in the input, the feature vector would be built entirely from training medians and the prediction would say nothing about the sample, so `predict()` raises `PredictionError` instead of predicting
   - `predict()` logs, at `INFO`, how many registry variants were present out of the registry total, so the distribution of coverage across real inputs is visible
4. Apply feature selection using stored `feature_registry` (select same variants, same column order)
5. Run the loaded `xgb.Booster` on the feature vector (`booster.predict(xgb.DMatrix(X))`) to get class probabilities
6. Derive predicted label and confidence score from probabilities
7. Compute per-sample SHAP marker contributions via `predict(X, pred_contribs=True)`
8. Assemble and validate `PredictionResult` via Pydantic
9. Return result to caller

## Python API

```python
def predict(
    vcf_bytes: bytes,
    artifact: dict,
    model_artifact_version: str = "",
    top_n_markers: int = 20,
) -> PredictionResult:
```

The caller supplies a pre-loaded artifact bundle (the dict returned by `load_artifact()`). This separates artifact loading from inference, allowing the Lambda to reuse its warm cache across requests without hitting S3 on every call. The `model_artifact_version` string is recorded verbatim in `PredictionResult` for traceability; the Lambda passes the bundle prefix built from its `MODEL_RUN_ID` environment variable, `models/{MODEL_RUN_ID}/` (see § Artifact Version Identity).

## Model Handle

`load_artifact()` deserializes `model.json` into a plain `xgb.Booster` and returns it under the bundle key `booster`. It does not construct the scikit-learn wrapper `xgb.XGBClassifier`.

Training fits an `XGBClassifier` (`docs/llds/04_model-training.md`), but the `model.json` it saves is the underlying booster's own format, objective included, so it loads into a `Booster` unchanged. The model is multi-class, so training's objective is `multi:softprob` and `Booster.predict` on a `DMatrix` returns one probability per class, row-aligned with the label encoder — the same numbers `XGBClassifier.predict_proba` would return. SHAP contributions already come from the `Booster` (`pred_contribs=True`), so inference needs nothing from the wrapper. `predict()` accepts only an `xgb.Booster` in the bundle's `booster` slot.

The probability reading depends on the saved objective, which training sets implicitly (`XGBClassifier`'s multi-class default). Under `multi:softmax`, for instance, the same call would return a class index rather than probabilities. `predict()` therefore checks that the Booster's output for the sample is a row of exactly as many values as the label encoder has classes, and raises `ModelOutputError` otherwise; the Lambda's error contract maps that to `500 INFERENCE_FAILED`. `ModelOutputError` is its own exception class — neither a `ValueError` nor a `PredictionError` — because it signals a model fault, not bad input, and must never be mapped to a 400 however input errors are classified. The check sits here rather than as a constraint on training so that a training change surfaces as a loud inference failure, never as silently wrong confidence scores.

The inference path — `load_artifact()`, `predict()`, and the modules they import — runs without scikit-learn, pandas, or requests installed. `XGBClassifier` raises `ImportError` without scikit-learn, and `xgboost` imports scikit-learn eagerly whenever it is installed, which loads scikit-learn and SciPy at import time. Keeping them out of the inference path lets the Lambda image omit them, which shortens the Lambda cold start (`docs/llds/06_deployment.md § Cold start`). Training keeps the wrapper and these libraries.

Both properties are pinned by tests against a real model rather than mocks, because mocks cannot catch either regression: one test trains a small three-class `XGBClassifier`, saves it, and checks that the `Booster` loaded from the saved file returns the same probabilities as `predict_proba`; another runs `load_artifact()` and `predict()` with scikit-learn, pandas, and requests made unimportable.

## Artifact Version Identity

The `model_artifact_version` field in `PredictionResult` is the **S3 bundle prefix** for the loaded artifact — the key under which all bundle files (`model.json`, `feature_registry.json`, etc.) reside. Format: `models/{run_id}/` (with the trailing slash), e.g. `models/20240115-a3f2c1/`. Not the full S3 URI (no `s3://` prefix); not the `run_id` alone.

## Confidence Score

The confidence score is the **maximum class probability** from the Booster's class-probability output. This is a simple, interpretable proxy for model certainty:

- Score near 1.0 → model strongly favours one class
- Score near `1/n_classes` → model is uncertain

No probability calibration (e.g., Platt scaling) is applied in the initial version — calibration requires a held-out calibration set and adds complexity that is out of scope for the SDD demo. This is noted as a deferred decision.

## Marker Traceability

Marker contributions are derived using XGBoost's built-in per-sample SHAP values via `booster.predict(X, pred_contribs=True)`. This returns a contribution score for each feature for each individual sample — the markers shown to the Business Analyst reflect the specific input, not a global model average.

No `shap` package is required; XGBoost's tree SHAP implementation is used directly.

For each prediction, the top-N features by absolute SHAP contribution are returned, mapped back to genomic coordinates via the `FeatureRegistry`. Default N = 20 (configurable).

```
MarkerContribution
  variant_id: str
  chrom: str
  pos: int
  ref: str
  alt: str
  shap_contribution: float  # per-sample SHAP value (signed; magnitude = importance)
  rank: int                 # 1 = largest absolute contribution
```

## Output Contract

```
PredictionResult
  sample_id: str
  predicted_phenotype: str
  confidence_score: float                  # in [0.0, 1.0]
  class_probabilities: dict[str, float]    # all classes
  top_markers: list[MarkerContribution]
  model_artifact_version: str              # S3 key of artifact bundle used
```

Inference results are returned to the caller only — they are not persisted to S3.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| Confidence score | Max class probability | Calibrated probability, entropy | Max probability is simple and interpretable; calibration is deferred; entropy is less intuitive for non-technical users |
| Marker attribution | XGBoost built-in per-sample SHAP (`pred_contribs=True`) | Global gain importance, `shap` package, permutation importance | Per-sample attributions are more meaningful for the Business Analyst persona; XGBoost's tree SHAP requires no extra dependency and adds negligible compute |
| Inference-time preprocessing | Stored medians + registry from artifact | Re-run preprocessing pipeline | Avoids reprocessing training data at inference time; ensures identical transformations |
| Inference model handle | Plain `xgb.Booster` loaded from `model.json` | `xgb.XGBClassifier` (scikit-learn wrapper), as used in training | The wrapper adds nothing at inference — probabilities and SHAP both come from the booster — but requires scikit-learn, whose import (with SciPy) is the largest part of the Lambda's cold-start import time. The saved model format is the same either way, so training is unaffected. |
| Artifact loading | Caller's responsibility (passed as dict) | `predict()` loads from S3 internally | Lambda caches the artifact on cold start; having `predict()` reload on every call would hit S3 on every inference request, adding latency and cost. Separating loading from inference lets the Lambda pass its warm cache and lets tests pass fixtures directly. |

## Open Questions & Future Decisions

### Resolved
1. ✅ Per-sample SHAP via XGBoost `pred_contribs=True` — no extra dependency, per-sample not global

### Deferred
1. Probability calibration (Platt scaling / isotonic regression) for more accurate confidence scores
2. Minimum variant coverage threshold — a prediction is refused only when no registry variant is present; a higher threshold (e.g. a minimum fraction of registry variants) needs calibration against real inputs, which the coverage log line (`PRED-PROC-021`) is there to collect
3. Reference-build detection — the model's variants are GRCh37 coordinates. A GRCh38 VCF names the same SNPs at different positions, so it matches few or no registry variants even after `chr` normalisation: no matches is refused, but a few coincidental matches would still yield a weak prediction. Detecting the build (from `##reference` / `##contig` headers, or by position checks) and refusing or converting non-GRCh37 input is deferred
4. Batch prediction API vs single-sample prediction — current design processes one sample at a time; batching deferred

## References

- `docs/llds/04_model-training.md` — produces the model artifact bundle
- `docs/llds/03_feature-engineering.md` — defines `FeatureRegistry`
- `docs/llds/06_deployment.md` — S3 paths for artifact and result storage
- `docs/llds/07_ui.md` — primary consumer of `PredictionResult`
