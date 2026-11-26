# Prediction Specs

## Artifact Loading

- [x] **PRED-PROC-001**: The `predict` function shall accept a pre-loaded artifact bundle dict from the caller and shall not load from S3 internally; the caller is responsible for providing a complete bundle (as defined in TRAIN-DATA-001) before invoking inference.
- [x] **PRED-PROC-002**: The system shall record the S3 bundle prefix of the artifact used in every `PredictionResult` under `model_artifact_version`, in the form `models/{run_id}/` (e.g. `models/20240115-a3f2c1/`).

- [x] **PRED-PROC-014**: `load_artifact()` shall deserialize `model.json` into a plain `xgb.Booster` and return it under the bundle key `booster`; it shall not construct an `xgb.XGBClassifier`.
- [x] **PRED-PROC-016**: `predict()` shall accept the bundle's `booster` only as an `xgb.Booster`, and shall not call `predict_proba()` or `get_booster()`.
- [x] **PRED-PROC-017**: If the `xgb.Booster`'s class-probability output for the input sample is not a single row with exactly as many values as the label encoder has classes, then `predict()` shall raise `ModelOutputError`, an exception class that is neither a `ValueError` nor a `PredictionError`, and shall not return a `PredictionResult`.
- [x] **PRED-PROC-015**: `load_artifact()` and `predict()`, including every module they import, shall run to completion when scikit-learn, pandas, and requests cannot be imported.

## Input Validation

- [x] **PRED-PROC-012**: If the input VCF contains more than one sample, then `predict()` shall raise `PredictionError` and shall not produce a prediction.
- [ ] **PRED-PROC-019**: If none of the `FeatureRegistry`'s variants has a genotype in the input VCF that the parser can read, then `predict()` shall raise `PredictionError` and shall not produce a prediction.
- [ ] **PRED-PROC-022**: The VCF parser shall remove a leading `chr` prefix, compared case-insensitively, from the `CHROM` field before forming a variant ID, so that `chr15` and `15` identify the same chromosome.
- [ ] **PRED-PROC-020**: The VCF parser shall read the unphased genotypes `0/0`, `0/1`, `1/0`, and `1/1` as dosages 0, 1, 1, and 2 respectively, the same as their phased forms `0|0`, `0|1`, `1|0`, and `1|1`.
- [x] **PRED-PROC-018**: If the input VCF's `#CHROM` header line has no sample columns, then `predict()` shall raise `PredictionError` and shall not produce a prediction.

## Inference Preprocessing

- [x] **PRED-PROC-003**: At inference time, the system shall impute missing genotype values using the `imputation_medians` from the loaded artifact bundle, without refitting on the input sample.
- [x] **PRED-PROC-004**: At inference time, the system shall select and order features using the `FeatureRegistry` from the loaded artifact bundle, without refitting marker selection. Variants present in the registry but absent from the input VCF shall be imputed to their stored training median.
- [x] **PRED-PROC-013**: At inference time, variants present in the input VCF but absent from the loaded `FeatureRegistry` shall be silently dropped before the feature vector is assembled — the model has no weight for them and they cannot affect the prediction.

## Inference Observability

- [ ] **PRED-PROC-021**: When `predict()` has parsed the input VCF, it shall emit an `INFO` log record containing the number of `FeatureRegistry` variants that have a genotype the parser can read and the total number of `FeatureRegistry` variants.

## Prediction Output

- [x] **PRED-PROC-005**: The system shall produce a predicted phenotype label for each input sample by applying the loaded XGBoost model to the preprocessed feature vector.
- [x] **PRED-PROC-006**: The system shall produce a confidence score for each prediction, defined as the probability assigned to the predicted class by the loaded `xgb.Booster`'s class-probability output (`booster.predict(xgb.DMatrix(X))`), i.e. the maximum class probability, in the range [0.0, 1.0].
- [x] **PRED-DATA-001**: The system shall include the full class probability distribution (all classes, mapped to human-readable labels via `label_encoder.json`) in the `PredictionResult`.

## Marker Traceability

- [x] **PRED-PROC-007**: The system shall compute per-sample SHAP contribution values using XGBoost's built-in `predict(X, pred_contribs=True)`, producing a contribution score per feature per sample.
- [x] **PRED-PROC-008**: The system shall return the top-20 features by absolute SHAP contribution (configurable N) for each prediction, mapped to genomic coordinates via the `FeatureRegistry`.
- [x] **PRED-DATA-002**: Each marker contribution in the `PredictionResult` shall include: variant ID, chromosome, position, reference allele, alternate allele, signed SHAP contribution value, and rank.

## Output Validation

- [x] **PRED-PROC-009**: The system shall validate the `PredictionResult` using its Pydantic schema before returning it to the caller.
- [x] **PRED-PROC-010**: If Pydantic validation of the `PredictionResult` fails, then the system shall raise a structured error without returning a partial result.
