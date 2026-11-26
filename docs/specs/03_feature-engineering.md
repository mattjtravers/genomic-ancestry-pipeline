# Feature Engineering Specs

## Marker Selection Filters

- [x] **FEAT-PROC-001**: The system shall drop variant columns (not samples) with minor allele frequency below 0.01 (configurable), where MAF is computed on training-split samples only.
- [x] **FEAT-PROC-003**: The system shall drop near-zero-variance variant columns (not samples) after dosage encoding, where variance is computed on training-split samples only.
- [x] **FEAT-PROC-004**: Where the association filter is enabled (configurable pipeline parameter), the system shall retain the top-N variants by univariate chi-squared association score with the population label, computed on training-split samples only. Default N = 10,000.

## Feature Registry

- [x] **FEAT-DATA-001**: The system shall produce a `FeatureRegistry` that maps each column index in the feature matrix to its source variant's ID, chromosome, position, reference allele, alternate allele, and minor allele frequency.
- [x] **FEAT-DATA-002**: The `FeatureRegistry` shall be produced from the training-split marker selection and shall be applied unchanged to the test split, ensuring identical column ordering at inference time.

## Output Contract

- [x] **FEAT-DATA-003**: The system shall output a `FeatureMatrix` containing: a numeric array `X` of shape (n_samples, n_features) with dosage-encoded values, a label array `y` of population class indices, a list of sample IDs, and the `FeatureRegistry`.
- [x] **FEAT-PROC-005**: The system shall preserve the `split` field from the `CleanSnpDataset` (defined in PREP-DATA-001) on each sample in the `FeatureMatrix` output, so downstream components can enforce train/test separation.
