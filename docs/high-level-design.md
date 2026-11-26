# High-Level Design: Genomic Ancestry Pipeline

## Problem

 Building machine learning pipelines can be a siloed process — business analysts, data scientists, and software engineers each maintain separate artifacts with no enforced traceability between intent and implementation. This project predicts ancestral population from raw SNP (single nucleotide polymorphism) data using the 1000 Genomes Project cohorts as training labels, and addresses both the ML prediction task and the process problem by using [Linked-Intent Development (LID)](https://github.com/jszmajda/lid) to unify intent from requirements through production code in a single "Spec-to-Production" lifecycle.

## Approach

- **Input**: Raw SNP data from the public [1000 Genomes Project](https://www.internationalgenome.org/) dataset
- **Model**: XGBoost (gradient boosted decision trees)
- **Preprocessing**: Median imputation for missing genotype values
- **Validation**: K-fold cross-validation
- **Output**: Ancestral population prediction + confidence score, traceable to specific genomic markers
- **Evaluation**: Confusion matrix and F1-score

## Target Users

Three personas collaborate on this pipeline:

| Persona | Domain | Role |
|---|---|---|
| **Business Analyst** | Intent | Defines the population labels the model predicts, interprets confidence scores, requires marker traceability |
| **Data Scientist** | Rigor | Owns model selection, validation methodology, and evaluation metrics |
| **Software Engineer** | Integrity | Owns infrastructure (AWS S3/Lambda/SageMaker), data contracts (Pydantic), and test coverage (EARS-driven TDD) |

## Goals

- Predict ancestral population from SNP data with per-prediction confidence scores
- Trace every prediction to the contributing genomic markers
- Enforce data schemas and validation at every pipeline stage via Pydantic
- Demonstrate a single "Spec-to-Production" LID lifecycle that replaces compartmentalized ML development

## Non-Goals

- Guaranteed low-latency inference: there is no provisioned capacity, so the first live prediction after an idle period may wait on a cold start
- Support for omics data types beyond SNPs (e.g., RNA-seq, methylation) — deferred
- Clinical diagnostic use (research and demonstration scope only)
- Production-hardened UI (the static site is a demo interface, not a production web app)

## System Design

```
1. Data Ingestion      — fetch SNP data from S3 (1000 Genomes)
2. Preprocessing       — median imputation, Pydantic schema validation
3. Feature Engineering — SNP encoding, marker selection
4. Model Training      — XGBoost with k-fold cross-validation
5. Evaluation          — confusion matrix, F1-score reporting
6. Prediction          — ancestral population label + confidence score + marker contribution trace
7. Deployment          — S3 storage, SageMaker training job, Lambda execution, IAM
8. UI                  — static HTML/JS site on GitHub Pages; Business Analyst submits SNP data and views predictions
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| ML algorithm | Open-source XGBoost | Handles sparse SNP features well; produces per-sample SHAP values for marker traceability; custom SageMaker container gives full control over training logic and artifact format. |
| Missing data | Median imputation | Simple, robust baseline for genotype missingness |
| Validation strategy | K-fold cross-validation | Prevents overfitting on the relatively small 1000 Genomes cohort |
| Schema enforcement | Pydantic | Catches malformed data at every pipeline boundary |
| Workload split | Local data pull/prep → S3 → SageMaker training → S3 → Lambda inference | Deliberate use of each AWS service where it is the industry-standard fit; data acquisition stays local, training and inference are cloud-only |
| Inference IaC | AWS SAM (`template.yaml` + `samconfig.toml`) | Showcases SAM; clean fit for packaging Lambda + API Gateway as a single deployable stack |
| Training launch | SageMaker Python SDK (`sagemaker.estimator.Estimator`) from a local launcher script | Industry-standard programmatic interface for SageMaker training; treats training as the imperative one-shot job it is, rather than forcing it into declarative CloudFormation |
| UI framework | Static HTML/JS, calling the inference API directly from the browser | The UI is thin (pick a VCF, POST it, render a result), so it needs no server-side runtime; a static page loads instantly and costs nothing to host. Streamlit was rejected because it needs a running Python server, which means either an always-on host or a sleeping one with a wake-up wait. |
| UI delivery | GitHub Pages, deployed by GitHub Actions | Permanent public URL served from a CDN with no server to wake, so demo visitors see the page immediately; free; lives in the same repo as the code, so sample files ship with the site. Streamlit Community Cloud was rejected because it puts idle apps to sleep and a visitor must wait for the app to wake up, which is too slow for a demo visitor. EC2 and GitHub Codespaces were rejected as cost and setup burdens for a non-technical audience. |
| Inference cold-start mitigation | Precomputed results for the bundled samples, plus a warm-up request to the API on page load | Lambda cold start (container pull plus model load) is the only remaining wait. Sample selections, the one-click path for visitors without their own VCF, return instantly from precomputed results. Uploaded files go to the live API and may wait on a cold start after an idle period; the page-load warm-up starts that cold start while the visitor chooses a file. A scheduled keep-warm ping and provisioned concurrency were rejected: both pay to keep Lambda warm around the clock for a demo that is idle most of the time. |
| Development process | LID (EARS → Tests → Code) | Collapses BA/DS/Eng silos into one traceable artifact chain |

## Cross-Cutting Code Standards

Three engineering standards apply uniformly across every code-generating segment. They exist to make the codebase reviewable by humans and operable in production. The HLD is their canonical declaration; conformance is enforced at code-review time rather than via a dedicated arrow segment.

| Code Standard | Rule | Rationale |
|---|---|---|
| Function and class documentation | **Google-style docstrings** on every public function and class, declaring `Args`, `Returns`, `Raises` (and `Yields` for generators). Private helpers may use a one-line summary. | The codebase is reviewed by humans (interview, code-review, audit contexts); structured docstrings are the lowest-friction way to make intent legible without reading the implementation. Tooling (Sphinx, mkdocs, IDE hovers) all consume this format. |
| Inline comments | Reserved for **non-obvious "why"**: hidden constraints, workarounds, ordering invariants, subtle correctness conditions. No "what" comments — identifier names already carry that. | Comments rot; identifiers don't (the linter catches stale names). Only commit a comment if removing it would leave a future reader confused. |
| Production observability | All server-side code (data pull, training, the inference Lambda) emits **structured logs to stdout/stderr** suitable for ingestion by Splunk / CloudWatch Logs Insights / equivalent aggregators. Every error path emits a `logger.error` line carrying enough context (operation, identifiers, exception type, request/run id) for a developer to triage from log output alone, without re-running the failing code. | Production code is debugged from logs, not from interactive sessions. If an operator can't determine *which operation failed, on what input, with what cause* from a single log line, the log line is insufficient. This is a non-negotiable, demo or otherwise. The browser client is out of this standard's scope: it has no stdout reaching an aggregator, and every request it makes is logged server-side by the Lambda. Its console logging is defined in `docs/llds/07_ui.md`. |

The two standards together mean: *any function a human reads should explain what it does to a human, and any failure a human investigates should explain itself in the log stream.*

## Success Metrics

- F1-score on held-out test set reported per population class
- 100% of predictions include a confidence score
- 100% of predictions traceable to contributing SNP markers
- All EARS specs have corresponding passing tests

## References

- [1000 Genomes Project](https://www.internationalgenome.org/)
- [Linked-Intent Development](https://github.com/jszmajda/lid)
- [README.md](../README.md) — persona constraints and vision statement
