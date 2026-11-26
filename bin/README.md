# bin/

Runnable scripts for the Quick Start pipeline. Run them in order; each script prints the `export VAR=...` line that the next script depends on.

| Script | What it does |
|---|---|
| [01_setup_aws.sh](01_setup_aws.sh) | Creates S3 bucket + SageMaker IAM role via AWS CLI |
| [02_ingest.sh](02_ingest.sh) | One-time ETL from public 1000 Genomes bucket |
| [03_train.sh](03_train.sh) | Launches SageMaker training job, captures `run_id` |
| [04_deploy.sh](04_deploy.sh) | `sam build` + `sam deploy`, extracts API endpoint from CloudFormation, then warms the Lambda with one `GET /health` (exits non-zero unless it returns 200) |
| [05_ui.sh](05_ui.sh) | Assembles the static site (`site/` + `samples/`) with `assemble_site.py` and serves it at http://localhost:8000; uses `PHENO_API_ENDPOINT` for live predictions (samples-only without it) |
| [assemble_site.py](assemble_site.py) | Builds the servable site directory: copies `site/`, writes `config.js` and `samples/manifest.json`; used by `05_ui.sh`, the CI deploy job, and the browser tests |
| [06_precompute_samples.sh](06_precompute_samples.sh) | Posts each `samples/sample_snp_N.vcf` to the deployed `POST /predict` (needs `PHENO_API_ENDPOINT`) and writes `samples/sample_snp_N.result.json`, the precomputed results the static site shows instantly |

**Re-run `06_precompute_samples.sh` after every `04_deploy.sh` that binds a new `ModelRunId`, then commit the updated `samples/*.result.json` files.** Each result records the `model_artifact_version` that produced it, and a test fails if the committed results come from more than one model.

See the [Quick Start](../README.md#quick-start) section in the root README for full usage instructions.
