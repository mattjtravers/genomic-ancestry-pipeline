# Deployment Specs

## S3 Bucket Structure

- [x] **DEPLOY-BE-001**: All pipeline data shall be stored under a single S3 bucket configured via the `PHENO_S3_BUCKET` environment variable.
- [x] **DEPLOY-BE-002**: The system shall assign each training run a unique `run_id` composed of a UTC timestamp and a short UUID (e.g., `20240115-a3f2c1`), used as the S3 key prefix for model artifacts and prediction results.
- [x] **DEPLOY-BE-003**: The S3 bucket shall follow the structure: `data/raw/` for source VCF and metadata, `models/{run_id}/` for model artifact bundles, and `logs/` for training and inference logs.

## Container Images

- [x] **DEPLOY-BE-023**: The Lambda inference container image shall be built from `Dockerfile.predict` using `public.ecr.aws/lambda/python:3.12` as the base; the `pip install` in `Dockerfile.predict` shall name exactly `xgboost-cpu`, NumPy, Pydantic, and boto3 (their transitive dependencies are permitted), and scikit-learn, pandas, requests, and sagemaker shall not be installed in the image; the `genomic_ancestry_pipeline` package shall be copied to `${LAMBDA_TASK_ROOT}`; the Lambda handler entry point shall be `genomic_ancestry_pipeline.deployment.lambda_handler`.
- [ ] **DEPLOY-BE-024**: The SageMaker training container shall provide an executable `train` script on `PATH` (at `/opt/ml/code/train`) that invokes `python -m genomic_ancestry_pipeline.sagemaker_train`; SageMaker BYOC calls `docker run <image> train`, overriding `CMD`, so the named executable is required. The entry point module shall read `PHENO_S3_BUCKET`, `MODEL_RUN_ID`, `K_FOLDS`, `MAF_THRESHOLD`, `TOP_N`, and `RANDOM_STATE` from environment variables and orchestrate the full training pipeline in sequence: `load_raw_dataset` → `preprocess` → `build_feature_matrix` → `train` → `save_artifact`.
- [ ] **DEPLOY-BE-025**: Before each SageMaker training job submission, `bin/03_train.sh` shall build `Dockerfile.train`, push the resulting image to the ECR repository `phenotype-pipeline-training` in `us-east-1`, and export the image URI as `PHENO_TRAINING_IMAGE_URI`; the training launcher shall read `PHENO_TRAINING_IMAGE_URI` as the container image URI passed to `sagemaker.estimator.Estimator`.
- [ ] **DEPLOY-BE-026**: The SageMaker Training Role shall have ECR pull permissions (`ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchCheckLayerAvailability`) on the `phenotype-pipeline-training` repository; these shall be satisfied by the `AmazonSageMakerFullAccess` managed policy attached in `bin/01_setup_aws.sh`.

- [x] **DEPLOY-BE-038**: Importing `genomic_ancestry_pipeline.deployment` shall succeed when scikit-learn, pandas, and requests cannot be imported.
- [x] **DEPLOY-BE-043**: Importing `genomic_ancestry_pipeline.deployment` shall not add any handler to the root logger and shall not call `configure_logging()`.
- [x] **DEPLOY-BE-039**: Importing `genomic_ancestry_pipeline.deployment` shall perform no S3 read and shall not load the model artifact bundle; the first bundle load shall happen inside a request, per `DEPLOY-BE-021`.

## SageMaker Training Job

- [x] **DEPLOY-BE-004**: The SageMaker Training Job shall use the project's custom training Docker container (built from `Dockerfile.train`) as its execution environment.
- [x] **DEPLOY-BE-005**: The SageMaker Training Job shall receive pipeline parameters (k-fold k, MAF threshold, marker count N, random seed) as environment variables.
- [x] **DEPLOY-BE-006**: The SageMaker Training Job shall write the model artifact bundle to the S3 output path corresponding to the run's `run_id`.

## Lambda Inference

- [x] **DEPLOY-BE-007**: The inference Lambda function shall run as a container image (`PackageType: Image`) built from `Dockerfile.predict` on the AWS Lambda Python 3.12 base, containing only the libraries inference imports, with XGBoost supplied by the CPU-only `xgboost-cpu` package (exact contents per `DEPLOY-BE-023`).
- [x] **DEPLOY-BE-008**: The Lambda function shall be allocated 3008 MB memory and a 29-second timeout.
- [x] **DEPLOY-BE-009**: The Lambda function shall cache the model artifact bundle in `/tmp` after the first load, and reuse the cached bundle on subsequent warm invocations within the same container lifecycle.
- [x] **DEPLOY-BE-010**: The Lambda function shall accept invocation via an API Gateway HTTP API endpoint.
- [x] **DEPLOY-BE-027**: The `POST /predict` Lambda route shall accept a JSON request body of shape `{"vcf": "<vcf_string>"}` where `vcf` is the raw VCF content as a UTF-8 string; no `phenotype` field is accepted or required — the model determines the ancestral population from the VCF data alone.
- [x] **DEPLOY-BE-047**: When the Lambda handles a `POST /predict` request, it shall pass `model_artifact_version` to `predict()` as `models/{MODEL_RUN_ID}/`, built from the `MODEL_RUN_ID` environment variable.
- [x] **DEPLOY-BE-028**: The HTTP API shall expose a `GET /health` route integrated with the same Lambda function as `POST /predict`.
- [x] **DEPLOY-BE-029**: When the Lambda receives a `GET /health` request, it shall load the model artifact bundle through the same cached load path used by `POST /predict` (loading it into `/tmp` if not already cached) and, if the load succeeds, return HTTP 200 with body `{"status": "ok"}`.

## Browser Access (CORS)

- [x] **DEPLOY-BE-030**: The HTTP API's CORS configuration shall allow exactly two origins: `https://mattjtravers.github.io` and `http://localhost:8000`.
- [x] **DEPLOY-BE-031**: The HTTP API's CORS configuration shall allow exactly the methods `POST` and `GET`.
- [x] **DEPLOY-BE-032**: The HTTP API's CORS configuration shall allow exactly the request header `Content-Type`.
- [x] **DEPLOY-BE-033**: The HTTP API's CORS configuration shall not allow credentials.
- [x] **DEPLOY-BE-034**: The HTTP API's CORS configuration shall set a preflight `MaxAge` of 600 seconds.
- [x] **DEPLOY-BE-035**: The Lambda handler shall not set any `Access-Control-*` response header; CORS response headers and `OPTIONS` preflight responses shall come from the HTTP API's CORS configuration only.

## Throttling

- [x] **DEPLOY-BE-036**: The HTTP API's default stage shall set `DefaultRouteSettings` with `ThrottlingRateLimit: 2` and `ThrottlingBurstLimit: 10`, applying to every route.

## IAM

- [ ] **DEPLOY-BE-011**: The SageMaker Training Role shall have `s3:GetObject` on `s3://1000genomes/*` and `s3:GetObject` / `s3:PutObject` on `s3://{PHENO_S3_BUCKET}/*`.
- [ ] **DEPLOY-BE-012**: The Lambda Execution Role shall have `s3:GetObject` on `s3://{PHENO_S3_BUCKET}/models/*`.

## Infrastructure-as-Code (SAM)

- [x] **DEPLOY-BE-014**: The inference Lambda, its HTTP API routes, its execution role, and its CloudWatch log group shall be declared in a SAM template at the repo root and deployable end-to-end via `sam build && sam deploy`.
- [x] **DEPLOY-BE-015**: The SAM template shall expose `MODEL_RUN_ID` and `PHENO_S3_BUCKET` as CloudFormation stack parameters, surfaced to the Lambda as environment variables of the same name.
- [x] **DEPLOY-BE-044**: When `sam deploy` in `bin/04_deploy.sh` succeeds, the script shall send one `GET {ApiEndpoint}/health` request with a 35-second client timeout and print the HTTP status and elapsed time.
- [x] **DEPLOY-BE-045**: If the post-deploy `GET /health` in `bin/04_deploy.sh` does not return HTTP 200, then the script shall exit non-zero.
- [x] **DEPLOY-BE-046**: If `sam deploy` in `bin/04_deploy.sh` fails, then the script shall exit non-zero without sending the post-deploy `GET /health` request.
- [x] **DEPLOY-BE-020**: The SAM template shall declare an `AWS::Logs::LogGroup` for the inference Lambda with `RetentionInDays: 14`.
- [x] **DEPLOY-BE-042**: The SAM template shall set the inference Lambda's `LoggingConfig` to `LogFormat: JSON`, `ApplicationLogLevel: INFO`, and `SystemLogLevel: INFO`, with no `LogGroup` override, so records go to the log group declared per `DEPLOY-BE-020`.

## Training Launcher

- [x] **DEPLOY-BE-016**: SageMaker training jobs shall be launched via `pipeline/launch_training.py` using `sagemaker.estimator.Estimator`; the script shall accept k-fold k, MAF threshold, marker count N, random seed, and SageMaker instance type as CLI flags (instance type defaulting to `ml.m5.2xlarge`); the script shall block until the training job reports terminal status and shall emit the resulting `run_id` to stdout only on successful completion.
- [x] **DEPLOY-BE-017**: The training launcher shall exit non-zero before calling `Estimator.fit` if AWS credentials are missing or the configured S3 bucket is unreachable.
- [x] **DEPLOY-BE-018**: The training launcher shall check `s3://{PHENO_S3_BUCKET}/models/{run_id}/` for any existing object before submission and shall exit non-zero with a clear error if the prefix is occupied.

## Region

- [x] **DEPLOY-BE-019**: All AWS resources (SAM stack, SageMaker training jobs, S3 bucket reads/writes, ECR repository) shall be configured for the `us-east-1` region; the launcher and `samconfig.toml` shall pin the region explicitly rather than rely on `AWS_DEFAULT_REGION`.

## Inference Error Contract

- [x] **DEPLOY-BE-021**: On its first model load (cold start, on the first `POST /predict` or `GET /health` request), the Lambda shall verify that every file in the artifact bundle defined by `TRAIN-DATA-001` is present and readable at `s3://{PHENO_S3_BUCKET}/models/{MODEL_RUN_ID}/`; if any file is missing or unreadable, every request in that container lifecycle shall short-circuit to a `503 MODEL_UNAVAILABLE` response without re-attempting the load.
- [x] **DEPLOY-BE-040**: When `predict()` raises `PredictionError` during a `POST /predict` request, the Lambda shall return `400 INVALID_VCF`.
- [x] **DEPLOY-BE-041**: When `predict()` raises any exception other than `PredictionError` during a `POST /predict` request, including a plain `ValueError`, the Lambda shall return `500 INFERENCE_FAILED`.
- [x] **DEPLOY-BE-037**: If an invocation reaches the Lambda timeout while loading the model artifact bundle, the Lambda shall not record the load as failed, and the next request in that container shall re-attempt the load.
- [x] **DEPLOY-BE-022**: Every Lambda error response shall carry the HTTP status defined in the LLD's `§ Error contract` table (`503 MODEL_UNAVAILABLE`, `400 INVALID_INPUT`, `400 INVALID_VCF`, `500 INFERENCE_FAILED`, `500 INTERNAL_ERROR`), a JSON body of shape `{"error": <code>, "detail": <message>}`, and a `logger.error` line carrying the same code, the detail, and the API Gateway request id (`requestContext.requestId` from the event).
