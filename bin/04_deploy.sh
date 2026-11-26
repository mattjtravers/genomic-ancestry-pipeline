#!/usr/bin/env bash
# Builds the Lambda container image and deploys the SAM stack.
# sam deploy will prompt for confirmation before creating resources.
# On success, warms the Lambda with one GET /health (exiting non-zero unless it
# returns 200), then prints an export statement for PHENO_API_ENDPOINT.
# @spec DEPLOY-BE-044, DEPLOY-BE-045, DEPLOY-BE-046
set -euo pipefail

: "${RUN_ID:?Set RUN_ID before running this script (see 03_train.sh output)}"
: "${PHENO_S3_BUCKET:?Set PHENO_S3_BUCKET before running this script}"

REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --no-cli-pager)
ECR_REPO="phenotype-inference"
IMAGE_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

echo "Ensuring ECR repository: ${ECR_REPO}"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" \
    --no-cli-pager > /dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" \
        --no-cli-pager > /dev/null

echo "Authenticating Docker to ECR"
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin \
        "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

uv run sam build

uv run sam deploy \
    --image-repository "$IMAGE_REPO" \
    --no-confirm-changeset \
    --parameter-overrides \
        ModelRunId="$RUN_ID" \
        PhenoS3Bucket="$PHENO_S3_BUCKET"

ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name phenotype-inference \
    --region us-east-1 \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
    --output text \
    --no-cli-pager)

# The first cold start after a deploy is the slowest a visitor could hit
# (docs/llds/06_deployment.md § Cold start). Pay it here, and treat a non-200
# as a broken deploy. 35 s outlasts API Gateway's 30 s integration timeout.
echo ""
echo "Warming the Lambda: GET ${ENDPOINT}/health"
HEALTH=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time 35 "${ENDPOINT}/health" || true)
HEALTH_STATUS=${HEALTH%% *}
HEALTH_TIME=${HEALTH#* }
echo "  /health -> HTTP ${HEALTH_STATUS:-000} in ${HEALTH_TIME:-?}s"
if [[ "${HEALTH_STATUS}" != "200" ]]; then
    echo "ERROR: post-deploy GET /health did not return 200; check the function's CloudWatch logs." >&2
    exit 1
fi

echo ""
echo "Deployment complete. Run the following before 05_ui.sh:"
echo "  export PHENO_API_ENDPOINT=${ENDPOINT}"
