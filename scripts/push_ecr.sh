#!/usr/bin/env bash
set -euo pipefail

AWS_PROFILE="esflow"
AWS_REGION="us-east-2"
ECR_REGISTRY="793452511035.dkr.ecr.us-east-2.amazonaws.com"
ECR_REPO="esflow"
IMAGE_TAG="latest"
IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPO}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# If the SSO session has expired, run this first:
#   aws sso login --profile esflow

echo "Authenticating to ECR (${ECR_REGISTRY}) using profile ${AWS_PROFILE}..."
aws ecr get-login-password --region "${AWS_REGION}" --profile "${AWS_PROFILE}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "Building image ${IMAGE_NAME}:${IMAGE_TAG}..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${REPO_ROOT}/Dockerfile" "${REPO_ROOT}"

echo "Pushing ${IMAGE_NAME}:${IMAGE_TAG}..."
docker push "${IMAGE_NAME}:${IMAGE_TAG}"

echo "Done. Pushed ${IMAGE_NAME}:${IMAGE_TAG}"
