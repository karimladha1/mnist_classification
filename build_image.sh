#!/usr/bin/env bash
set -e

IMAGE_NAME="mnist-fastapi"
IMAGE_TAG="latest"

echo "Building Docker image ${IMAGE_NAME}:${IMAGE_TAG} ..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "Done."
echo "You can run it with:"
echo "  docker run --rm -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
