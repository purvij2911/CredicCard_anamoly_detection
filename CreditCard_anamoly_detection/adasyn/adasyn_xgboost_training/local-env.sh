#!/bin/bash
echo "Building docker image"
docker build -t adasyn-xgboost-training .

echo "Docker run + Mount local credentials to docker"
docker run -p 9000:8080 -v ~/.aws:/root/.aws  adasyn-xgboost-training