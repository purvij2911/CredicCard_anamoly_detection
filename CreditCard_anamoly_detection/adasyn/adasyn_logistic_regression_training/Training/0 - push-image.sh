#!/bin/bash

echo "Connecting to ECR to fetch base image"
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 676101537567.dkr.ecr.us-east-1.amazonaws.com

echo "Building local image"
docker build -t adasyn-logistic-training .

echo "Tagging image to be used to push to ECR"
docker tag adasyn-logistic-training:latest 676101537567.dkr.ecr.us-east-1.amazonaws.com/fraud-detection-testing:adasyn-logistic-training-testing

echo "Pushing image to ECR"
docker push 676101537567.dkr.ecr.us-east-1.amazonaws.com/fraud-detection-testing:adasyn-logistic-training-testing