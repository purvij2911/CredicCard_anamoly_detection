#!/bin/bash

echo 'Define Params'
AWS_REGION='us-east-1'
GATEWAY_NAME='fraud-detection'

API_ID=$(aws apigateway get-rest-apis --region $AWS_REGION --query "items[?name=='$GATEWAY_NAME']" | grep id | cut -d'"' -f4)

if test -z "$API_ID"
then
    echo 'Create API GATEWAY'
    aws apigateway create-rest-api \
        --region $AWS_REGION \
        --name $GATEWAY_NAME
fi

echo 'Retrieve API ID'
API_ID=$(aws apigateway get-rest-apis --region $AWS_REGION --query "items[?name=='$GATEWAY_NAME']" | grep id | cut -d'"' -f4)
RESOURCE_ID=$(aws apigateway get-resources --region us-east-1 --rest-api-id $API_ID --query "items[?path=='/']" | grep id | cut -d'"' -f4)

PATH_PART="adasyn-xgboost-training"
echo 'Creating Resource'
aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $RESOURCE_ID \
    --path-part $PATH_PART

echo 'Retriving Child Resource'
RESOURCE_ID=$(aws apigateway get-resources --region us-east-1 --rest-api-id $API_ID --query "items[?path=='/$PATH_PART']" | grep id | cut -d'"' -f4)

echo 'Creating method for resource '
aws apigateway put-method \
    --region $AWS_REGION \
    --rest-api-id $API_ID \
    --resource-id $RESOURCE_ID \
    --http-method POST \
    --authorization-type "AWS_IAM"

FUNCTION_NAME="fraud-detection-adasyn-xgboost-training"

echo 'Creating method integration for resource'
aws apigateway put-integration \
        --region $AWS_REGION \
        --rest-api-id $API_ID \
        --resource-id $RESOURCE_ID \
        --http-method POST \
        --type AWS \
        --credentials arn:aws:iam::\*:user/\*\
        --integration-http-method POST \
        --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:$AWS_REGION:676101537567:function:$FUNCTION_NAME/invocations 

echo 'Creating deployment for method'
aws apigateway create-deployment \
    --rest-api-id $API_ID \
    --stage-name dev \
    --stage-description 'Development Stage' \
    --description 'First deployment to the dev stage'

# aws apigateway create-resource \
#     --rest-api-id API_ID \
#     --parent-id $RESOURCE_ID \
#     --path-part 'model-training'

# aws apigateway create-resource \
#     --rest-api-id API_ID \
#     --parent-id $RESOURCE_ID \
#     --path-part 'predict'