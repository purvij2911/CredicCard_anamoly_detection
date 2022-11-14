FROM 676101537567.dkr.ecr.us-east-1.amazonaws.com/lambda-base-image:latest

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY adasyn_xgboosting.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "adasyn_xgboosting.handler" ] 