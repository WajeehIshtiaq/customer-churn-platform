FROM public.ecr.aws/lambda/python:3.14

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/

RUN pip install --no-cache-dir \
    -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY api.py lambda_handler.py ${LAMBDA_TASK_ROOT}/

COPY models/gradient_boosting_pipeline.joblib ${LAMBDA_TASK_ROOT}/models/
COPY models/prediction_config.json ${LAMBDA_TASK_ROOT}/models/

CMD ["lambda_handler.handler"]