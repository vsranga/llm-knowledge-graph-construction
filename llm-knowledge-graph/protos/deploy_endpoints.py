import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import json

# Initialize SageMaker session
sess = sagemaker.Session()

# Get or create SageMaker execution role
try:
    # role = sagemaker.get_execution_role()
    role = 'arn:aws:iam::387324564712:role/SageMakerExecutionRole'
except ValueError:
    # Create role ARN for your account
    sts_client = boto3.client('sts')
    account_id = sts_client.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"


# Deploy LLM endpoint
# llm_model = HuggingFaceModel(
#    role=role,
#    transformers_version="4.37",
#    pytorch_version="2.1",
#    py_version="py310",
#    env={
#        'HF_MODEL_ID': 'google/flan-t5-base',
#        'HF_TASK': 'text2text-generation'
#    }
# )


# llm_predictor = llm_model.deploy(
#    initial_instance_count=1,
#    instance_type="ml.m5.xlarge",
#    endpoint_name="flan-t5-base-endpoint"
# )



# Deploy embedding endpoint
embedding_model = HuggingFaceModel(
    role=role,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
    env={
        'HF_MODEL_ID': 'sentence-transformers/all-mpnet-base-v2',
        'HF_TASK': 'feature-extraction'
    }
)
embedding_predictor = embedding_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="mpnet-embedding-endpoint"
)
