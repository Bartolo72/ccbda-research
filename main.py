import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import boto3
from botocore.exceptions import ClientError
import json
import time
import botocore.exceptions

# VARS
region_name: str = "eu-north-1"
bucket_name: str = "ccbda-research"
role_arn: str = "arn:aws:iam::940819259195:role/AmazonSageMaker-TrainingExecutionRole"

model_image = "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-xgboost:1.5-1"
model_name = f"xgboost-iris-{int(time.time())}"
endpoint_config_name = f"xgboost-iris-endpoint-config-{int(time.time())}"
endpoint_name = f"xgboost-iris-endpoint-{int(time.time())}"

# Boto3 Instances
s3 = boto3.resource("s3")
sagemaker = boto3.client("sagemaker", region_name=region_name)
runtime = boto3.client("sagemaker-runtime", region_name=region_name)


def load_data():
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target

    iris_df = iris_df[["target"] + [col for col in iris_df.columns if col != "target"]]

    train_data, test_data = train_test_split(iris_df, test_size=0.2, random_state=42)

    train_data.to_csv("train.csv", index=False, header=False)
    test_data.to_csv("test.csv", index=False, header=False)
    return train_data, test_data


def object_exists(bucket_name: str, key: str) -> bool:
    try:
        s3.meta.client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise


def upload_data_to_s3(bucket_name: str):
    files = [
        ("train.csv", "iris/train/train.csv"),
        ("test.csv", "iris/test/test.csv"),
    ]

    for local_file, s3_key in files:
        if not object_exists(bucket_name, s3_key):
            print(f"Uploading {local_file} to s3://{bucket_name}/{s3_key}")
            s3.meta.client.upload_file(local_file, bucket_name, s3_key)
        else:
            print(f"Skipped: s3://{bucket_name}/{s3_key} already exists")


def create_training_job(role_arn: str, model_image: str, bucket_name: str):
    training_job_name = f"xgboost-iris-{int(time.time())}"

    sagemaker.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            "TrainingImage": model_image,
            "TrainingInputMode": "File",
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket_name}/iris/train/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket_name}/iris/output/"},
        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={
            "objective": "multi:softmax",
            "num_round": "100",
            "num_class": "3",
        },
    )
    return training_job_name


def wait_for_job(training_job_name: str):
    print("Waiting for the training job to complete...")
    try:
        sagemaker.get_waiter("training_job_completed_or_stopped").wait(
            TrainingJobName=training_job_name
        )
    except botocore.exceptions.WaiterError as e:
        print("Waiter encountered an error:", e)


def create_model(training_job_name: str):
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": model_image,
            "ModelDataUrl": f"s3://{bucket_name}/iris/output/{training_job_name}/output/model.tar.gz",
        },
        ExecutionRoleArn=role_arn,
    )


def create_endpoint():
    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.m5.large",
            }
        ],
    )

    sagemaker.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    print(f"Creating endpoint {endpoint_name}...")
    waiter = sagemaker.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} is in service.")


def test_predict(sample):
    sample_features = sample.drop(columns=["target"]).values

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=",".join(map(str, sample_features.flatten())),
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    predicted_class = result
    true_class = int(sample["target"].values[0])

    print(f"Predicted class: {predicted_class}, True class: {true_class}")


def clean_up():
    sagemaker.delete_endpoint(EndpointName=endpoint_name)


if __name__ == "__main__":
    train_data, test_data = load_data()
    upload_data_to_s3(bucket_name)

    training_job_name = create_training_job(role_arn, model_image, bucket_name)
    wait_for_job(training_job_name)

    create_model(training_job_name)
    create_endpoint()

    sample = test_data.sample(1)
    test_predict(sample)

    # clean_up()
