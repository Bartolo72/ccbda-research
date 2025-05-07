import os
import scipy.io
import urllib.request
from sklearn.model_selection import train_test_split
import shutil

# Download image files
os.makedirs("flowers", exist_ok=True)
urllib.request.urlretrieve(
    "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", "102flowers.tgz"
)
urllib.request.urlretrieve(
    "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
    "imagelabels.mat",
)

# Extract
import tarfile

with tarfile.open("102flowers.tgz") as tar:
    tar.extractall(path="flowers")

# Load labels
labels = scipy.io.loadmat("imagelabels.mat")["labels"][0]
image_paths = sorted(os.listdir("flowers/jpg"))

# Split train/val
train_idx, val_idx = train_test_split(
    range(len(image_paths)), test_size=0.2, random_state=42, stratify=labels
)


def organize(split_idx, split_name):
    for i in split_idx:
        label = labels[i]
        folder = f"data/{split_name}/{label}"
        os.makedirs(folder, exist_ok=True)
        src = f"flowers/jpg/{image_paths[i]}"
        dst = f"{folder}/{image_paths[i]}"
        shutil.copy(src, dst)


organize(train_idx, "train")
organize(val_idx, "validation")


import sagemaker
from sagemaker.s3 import S3Uploader

bucket = "ccbda-research-sagemaker"
prefix = "image-classification"
region = "eu-north-1"

train_s3 = S3Uploader.upload("data/train", f"s3://{bucket}/{prefix}/train")
val_s3 = S3Uploader.upload("data/validation", f"s3://{bucket}/{prefix}/validation")


from sagemaker import image_uris, Estimator

role = "arn:aws:iam::940819259195:role/AmazonSageMaker-TrainingExecutionRole"

image_uri = image_uris.retrieve(framework="image-classification", region=region)

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",  # Or use 'ml.p3.2xlarge' for GPU
    volume_size=50,
    max_run=3600,
    input_mode="File",
    output_path=f"s3://{bucket}/{prefix}/output",
    sagemaker_session=sagemaker.Session(),
)

# Define hyperparameters for transfer learning
estimator.set_hyperparameters(
    num_layers=18,
    use_pretrained_model=1,
    num_classes=2,
    mini_batch_size=32,
    epochs=10,
    learning_rate=0.01,
    image_shape="3,224,224",
    augmentation_type="crop_color_transform",
    precision_dtype="float32",
)


from sagemaker.inputs import TrainingInput

train_input = TrainingInput(train_s3, content_type="application/x-image")
val_input = TrainingInput(val_s3, content_type="application/x-image")

estimator.fit({"train": train_input, "validation": val_input})


predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",  # Or 'ml.m5.xlarge' for better performance
)

# Predict with a single image
from PIL import Image
import numpy as np

image = Image.open("data/validation/daisy/001.jpg").resize((224, 224))
np_image = np.asarray(image).transpose(2, 0, 1).astype(np.float32)
payload = np_image.tobytes()

response = predictor.predict(payload)
print(response)
