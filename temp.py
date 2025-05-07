import sagemaker

image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost", region="eu-north-1", version="1.5-1"
)

print("Image URI:", image_uri)
