
export RUNTIME_VERSION=2.1
export PYTHON_VERSION=3.7
export REGION=us-central1

# GCP PROJECT ID
export PROJECT_ID="your-project-id"

# unique name for a GCS bucket. Do not include gs://
export BUCKET_NAME="your-bucket-name"

echo "RUNTIME_VERSION='${RUNTIME_VERSION}'"
echo "PYTHON_VERSION='${PYTHON_VERSION}'"
echo "REGION='${REGION}'"
echo "PROJECT_ID='${PROJECT_ID}'"
echo "BUCKET_NAME='${BUCKET_NAME}'"