echo "Submitting AI Platform Transformers job"

# BUCKET_NAME: unique bucket name
BUCKET_NAME=-name-of-your-gs-bucket

# The PyTorch image provided by AI Platform Training.
IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-gpu.1-4

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=transformers_job_$(date +%Y%m%d_%H%M%S)

PACKAGE_PATH=./trainer # this can be a GCS location to a zipped and uploaded package

REGION=us-central1

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --config config.yaml \
    --job-dir ${JOB_DIR} \
    --module-name trainer.task \
    --package-path ${PACKAGE_PATH} \
    -- \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 2e-5

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}