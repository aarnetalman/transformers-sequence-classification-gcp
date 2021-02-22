# A name for the service account you are about to create:
export SERVICE_ACCOUNT_NAME=nli-experiments

# Create service account:
gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME} --display-name="Service Account for ai-platform-samples repo"

# Grant the required roles:
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com --role roles/ml.developer
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com --role roles/storage.objectAdmin

# Download the service account key and store it in a file specified by GOOGLE_APPLICATION_CREDENTIALS:
gcloud iam service-accounts keys create ${GOOGLE_APPLICATION_CREDENTIALS} --iam-account ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com
