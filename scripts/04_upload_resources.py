# 04_upload_resources.py
# Purpose: Upload prepared files in ./resources to GCP/AWS buckets based on file_to_storage_info.json
# Note: Business logic preserved. Comments normalized; [INFO]/[ERROR] log tags.

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage as gcp_storage
import boto3

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

# Load .env from current working directory
load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

GCP_ADMIN_KEY_PATH = os.environ["GCP_ADMIN_KEY_PATH"]
AWS_REGION = os.environ["AWS_REGION"]
AWS_ADMIN_ACCESS_KEY = os.environ["AWS_ADMIN_ACCESS_KEY"]
AWS_ADMIN_SECRET_KEY = os.environ["AWS_ADMIN_SECRET_KEY"]

ARTIFACTS_DIR = Path("artifacts")

STORAGE_MAP_FILE = ARTIFACTS_DIR / "file_to_storage_info.json"
RESOURCE_DIR = ARTIFACTS_DIR / "resources"

# ---------------------------------------------------------------------
# SDK clients
# ---------------------------------------------------------------------

# GCP
gcp_client = gcp_storage.Client.from_service_account_json(GCP_ADMIN_KEY_PATH)

# AWS
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ADMIN_ACCESS_KEY,
    aws_secret_access_key=AWS_ADMIN_SECRET_KEY,
    region_name=AWS_REGION,
)

# ---------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------

def upload_to_gcp(bucket_name: str, file_path: str, blob_name: str) -> None:
    """Upload a single file to a GCS bucket."""
    bucket = gcp_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    print(f"[GCP][INFO] Uploaded: {blob_name} → {bucket_name}")

def upload_to_aws(bucket_name: str, file_path: str, key: str) -> None:
    """Upload a single file to an S3 bucket."""
    s3_client.upload_file(file_path, bucket_name, key)
    print(f"[AWS][INFO] Uploaded: {key} → {bucket_name}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def upload_all() -> None:
    """Upload all files under ./resources according to the storage map."""
    with open(STORAGE_MAP_FILE, "r", encoding="utf-8") as f:
        storage_map = json.load(f)

    for file_name, info in storage_map.items():
        provider = info["provider"]
        bucket = info["bucket"]
        file_path = os.path.join(RESOURCE_DIR, provider.upper(), bucket, file_name)

        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            continue

        if provider == "gcp":
            upload_to_gcp(bucket, file_path, file_name)
        elif provider == "aws":
            upload_to_aws(bucket, file_path, file_name)
        else:
            print(f"[ERROR] Unknown provider: {provider}")

    print("[INFO] All uploads complete.")

# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    upload_all()
