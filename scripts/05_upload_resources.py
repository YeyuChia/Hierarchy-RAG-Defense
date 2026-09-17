# 04_upload_resources.py
# Purpose: Upload prepared files in ./resources to GCP/AWS buckets based on file_to_storage_info.json
# Note: Business logic preserved. Comments normalized; [INFO]/[ERROR] log tags.

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SKIP_AWS = os.environ.get("SKIP_AWS", "0").strip().lower() in {"1", "true", "yes"}
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
AWS_ADMIN_ACCESS_KEY = os.environ.get("AWS_ADMIN_ACCESS_KEY", "")
AWS_ADMIN_SECRET_KEY = os.environ.get("AWS_ADMIN_SECRET_KEY", "")

ARTIFACTS_DIR = Path("artifacts")

STORAGE_MAP_FILE = ARTIFACTS_DIR / "file_to_storage_info.json"
RESOURCE_DIR = ARTIFACTS_DIR / "resources"

# ---------------------------------------------------------------------
# SDK clients
# ---------------------------------------------------------------------

# GCP
gcp_client = gcp_storage.Client.from_service_account_json(GCP_ADMIN_KEY_PATH)

# AWS (optional)
s3_client = None
if not SKIP_AWS:
    if not AWS_ADMIN_ACCESS_KEY or not AWS_ADMIN_SECRET_KEY:
        raise SystemExit("[ERROR] AWS keys missing. Set AWS_* in .env or SKIP_AWS=1")
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ADMIN_ACCESS_KEY,
        aws_secret_access_key=AWS_ADMIN_SECRET_KEY,
        region_name=AWS_REGION,
    )
else:
    print("[INFO] SKIP_AWS=1 → AWS uploads disabled.")

# ---------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------

def upload_to_gcp(bucket_name: str, file_path: str, blob_name: str) -> None:
    """Upload a single file to a GCS bucket (skip if already present)."""
    bucket = gcp_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists():
        return
    blob.upload_from_filename(file_path)
    try:
        print(f"[GCP][INFO] Uploaded: {blob_name} -> {bucket_name}")
    except UnicodeEncodeError:
        print(f"[GCP][INFO] Uploaded -> {bucket_name}")

def upload_to_aws(bucket_name: str, file_path: str, key: str) -> None:
    """Upload a single file to an S3 bucket."""
    if s3_client is None:
        print(f"[AWS][SKIP] {key} → {bucket_name} (SKIP_AWS=1)")
        return
    s3_client.upload_file(file_path, bucket_name, key)
    print(f"[AWS][INFO] Uploaded: {key} → {bucket_name}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _list_existing_gcp(bucket_name):
    """One list call per bucket instead of exists() per file."""
    names = {blob.name for blob in gcp_client.list_blobs(bucket_name)}
    print(f"[INFO] {bucket_name}: {len(names)} existing objects")
    return names


def _upload_one_gcp(bucket_name, file_path, blob_name):
    gcp_client.bucket(bucket_name).blob(blob_name).upload_from_filename(file_path)
    return blob_name


def upload_all() -> None:
    """Upload all files under ./resources according to the storage map."""
    with open(STORAGE_MAP_FILE, "r", encoding="utf-8") as f:
        storage_map = json.load(f)

    gcp_buckets = sorted({
        info["bucket"]
        for info in storage_map.values()
        if info.get("provider") == "gcp"
    })
    existing = {b: _list_existing_gcp(b) for b in gcp_buckets}

    pending = []
    n = len(storage_map)
    ok = skip = err = 0
    for file_name, info in storage_map.items():
        provider = info["provider"]
        bucket = info["bucket"]
        file_path = os.path.join(RESOURCE_DIR, provider.upper(), bucket, file_name)

        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            err += 1
            continue

        if provider == "gcp":
            if file_name in existing.get(bucket, set()):
                skip += 1
            else:
                pending.append((provider, bucket, file_path, file_name))
        elif provider == "aws":
            pending.append((provider, bucket, file_path, file_name))
        else:
            print(f"[ERROR] Unknown provider: {provider}")
            err += 1

    print(f"[INFO] map={n} skip={skip} pending={len(pending)} missing_local={err}")
    workers = int(os.environ.get("UPLOAD_WORKERS", "16"))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for provider, bucket, file_path, file_name in pending:
            if provider == "gcp":
                futs[pool.submit(_upload_one_gcp, bucket, file_path, file_name)] = file_name
            else:
                futs[pool.submit(upload_to_aws, bucket, file_path, file_name)] = file_name
        for fut in as_completed(futs):
            done += 1
            try:
                fut.result()
                ok += 1
            except Exception as e:
                err += 1
                print(f"[ERROR] {futs[fut]}: {e}")
            if done % 200 == 0 or done == len(pending):
                print(f"[PROGRESS] {done}/{len(pending)} uploaded={ok} skipped={skip} err={err}")

    print(f"[INFO] All uploads complete. uploaded={ok} skipped={skip} err={err}")

# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    upload_all()
