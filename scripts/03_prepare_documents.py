# 03_prepare_documents.py
# Purpose: Build per-document text files from HotpotQA and assign them to GCP/AWS buckets.
# Note: Business logic preserved. Comments normalized; no emojis; [INFO]/[WARN]/[ERROR] tags.

import os
import json
import random
import re
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

# Load environment variables from .env in current working directory
load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")

HOTPOT_FILE = ARTIFACTS_DIR / "hotpot_dev_distractor_v1.json"
ACCESS_MODEL_PATH = ARTIFACTS_DIR / "generated_access_model.json"
OUTPUT_BASE_DIR = ARTIFACTS_DIR / "resources"
STORAGE_MAP_OUTPUT = ARTIFACTS_DIR / "file_to_storage_info.json"

GCP_REGION = "asia-northeast3"
GCP_ENDPOINT = "https://storage.googleapis.com"  # reserved for future use
GCP_IAM = "google iam"

AWS_REGION = "ap-northeast-2"
AWS_ENDPOINT = "https://s3.ap-northeast-2.amazonaws.com"  # reserved for future use
AWS_IAM = "aws iam"

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def _ensure_parent_dir(path_str: str) -> None:
    """Create parent directory for a file path if it does not exist."""
    p = Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

def sanitize_filename(title: str) -> str:
    """Lowercase, replace spaces with underscores, strip non-word characters."""
    title = title.lower().replace(" ", "_")
    return re.sub(r"[^\w\d_]", "", title)

def save_to_local(bucket_dir: str, filename: str, content: str) -> None:
    """Persist content under resources/<PROVIDER>/<BUCKET>/<filename>."""
    os.makedirs(bucket_dir, exist_ok=True)
    with open(os.path.join(bucket_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)

def extract_documents(data: list) -> dict:
    """Build a mapping from article title to its sentence list."""
    doc_dict = {}
    for item in data:
        context = item["context"]
        for title, sents in context:
            if title not in doc_dict:
                doc_dict[title] = sents
    return doc_dict

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def generate_documents_and_storage_info() -> None:
    # Load inputs
    with open(HOTPOT_FILE, "r", encoding="utf-8") as f:
        hotpot_data = json.load(f)
    with open(ACCESS_MODEL_PATH, "r", encoding="utf-8") as f:
        access_model = json.load(f)

    GCP_BUCKETS = list(set(access_model["GCP_RBAC"].values()))
    AWS_BUCKETS = list(set(access_model["AWS_ABAC"].keys()))
    PROVIDERS = ["gcp", "aws"]

    doc_map = extract_documents(hotpot_data)
    storage_map = {}

    used_titles = set()

    # Statistics counters
    count_by_provider = {"gcp": 0, "aws": 0}
    count_by_bucket = defaultdict(int)

    for item in tqdm(hotpot_data):
        for title, _ in item["supporting_facts"]:
            if title in used_titles or title not in doc_map:
                continue
            used_titles.add(title)

            content = "\n".join(doc_map[title])
            filename = sanitize_filename(title) + ".txt"

            provider = random.choice(PROVIDERS)
            if provider == "gcp":
                bucket = random.choice(GCP_BUCKETS)
                save_to_local(os.path.join(OUTPUT_BASE_DIR, "GCP", bucket), filename, content)
                count_by_provider["gcp"] += 1
            else:
                bucket = random.choice(AWS_BUCKETS)
                save_to_local(os.path.join(OUTPUT_BASE_DIR, "AWS", bucket), filename, content)
                count_by_provider["aws"] += 1

            count_by_bucket[bucket] += 1

            storage_map[filename] = {
                "provider": provider,
                "bucket": bucket,
                "region": GCP_REGION if provider == "gcp" else AWS_REGION,
                "iam": GCP_IAM if provider == "gcp" else AWS_IAM,
            }

    # Write storage map
    _ensure_parent_dir(STORAGE_MAP_OUTPUT)
    with open(STORAGE_MAP_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(storage_map, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n===== Document Storage Summary =====")
    print(f"[INFO] Total documents processed: {len(storage_map)}")
    print(f"[INFO] GCP documents: {count_by_provider['gcp']}")
    print(f"[INFO] AWS documents: {count_by_provider['aws']}")
    print("\n[INFO] Documents per bucket:")
    for bucket, count in count_by_bucket.items():
        print(f"  - {bucket}: {count} files")
    print(f"\n[INFO] Storage info saved to: {STORAGE_MAP_OUTPUT}")

# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    generate_documents_and_storage_info()
