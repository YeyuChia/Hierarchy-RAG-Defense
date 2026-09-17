# 03_prepare_documents.py
# Purpose: Build per-document text files from HotpotQA (supporting + distractor
# paragraphs) and assign them to GCP buckets.
#
# Existing file_to_storage_info.json assignments are preserved so already-uploaded
# supporting-fact blobs keep their buckets. New titles (distractors) are assigned
# with a fixed RNG seed.

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

SKIP_AWS = os.environ.get("SKIP_AWS", "0").strip().lower() in {"1", "true", "yes"}

ARTIFACTS_DIR = Path("artifacts")
HOTPOT_FILE = ARTIFACTS_DIR / "hotpot_dev_distractor_v1.json"
ACCESS_MODEL_PATH = ARTIFACTS_DIR / "generated_access_model.json"
OUTPUT_BASE_DIR = ARTIFACTS_DIR / "resources"
STORAGE_MAP_OUTPUT = ARTIFACTS_DIR / "file_to_storage_info.json"

GCP_REGION = "asia-northeast3"
GCP_IAM = "google iam"
AWS_REGION = "ap-northeast-2"
AWS_IAM = "aws iam"
ASSIGN_SEED = 42


def _ensure_parent_dir(path_str: str) -> None:
    p = Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def sanitize_filename(title: str) -> str:
    title = title.lower().replace(" ", "_")
    return re.sub(r"[^\w\d_]", "", title)


def save_to_local(bucket_dir: str, filename: str, content: str) -> None:
    os.makedirs(bucket_dir, exist_ok=True)
    path = os.path.join(bucket_dir, filename)
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_documents(data: list) -> dict:
    """Every unique article in item['context'] (gold + distractors)."""
    doc_dict = {}
    for item in data:
        for title, sents in item["context"]:
            if title not in doc_dict:
                doc_dict[title] = sents
    return doc_dict


def generate_documents_and_storage_info() -> None:
    with open(HOTPOT_FILE, "r", encoding="utf-8") as f:
        hotpot_data = json.load(f)
    with open(ACCESS_MODEL_PATH, "r", encoding="utf-8") as f:
        access_model = json.load(f)

    gcp_buckets = sorted(set(access_model["GCP_RBAC"].values()))
    aws_buckets = sorted(set(access_model["AWS_ABAC"].keys()))
    if SKIP_AWS:
        print("[INFO] SKIP_AWS=1 → assigning new documents to GCP buckets only.")

    existing = {}
    if STORAGE_MAP_OUTPUT.exists():
        existing = json.loads(STORAGE_MAP_OUTPUT.read_text(encoding="utf-8"))
        print(f"[INFO] preserving {len(existing)} existing storage-map entries")

    doc_map = extract_documents(hotpot_data)
    print(f"[INFO] unique context titles (supporting+distractor)={len(doc_map)}")

    rng = random.Random(ASSIGN_SEED)
    storage_map = dict(existing)
    count_by_provider = {"gcp": 0, "aws": 0}
    count_by_bucket = defaultdict(int)
    n_kept = n_new = n_collision = 0
    seen_files = set()

    for title, sents in tqdm(doc_map.items(), desc="Prepare docs"):
        filename = sanitize_filename(title) + ".txt"
        if filename in seen_files:
            n_collision += 1
            continue
        seen_files.add(filename)
        content = "\n".join(sents)

        if filename in existing:
            info = storage_map[filename]
            provider, bucket = info["provider"], info["bucket"]
            n_kept += 1
        else:
            provider = "gcp" if SKIP_AWS else rng.choice(["gcp", "aws"])
            bucket = rng.choice(gcp_buckets if provider == "gcp" else aws_buckets)
            storage_map[filename] = {
                "provider": provider,
                "bucket": bucket,
                "region": GCP_REGION if provider == "gcp" else AWS_REGION,
                "iam": GCP_IAM if provider == "gcp" else AWS_IAM,
            }
            n_new += 1

        save_to_local(
            os.path.join(OUTPUT_BASE_DIR, provider.upper(), bucket),
            filename,
            content,
        )
        count_by_provider[provider] += 1
        count_by_bucket[bucket] += 1

    _ensure_parent_dir(STORAGE_MAP_OUTPUT)
    STORAGE_MAP_OUTPUT.write_text(
        json.dumps(storage_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n===== Document Storage Summary =====")
    print(f"[INFO] Total documents in map: {len(storage_map)}")
    print(f"[INFO] Reused existing assignments: {n_kept}")
    print(f"[INFO] Newly assigned: {n_new}")
    print(f"[INFO] Filename collisions skipped: {n_collision}")
    print(f"[INFO] GCP documents: {count_by_provider['gcp']}")
    print(f"[INFO] AWS documents: {count_by_provider['aws']}")
    print("\n[INFO] Documents per bucket:")
    for bucket, count in sorted(count_by_bucket.items()):
        print(f"  - {bucket}: {count} files")
    print(f"\n[INFO] Storage info saved to: {STORAGE_MAP_OUTPUT}")


if __name__ == "__main__":
    generate_documents_and_storage_info()
