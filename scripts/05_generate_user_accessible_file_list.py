# 05_generate_user_accessible_file_list.py
# Purpose: Produce per-user visible file lists from RBAC/ABAC model and storage map.
# Note: Business logic preserved. Comments normalized; [INFO] logs; no emojis.

import json
from pathlib import Path
from collections import defaultdict

ARTIFACTS_DIR = Path("artifacts")

ACCESS_MODEL_FILE = ARTIFACTS_DIR / "generated_access_model.json"
FILE_TO_STORAGE_FILE = ARTIFACTS_DIR / "file_to_storage_info.json"
OUTPUT_FILE = ARTIFACTS_DIR / "user_accessible_files.json"


def _ensure_parent_dir(path_str: str) -> None:
    """Create parent directory if it does not exist."""
    p = Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def generate_access_visibility() -> None:
    # Load inputs
    with open(ACCESS_MODEL_FILE, "r", encoding="utf-8") as f:
        access_model = json.load(f)

    with open(FILE_TO_STORAGE_FILE, "r", encoding="utf-8") as f:
        file_to_storage = json.load(f)

    GCP_RBAC = access_model["GCP_RBAC"]
    AWS_ABAC = access_model["AWS_ABAC"]
    USERS = access_model["USERS"]

    # Build bucket -> files map
    bucket_to_files = defaultdict(list)
    for file_name, storage_info in file_to_storage.items():
        bucket = storage_info["bucket"]
        bucket_to_files[bucket].append(file_name)

    user_visibility = {}

    for user in USERS:
        name = user["name"]

        # Full-access users
        if user.get("full_access"):
            user_visibility[name] = {"full_access": True}
            continue

        visible_files = set()

        # GCP RBAC
        user_roles = user.get("gcp_roles") or []
        for role in user_roles:
            bucket = GCP_RBAC.get(role)
            if bucket and bucket in bucket_to_files:
                visible_files.update(bucket_to_files[bucket])

        # AWS ABAC
        user_attr = user.get("aws_attributes") or {}
        for bucket, required_attr in AWS_ABAC.items():
            match = all(user_attr.get(k) == v for k, v in required_attr.items())
            if match and bucket in bucket_to_files:
                visible_files.update(bucket_to_files[bucket])

        user_visibility[name] = {"files": sorted(visible_files)}

    # Write output
    _ensure_parent_dir(OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(user_visibility, f, indent=2, ensure_ascii=False)

    print(f"[INFO] User-wise access visibility saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_access_visibility()
