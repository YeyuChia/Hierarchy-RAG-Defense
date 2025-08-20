# 01_generate_access_model.py

import json
import random
import string
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("artifacts")
OUTPUT_FILE = OUTPUT_DIR / "generated_access_model.json"

# Random project prefix (lowercase letters, length 8)
PROJECT_PREFIX = "".join(random.choices(string.ascii_lowercase, k=8))

# GCP role mapping
GCP_RBAC = {
    f"role_read_gcp_bucket_{i+1}": f"{PROJECT_PREFIX}-gcp-bucket-{i+1}"
    for i in range(5)
}

# AWS bucket attribute-based conditions
AWS_ABAC = {
    f"{PROJECT_PREFIX}-aws-bucket-1": {"attribute_a": "a1"},
    f"{PROJECT_PREFIX}-aws-bucket-2": {"attribute_a": "a1", "attribute_b": "b1"},
    f"{PROJECT_PREFIX}-aws-bucket-3": {"attribute_c": "c2"},
    f"{PROJECT_PREFIX}-aws-bucket-4": {"attribute_b": "b2", "attribute_c": "c1"},
    f"{PROJECT_PREFIX}-aws-bucket-5": {"attribute_b": "b2"},
}

# Users (expected ascending access ratio in the order listed)
USERS = [
    {"name": "user-1", "gcp_roles": None, "aws_attributes": None},
    {"name": "user-2", "gcp_roles": ["role_read_gcp_bucket_2"], "aws_attributes": None},
    {
        "name": "user-3",
        "gcp_roles": ["role_read_gcp_bucket_2", "role_read_gcp_bucket_3"],
        "aws_attributes": {"attribute_c": "c2"},
    },
    {
        "name": "user-4",
        "gcp_roles": [
            "role_read_gcp_bucket_1",
            "role_read_gcp_bucket_3",
            "role_read_gcp_bucket_5",
        ],
        "aws_attributes": {"attribute_a": "a1"},
    },
    {
        "name": "user-5",
        "gcp_roles": [
            "role_read_gcp_bucket_1",
            "role_read_gcp_bucket_2",
            "role_read_gcp_bucket_3",
        ],
        "aws_attributes": {"attribute_b": "b2", "attribute_c": "c2"},
    },
    {
        "name": "user-6",
        "gcp_roles": [
            "role_read_gcp_bucket_1",
            "role_read_gcp_bucket_2",
            "role_read_gcp_bucket_3",
            "role_read_gcp_bucket_4",
            "role_read_gcp_bucket_5",
        ],
        "aws_attributes": {"attribute_a": "a1", "attribute_c": "c1"},
    },
    {
        "name": "user-7",
        "gcp_roles": [
            "role_read_gcp_bucket_1",
            "role_read_gcp_bucket_2",
            "role_read_gcp_bucket_3",
            "role_read_gcp_bucket_4",
            "role_read_gcp_bucket_5",
        ],
        "aws_attributes": {"attribute_a": "a1", "attribute_b": "b1", "attribute_c": "c1"},
    },
    {"name": "baseline-admin", "full_access": True},
]

# Group configuration
USER_GROUPS = {
    "abac-test-group": [u["name"] for u in USERS if u.get("aws_attributes")]
}

# Final model structure
access_model = {
    "GCP_RBAC": GCP_RBAC,
    "AWS_ABAC": AWS_ABAC,
    "USERS": USERS,
    "USER_GROUPS": USER_GROUPS,
}

if __name__ == "__main__":
    # Ensure artifacts dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(access_model, f, indent=2)

    print(f"[INFO] Access model saved to {OUTPUT_FILE}")
