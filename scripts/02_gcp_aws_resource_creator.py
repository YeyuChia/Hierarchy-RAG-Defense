# 02_gcp_aws_resource_creator.py
# Purpose: Provision GCP/AWS buckets, IAM entities, and emit per-user credentials.
# Note: Business logic preserved. Comments normalized; no emojis; [INFO]/[WARN]/[ERROR] log tags.

import os
import json
import time
import base64
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import boto3

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

# Load .env from repo root or current working directory
load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_ADMIN_KEY_PATH = os.environ["GCP_ADMIN_KEY_PATH"]
GCP_BUCKET_LOCATION = os.environ.get("GCP_BUCKET_LOCATION", "asia-northeast3")
GCP_BUCKET_STORAGE_CLASS = os.environ.get("GCP_BUCKET_STORAGE_CLASS", "STANDARD")

# Set SKIP_AWS=1 to run GCP-only (no AWS credentials / provisioning).
SKIP_AWS = os.environ.get("SKIP_AWS", "0").strip().lower() in {"1", "true", "yes"}

AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
AWS_ADMIN_ACCESS_KEY = os.environ.get("AWS_ADMIN_ACCESS_KEY", "")
AWS_ADMIN_SECRET_KEY = os.environ.get("AWS_ADMIN_SECRET_KEY", "")

ARTIFACTS_DIR = Path("artifacts")

ACCESS_MODEL_PATH = ARTIFACTS_DIR / "generated_access_model.json"
CREDENTIAL_OUTPUT_PATH = ARTIFACTS_DIR / "test_credential.txt"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_parent_dir(path_str: str) -> None:
    """Create parent directory if it does not exist."""
    p = Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load access model
# ---------------------------------------------------------------------

with open(ACCESS_MODEL_PATH, "r", encoding="utf-8") as f:
    access_model = json.load(f)

GCP_RBAC = access_model["GCP_RBAC"]
AWS_ABAC = access_model["AWS_ABAC"]
USERS = access_model["USERS"]
USER_GROUPS = access_model.get("USER_GROUPS", {})

# ---------------------------------------------------------------------
# SDK clients
# ---------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
gcp_credentials = ServiceAccountCredentials.from_json_keyfile_name(GCP_ADMIN_KEY_PATH, SCOPES)
gcp_storage = build("storage", "v1", credentials=gcp_credentials)
gcp_iam = build("iam", "v1", credentials=gcp_credentials)

aws_s3 = None
aws_iam = None
if not SKIP_AWS:
    if not AWS_ADMIN_ACCESS_KEY or not AWS_ADMIN_SECRET_KEY:
        raise SystemExit("[ERROR] AWS keys missing. Set AWS_* in .env or SKIP_AWS=1")
    aws_s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ADMIN_ACCESS_KEY,
        aws_secret_access_key=AWS_ADMIN_SECRET_KEY,
        region_name=AWS_REGION,
    )
    aws_iam = boto3.client(
        "iam",
        aws_access_key_id=AWS_ADMIN_ACCESS_KEY,
        aws_secret_access_key=AWS_ADMIN_SECRET_KEY,
        region_name=AWS_REGION,
    )
else:
    print("[INFO] SKIP_AWS=1 → AWS clients not created (GCP-only).")

# ---------------------------------------------------------------------
# GCP
# ---------------------------------------------------------------------

def create_gcp_buckets() -> None:
    """Create GCP buckets if they do not exist."""
    for bucket_name in set(GCP_RBAC.values()):
        try:
            gcp_storage.buckets().insert(
                project=GCP_PROJECT_ID,
                body={
                    "name": bucket_name,
                    "location": GCP_BUCKET_LOCATION,
                    "storageClass": GCP_BUCKET_STORAGE_CLASS,
                },
            ).execute()
            print(f"[GCP][INFO] Created bucket: {bucket_name}")
        except Exception as e:
            msg = str(e)
            if "already exists" in msg:
                print(f"[GCP][INFO] Bucket already exists: {bucket_name}")
            else:
                print(f"[GCP][ERROR] Error creating bucket {bucket_name}: {e}")

def setup_gcp_users() -> dict:
    """Create service accounts, bind storage.objectViewer to target buckets, and issue keys."""
    result = {}

    for user in USERS:
        name = user["name"]
        roles = user.get("gcp_roles") or []
        full_access = user.get("full_access", False)

        sa_id = name.lower()
        sa_email = f"{sa_id}@{GCP_PROJECT_ID}.iam.gserviceaccount.com"
        sa_resource = f"projects/{GCP_PROJECT_ID}/serviceAccounts/{sa_email}"

        # Create service account with simple backoff on quota
        for attempt in range(5):
            try:
                gcp_iam.projects().serviceAccounts().create(
                    name=f"projects/{GCP_PROJECT_ID}",
                    body={"accountId": sa_id, "serviceAccount": {"displayName": sa_id}},
                ).execute()
                print(f"[GCP][INFO] Created service account: {sa_email}")
                break
            except Exception as e:
                emsg = str(e).lower()
                if "quota" in emsg or "per minute" in emsg:
                    print("[GCP][WARN] Quota reached, sleeping 60s for retry...")
                    time.sleep(60)
                elif "already exists" in emsg:
                    print(f"[GCP][INFO] Service account already exists: {sa_email}")
                    break
                else:
                    print(f"[GCP][ERROR] Error creating service account {sa_id}: {e}")
                    break

        # Wait until service account is available
        for _ in range(10):
            try:
                gcp_iam.projects().serviceAccounts().get(name=sa_resource).execute()
                time.sleep(5)
                break
            except Exception:
                time.sleep(10)

        # Determine target buckets
        target_buckets = list(GCP_RBAC.values()) if full_access else [GCP_RBAC[r] for r in roles]

        # Grant roles/storage.objectViewer on target buckets
        for bucket_name in target_buckets:
            try:
                gcp_iam.projects().serviceAccounts().get(name=sa_resource).execute()
            except Exception as e:
                print(f"[GCP][ERROR] Skipping policy for {bucket_name}; service account not found: {e}")
                continue

            policy = gcp_storage.buckets().getIamPolicy(bucket=bucket_name).execute()
            member = f"serviceAccount:{sa_email}"
            role_key = "roles/storage.objectViewer"
            binding_exists = False

            for b in policy.get("bindings", []):
                if b["role"] == role_key:
                    if member not in b["members"]:
                        b["members"].append(member)
                    binding_exists = True
                    break

            if not binding_exists:
                policy.setdefault("bindings", []).append({"role": role_key, "members": [member]})

            for attempt in range(5):
                try:
                    gcp_storage.buckets().setIamPolicy(bucket=bucket_name, body=policy).execute()
                    break
                except Exception as e:
                    print(f"[GCP][WARN] Retry {attempt + 1} setIamPolicy for {bucket_name}: {e}")
                    time.sleep(2)

        # Issue key file for the service account (reuse existing file key if quota hit)
        try:
            key = gcp_iam.projects().serviceAccounts().keys().create(
                name=sa_resource,
                body={"privateKeyType": "TYPE_GOOGLE_CREDENTIALS_FILE"},
            ).execute()
            key_json = json.loads(base64.b64decode(key["privateKeyData"]).decode("utf-8"))
            result[name] = key_json
        except Exception as e:
            print(f"[GCP][WARN] Failed to create key for {name}: {e}")
            # Fall back to previously saved credential if present
            if CREDENTIAL_OUTPUT_PATH.exists():
                try:
                    prev = json.loads(CREDENTIAL_OUTPUT_PATH.read_text(encoding="utf-8"))
                    for row in prev:
                        if row.get("name") == name and row.get("gcp_credential"):
                            result[name] = row["gcp_credential"]
                            print(f"[GCP][INFO] Reused existing credential for {name}")
                            break
                except Exception:
                    pass
            if name not in result:
                print(f"[GCP][ERROR] No credential available for {name}")

    return result

# ---------------------------------------------------------------------
# AWS
# ---------------------------------------------------------------------

def create_aws_buckets() -> None:
    """Create S3 buckets if they do not exist."""
    for bucket_name in AWS_ABAC.keys():
        try:
            if AWS_REGION == "us-east-1":
                aws_s3.create_bucket(Bucket=bucket_name)
            else:
                aws_s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
                )
            print(f"[AWS][INFO] Created bucket: {bucket_name}")
        except aws_s3.exceptions.BucketAlreadyOwnedByYou:
            print(f"[AWS][INFO] Bucket already exists: {bucket_name}")
        except Exception as e:
            print(f"[AWS][ERROR] Error creating bucket {bucket_name}: {e}")

def setup_aws_users_and_abac_groups() -> dict:
    """
    Create ABAC-oriented IAM group(s), attach bucket policies conditioned on tags,
    and add users to groups per USER_GROUPS.
    """
    result = {}
    created_groups = set()

    # Create groups
    for group_name in USER_GROUPS:
        try:
            aws_iam.create_group(GroupName=group_name)
            print(f"[AWS][INFO] Created group: {group_name}")
        except aws_iam.exceptions.EntityAlreadyExistsException:
            print(f"[AWS][INFO] Group already exists: {group_name}")
        created_groups.add(group_name)

    # Add users to groups
    for group_name, members in USER_GROUPS.items():
        for user_name in members:
            try:
                aws_iam.add_user_to_group(GroupName=group_name, UserName=user_name)
                print(f"[AWS][INFO] Added {user_name} to group {group_name}")
            except Exception as e:
                print(f"[AWS][ERROR] Failed to add {user_name} to group {group_name}: {e}")

    # Create and attach policies derived from AWS_ABAC
    for bucket_name, attr_dict in AWS_ABAC.items():
        condition = {f"aws:PrincipalTag/{k}": v for k, v in attr_dict.items()}
        safe_sid = f"Access{''.join(filter(str.isalnum, bucket_name))}"
        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": safe_sid,
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                    "Condition": {"StringEquals": condition},
                }
            ],
        }

        policy_name = f"abac-{bucket_name}-policy"
        try:
            response = aws_iam.create_policy(PolicyName=policy_name, PolicyDocument=json.dumps(policy_doc))
            policy_arn = response["Policy"]["Arn"]
            print(f"[AWS][INFO] Created policy: {policy_name}")
        except aws_iam.exceptions.EntityAlreadyExistsException:
            account_id = boto3.client("sts").get_caller_identity()["Account"]
            policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
            print(f"[AWS][INFO] Policy already exists: {policy_name}")

        for group_name in created_groups:
            try:
                aws_iam.attach_group_policy(GroupName=group_name, PolicyArn=policy_arn)
                print(f"[AWS][INFO] Attached policy {policy_name} to group {group_name}")
            except Exception as e:
                print(f"[AWS][ERROR] Failed to attach policy {policy_name} to group {group_name}: {e}")

    return result

def create_aws_users() -> dict:
    """Create IAM users, tag them for ABAC, optionally attach full S3 access, and issue access keys."""
    result = {}

    for user in USERS:
        name = user["name"]
        tags = user.get("aws_attributes") or {}
        full_access = user.get("full_access", False)

        tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]
        try:
            aws_iam.create_user(UserName=name, Tags=tag_list)
            print(f"[AWS][INFO] Created user: {name}")
        except aws_iam.exceptions.EntityAlreadyExistsException:
            print(f"[AWS][INFO] User already exists: {name}")

        if full_access:
            try:
                policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
                aws_iam.attach_user_policy(UserName=name, PolicyArn=policy_arn)
                print(f"[AWS][INFO] Attached AmazonS3FullAccess to {name}")
            except Exception as e:
                print(f"[AWS][ERROR] Failed to attach full access to {name}: {e}")

        try:
            key = aws_iam.create_access_key(UserName=name)["AccessKey"]
            result[name] = {
                "aws_access_key": key["AccessKeyId"],
                "aws_secret_key": key["SecretAccessKey"],
            }
        except Exception as e:
            print(f"[AWS][ERROR] Failed to create access key for {name}: {e}")

    return result

# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def setup_gcp() -> None:
    """Provision GCP resources and write partial credential file."""
    create_gcp_buckets()
    gcp_keys = setup_gcp_users()

    merged = []
    for user in USERS:
        name = user["name"]
        merged.append(
            {
                "name": name,
                "gcp_credential": gcp_keys.get(name, {}),
                "aws_access_key": "",
                "aws_secret_key": "",
            }
        )

    _ensure_parent_dir(CREDENTIAL_OUTPUT_PATH)
    with open(CREDENTIAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[GCP][INFO] GCP credentials saved to {CREDENTIAL_OUTPUT_PATH}")

def setup_aws() -> None:
    """Provision AWS resources and append access keys into the credential file."""
    create_aws_buckets()
    aws_keys = create_aws_users()
    setup_aws_users_and_abac_groups()

    with open(CREDENTIAL_OUTPUT_PATH, "r", encoding="utf-8") as f:
        merged = json.load(f)

    for user in merged:
        uname = user["name"]
        user["aws_access_key"] = aws_keys.get(uname, {}).get("aws_access_key", "")
        user["aws_secret_key"] = aws_keys.get(uname, {}).get("aws_secret_key", "")

    with open(CREDENTIAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[AWS][INFO] AWS credentials appended to {CREDENTIAL_OUTPUT_PATH}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    setup_gcp()
    if SKIP_AWS:
        print("[INFO] Skipped setup_aws() because SKIP_AWS=1")
    else:
        setup_aws()
