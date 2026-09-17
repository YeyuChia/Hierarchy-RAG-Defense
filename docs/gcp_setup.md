# GCP setup (this repo)

This project talks to **real GCP IAM**. You need a billed project and an admin key. Do not commit keys.

## What you need

1. A GCP project with **Billing** enabled
2. An admin service-account JSON key
3. APIs: Cloud Storage, IAM, IAM Service Account Credentials

## Console steps

1. [Cloud Console](https://console.cloud.google.com/) → project → copy **Project ID**
2. Enable the APIs above
3. IAM & Admin → Service Accounts → Create (`parag-admin` or similar)
4. Roles: `Owner`, or `Storage Admin` + `Service Account Admin` + `Project IAM Admin`
5. Keys → Add key → JSON → save as `artifacts/gcp-admin-key.json`

`.gitignore` already ignores `artifacts/*.json` at the repo root of `artifacts/`.

## `.env`

```env
GCP_PROJECT_ID=your-project-id
GCP_ADMIN_KEY_PATH=./artifacts/gcp-admin-key.json
GCP_BUCKET_LOCATION=asia-northeast3
SKIP_AWS=1
OPENAI_API_KEY=sk-...
```

Quick check:

```powershell
conda activate permission-aware-rag
$env:PYTHONPATH = (Get-Location).Path
python -c "from google.cloud import storage; import os; from dotenv import load_dotenv; load_dotenv(); c=storage.Client.from_service_account_json(os.environ['GCP_ADMIN_KEY_PATH']); print('OK project=', c.project)"
```

## After keys work

```powershell
python scripts/01_generate_access_model.py
python scripts/02_gcp_aws_resource_creator.py   # creates buckets + per-user SAs
python scripts/03_reconcile_and_verify_gcp_iam.py
```

If `02` fails with billing disabled, stop and fix billing — do not retry blindly.

Trust eval (`18`) sets `LOCAL_MOCK_IAM=0` and `SKIP_AWS=1` itself.

## Safety

- Do not paste the JSON key or `.env` into chat or a public repo
- If a key leaked: delete it in Console and create a new one
