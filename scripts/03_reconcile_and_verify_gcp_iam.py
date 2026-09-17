"""Reconcile GCP bucket IAM to the access model, then verify openness.

For each bucket:
  - desired objectViewer members = USERS who have a role (or full_access) on that bucket
  - keep non-experiment members (anything not serviceAccount:user-*@ / baseline-admin@)
  - set objectViewer binding to (kept ∪ desired)

Then recompute openness from live IAM and compare to JSON L_doc.
Writes artifacts/results/gcp_iam_openness_verify.{json,md}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

ACCESS_MODEL = ROOT / "artifacts" / "generated_access_model.json"
OUT_JSON = ROOT / "artifacts" / "results" / "gcp_iam_openness_verify.json"
OUT_MD = ROOT / "artifacts" / "results" / "gcp_iam_openness_verify.md"

ROLE = "roles/storage.objectViewer"
PROJECT = os.environ["GCP_PROJECT_ID"]
ADMIN_KEY = os.environ["GCP_ADMIN_KEY_PATH"]


def sa_member(user_name: str) -> str:
    email = f"{user_name.lower()}@{PROJECT}.iam.gserviceaccount.com"
    return f"serviceAccount:{email}"


def is_experiment_member(member: str) -> bool:
    prefix = f"serviceAccount:"
    if not member.startswith(prefix):
        return False
    email = member[len(prefix) :]
    local = email.split("@", 1)[0]
    return local.startswith("user-") or local == "baseline-admin"


def desired_members(am: dict, bucket: str) -> set:
    gcp_rbac = am["GCP_RBAC"]
    out = set()
    for u in am["USERS"]:
        name = u["name"]
        if u.get("full_access"):
            out.add(sa_member(name))
            continue
        roles = u.get("gcp_roles") or []
        if any(gcp_rbac.get(r) == bucket for r in roles):
            out.add(sa_member(name))
    return out


def get_policy(storage, bucket: str) -> dict:
    return storage.buckets().getIamPolicy(bucket=bucket).execute()


def set_policy(storage, bucket: str, policy: dict) -> None:
    for attempt in range(6):
        try:
            storage.buckets().setIamPolicy(bucket=bucket, body=policy).execute()
            return
        except Exception as e:
            print(f"[WARN] setIamPolicy {bucket} attempt {attempt + 1}: {e}")
            time.sleep(3 + attempt)
    raise SystemExit(f"failed to set IAM for {bucket}")


def reconcile_bucket(storage, bucket: str, desired: set) -> dict:
    policy = get_policy(storage, bucket)
    bindings = policy.setdefault("bindings", [])
    viewer = None
    for b in bindings:
        if b.get("role") == ROLE:
            viewer = b
            break
    if viewer is None:
        viewer = {"role": ROLE, "members": []}
        bindings.append(viewer)

    old = list(viewer.get("members") or [])
    kept = [m for m in old if not is_experiment_member(m)]
    new_members = sorted(set(kept) | desired)
    added = sorted(desired - set(old))
    removed = sorted(set(m for m in old if is_experiment_member(m)) - desired)

    viewer["members"] = new_members
    if added or removed:
        set_policy(storage, bucket, policy)
        # wait briefly for consistency
        time.sleep(2)
        policy = get_policy(storage, bucket)

    live_exp = sorted(
        m
        for b in policy.get("bindings", [])
        if b.get("role") == ROLE
        for m in (b.get("members") or [])
        if is_experiment_member(m)
    )
    return {
        "bucket": bucket,
        "desired_n": len(desired),
        "live_experiment_n": len(live_exp),
        "added": added,
        "removed": removed,
        "match": set(live_exp) == desired,
        "live_experiment_members": live_exp,
    }


def main() -> None:
    am = json.loads(ACCESS_MODEL.read_text(encoding="utf-8"))
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        ADMIN_KEY, ["https://www.googleapis.com/auth/cloud-platform"]
    )
    storage = build("storage", "v1", credentials=creds)

    n_users = len(am["USERS"])
    tiers = am.get("BUCKET_TIERS") or {}
    ranked = sorted(tiers, key=lambda b: int((tiers[b] or {}).get("tier_rank") or 99))

    print(f"[INFO] reconciling {len(ranked)} buckets for {n_users} users", flush=True)
    reports = []
    for bucket in ranked:
        desired = desired_members(am, bucket)
        print(
            f"[INFO] {bucket} ({tiers[bucket].get('tier')}): "
            f"desired experiment readers={len(desired)}",
            flush=True,
        )
        rep = reconcile_bucket(storage, bucket, desired)
        # JSON openness uses ALL USERS who can access, including via full_access
        json_readers = len(desired)  # desired already includes full_access + role users
        live_readers = rep["live_experiment_n"]
        rep.update(
            {
                "tier": tiers[bucket].get("tier"),
                "tier_rank": tiers[bucket].get("tier_rank"),
                "json_openness": json_readers / n_users,
                "json_L_doc": 1.0 - json_readers / n_users,
                "live_openness": live_readers / n_users,
                "live_L_doc": 1.0 - live_readers / n_users,
            }
        )
        reports.append(rep)
        status = "OK" if rep["match"] else "MISMATCH"
        print(
            f"  [{status}] live={live_readers}/{n_users} "
            f"L_doc_live={rep['live_L_doc']:.3f} "
            f"L_doc_json={rep['json_L_doc']:.3f} "
            f"+{len(rep['added'])} -{len(rep['removed'])}",
            flush=True,
        )

    all_ok = all(r["match"] for r in reports)
    payload = {
        "project": PROJECT,
        "n_users": n_users,
        "querying_user": (am.get("DESIGN") or {}).get("querying_user"),
        "all_match": all_ok,
        "buckets": reports,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# GCP IAM openness verification",
        "",
        f"- project: `{PROJECT}`",
        f"- n_users: {n_users}",
        f"- querying_user: `{payload['querying_user']}`",
        f"- all_match: **{all_ok}**",
        "",
        "| tier | bucket | JSON readers | live readers | JSON L_doc | live L_doc | match |",
        "|------|--------|--------------|--------------|------------|------------|-------|",
    ]
    for r in reports:
        lines.append(
            f"| {r['tier']} | `{r['bucket']}` | {r['desired_n']}/{n_users} | "
            f"{r['live_experiment_n']}/{n_users} | {r['json_L_doc']:.3f} | "
            f"{r['live_L_doc']:.3f} | {r['match']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {OUT_MD} all_match={all_ok}", flush=True)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
