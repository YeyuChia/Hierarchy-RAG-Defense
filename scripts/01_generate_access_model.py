# 01_generate_access_model.py
"""Generate nested GCP RBAC with monotonic numbering and a pyramidal org.

Numbering (paper-readable):
  - bucket-i: fewer readers as i increases (1=public ... 5=private)
  - user-i: more privilege as i increases (1=none ... 31=full / querying user)

Layer widths follow Ewens & Giroud (NBER WP 34162, 2025): corporate hierarchies
are pyramidal. L_doc is derived: L_doc = 1 - readers / |USERS|.

Real GCP service accounts are provisioned for every user in USERS (script 02).
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("artifacts")
OUTPUT_FILE = OUTPUT_DIR / "generated_access_model.json"

PROJECT_PREFIX = "whsdeuye"

GCP_RBAC = {
    f"role_read_gcp_bucket_{i}": f"{PROJECT_PREFIX}-gcp-bucket-{i}" for i in range(1, 6)
}
R = {i: f"role_read_gcp_bucket_{i}" for i in range(1, 6)}

AWS_ABAC = {
    f"{PROJECT_PREFIX}-aws-bucket-1": {"attribute_a": "a1"},
    f"{PROJECT_PREFIX}-aws-bucket-2": {"attribute_a": "a1", "attribute_b": "b1"},
    f"{PROJECT_PREFIX}-aws-bucket-3": {"attribute_c": "c2"},
    f"{PROJECT_PREFIX}-aws-bucket-4": {"attribute_b": "b2", "attribute_c": "c1"},
    f"{PROJECT_PREFIX}-aws-bucket-5": {"attribute_b": "b2"},
}

# (tier, roles, headcount). Nested: each tier reads every lower-numbered bucket.
# bucket-1 public ... bucket-5 private.
TIER_LADDER = [
    ("public", [R[1]], 12),
    ("team", [R[1], R[2]], 8),
    ("dept", [R[1], R[2], R[3]], 5),
    ("restricted", [R[1], R[2], R[3], R[4]], 3),
    ("private", [R[1], R[2], R[3], R[4], R[5]], 2),
]

# Sparse AWS attrs on a few real accounts (legacy ABAC smoke); not used in GCP Trust path.
AWS_ATTRS = {
    "user-14": {"attribute_c": "c2"},
    "user-22": {"attribute_a": "a1"},
    "user-27": {"attribute_b": "b2", "attribute_c": "c2"},
    "user-30": {"attribute_a": "a1", "attribute_c": "c1"},
    "user-31": {"attribute_a": "a1", "attribute_b": "b1", "attribute_c": "c1"},
}


def build_users() -> list:
    users = [{"name": "user-1", "gcp_roles": None, "aws_attributes": None, "tier": "none"}]
    next_id = 2
    for tier, roles, headcount in TIER_LADDER:
        for _ in range(headcount):
            name = f"user-{next_id}"
            next_id += 1
            entry = {
                "name": name,
                "gcp_roles": list(roles),
                "aws_attributes": AWS_ATTRS.get(name),
                "tier": tier,
            }
            if name == "user-31":
                entry["role_note"] = "querying user"
            users.append(entry)
    users.append({"name": "baseline-admin", "full_access": True, "tier": "admin"})
    return users


USERS = build_users()

BUCKET_TIERS = {
    f"{PROJECT_PREFIX}-gcp-bucket-1": {"tier": "public", "tier_rank": 1},
    f"{PROJECT_PREFIX}-gcp-bucket-2": {"tier": "team", "tier_rank": 2},
    f"{PROJECT_PREFIX}-gcp-bucket-3": {"tier": "dept", "tier_rank": 3},
    f"{PROJECT_PREFIX}-gcp-bucket-4": {"tier": "restricted", "tier_rank": 4},
    f"{PROJECT_PREFIX}-gcp-bucket-5": {"tier": "private", "tier_rank": 5},
}

USER_GROUPS = {
    "abac-test-group": [u["name"] for u in USERS if u.get("aws_attributes")],
}

access_model = {
    "GCP_RBAC": GCP_RBAC,
    "AWS_ABAC": AWS_ABAC,
    "USERS": USERS,
    "USER_GROUPS": USER_GROUPS,
    "BUCKET_TIERS": BUCKET_TIERS,
    "DESIGN": {
        "querying_user": "user-31",
        "openness": "readers / |USERS| ; L_doc = 1 - openness",
        "org_shape": {t: n for t, _, n in TIER_LADDER},
        "numbering": (
            "bucket-i: fewer readers as i increases; "
            "user-i: more privilege as i increases; querying user = highest user id"
        ),
        "note": (
            "Pyramidal widths per Ewens & Giroud (NBER w34162, 2025). "
            "Every USERS entry is provisioned as a real GCP service account (script 02); "
            "openness is verified against live bucket IAM (script 03)."
        ),
    },
}

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(access_model, f, indent=2)
    print(f"[INFO] Access model saved to {OUTPUT_FILE}")
    print(f"[INFO] n_users={len(USERS)} querying_user=user-31")
    for u in USERS:
        if u["name"] in ("user-1", "user-2", "user-13", "user-14", "user-21",
                         "user-22", "user-26", "user-27", "user-29", "user-30", "user-31"):
            roles = u.get("gcp_roles") or []
            print(f"  {u['name']:<10} tier={u.get('tier'):<10} roles={roles}")
    ranked = sorted(BUCKET_TIERS, key=lambda b: BUCKET_TIERS[b]["tier_rank"])
    for b in ranked:
        cnt = sum(
            1
            for u in USERS
            if u.get("full_access")
            or any(GCP_RBAC.get(r) == b for r in (u.get("gcp_roles") or []))
        )
        open_frac = cnt / len(USERS)
        print(
            f"  {b} tier={BUCKET_TIERS[b]['tier']:<10} "
            f"readers={cnt:>2}/{len(USERS)} open={open_frac:.3f} L_doc={1 - open_frac:.3f}"
        )
