"""Stamp written_by + writer_level with write-ACL consistency.

Rule (no separate write IAM in access_model → proxy by read role):
  A user may be written_by for bucket B only if they can access B
  (have role_read for B, or full_access).

- Clean and poison: deterministic sample among *eligible* writers for that doc's
  bucket (write proxy = read role). Same prior for both — no forged low-privilege
  author on a high bucket.

When an experiment moves poison to another physical bucket, it must re-stamp
written_by / writer_level for that bucket (see stamp_meta_for_bucket).

WRITER_SAMPLING selects the authorship prior:
  "uniform"  (default, primary result) — no free parameters; every user permitted
             to write to a bucket is equally likely to be the author.
  "low_skew" (appendix robustness) — authorship skewed toward low-privilege users
             via max(HIGH_LEVEL_FLOOR, (1 - L)^2). Makes clean authors *less*
             privileged than the attacker in open buckets, i.e. a conservative
             worst case for any writer-based signal.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
MANIFEST = ARTIFACTS / "poison" / "manifest.json"
METADATA = ARTIFACTS / "metadata.json"
ACCESS_MODEL = ARTIFACTS / "generated_access_model.json"

WRITER_SAMPLING = "uniform"  # "uniform" | "low_skew"
HIGH_LEVEL_FLOOR = 0.08  # only used by "low_skew"


def gcp_user_levels(access_model: dict) -> dict:
    """Same scale as L_doc: max exclusivity among readable GCP buckets."""
    from src.retrievers.trust_filter import compute_user_levels

    return compute_user_levels(access_model, provider_scope="gcp")


def users_who_can_access_bucket(access_model: dict, bucket: str) -> list:
    """Write proxy: same as read access to bucket."""
    gcp_rbac = access_model.get("GCP_RBAC") or {}
    out = []
    for u in access_model.get("USERS") or []:
        name = u.get("name") or ""
        if u.get("full_access"):
            out.append(name)
            continue
        roles = u.get("gcp_roles") or []
        if any(gcp_rbac.get(r) == bucket for r in roles):
            out.append(name)
    return out


def clean_pool(access_model: dict) -> list:
    """Every account that may author a clean doc: has at least one read role.

    Excludes baseline-admin (service account, not an author) and user-1 (no roles).
    Order is fixed by the access model so sampling stays deterministic.
    """
    gcp_rbac = access_model.get("GCP_RBAC") or {}
    out = []
    for u in access_model.get("USERS") or []:
        if u.get("full_access"):
            continue
        if any(gcp_rbac.get(r) for r in (u.get("gcp_roles") or [])):
            out.append(u["name"])
    return out


def eligible_clean_writers(access_model: dict, bucket: str, pool: list) -> list:
    can = set(users_who_can_access_bucket(access_model, bucket))
    return [u for u in pool if u in can]


def writer_weights(levels: dict, candidates: list) -> list:
    if WRITER_SAMPLING == "uniform":
        w = 1.0 / max(len(candidates), 1)
        return [(u, w) for u in candidates]
    if WRITER_SAMPLING != "low_skew":
        raise SystemExit(f"unknown WRITER_SAMPLING={WRITER_SAMPLING!r}")
    raw = []
    for u in candidates:
        lv = float(levels.get(u, 0.5))
        w = max(HIGH_LEVEL_FLOOR, (1.0 - lv) ** 2)
        raw.append((u, w))
    total = sum(w for _, w in raw) or 1.0
    return [(u, w / total) for u, w in raw]


def pick_writer(uuid_str: str, weighted: list) -> str:
    h = int(hashlib.md5(uuid_str.encode("utf-8")).hexdigest(), 16)
    r = (h % 10_000) / 10_000.0
    cum = 0.0
    for user, w in weighted:
        cum += w
        if r < cum:
            return user
    return weighted[-1][0]


def stamp_meta_for_bucket(meta: dict, access_model: dict, bucket: str,
                          levels: dict = None, pool: list = None) -> str:
    """Set written_by / writer_level for meta as if it lives in `bucket`.

    Deterministic in (global_uuid, bucket). Used when pointing poison at a
    physical copy in another bucket so T_writer stays write-ACL consistent.
    """
    levels = levels or gcp_user_levels(access_model)
    pool = pool if pool is not None else clean_pool(access_model)
    elig = eligible_clean_writers(access_model, bucket, pool)
    if not elig:
        elig = [u for u in users_who_can_access_bucket(access_model, bucket)
                if u != "baseline-admin"]
    if not elig:
        raise SystemExit(f"no eligible writer for bucket={bucket}")
    uid = meta.get("global_uuid") or meta.get("file_name") or ""
    writer = pick_writer(f"{uid}|{bucket}", writer_weights(levels, elig))
    meta["written_by"] = writer
    meta["writer_level"] = round(float(levels.get(writer, 0.5)), 4)
    return writer


def main() -> None:
    access_model = json.loads(ACCESS_MODEL.read_text(encoding="utf-8"))
    levels = gcp_user_levels(access_model)
    pool = clean_pool(access_model)

    print(f"[INFO] clean author pool ({len(pool)}): {pool}")

    gcp_rbac = access_model.get("GCP_RBAC") or {}
    for b in sorted(set(gcp_rbac.values())):
        elig = eligible_clean_writers(access_model, b, pool)
        print(f"[INFO] writable(proxy) {b}: n={len(elig)}")

    print(f"[INFO] WRITER_SAMPLING = {WRITER_SAMPLING}")
    print("[INFO] poison writers = same rule as clean (eligible ∩ uniform)")

    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    n_poison = n_clean = 0
    clean_by = Counter()
    poison_by = Counter()
    clean_by_bucket_writer = defaultdict(Counter)
    poison_by_bucket_writer = defaultdict(Counter)

    for m in meta:
        bucket = (m.get("parameters") or {}).get("bucket") or ""
        is_poison = bool(m.get("is_poisoned")) or "POISONED" in (m.get("file_name") or "")
        writer = stamp_meta_for_bucket(m, access_model, bucket, levels, pool)
        if is_poison:
            n_poison += 1
            poison_by[writer] += 1
            poison_by_bucket_writer[bucket][writer] += 1
        else:
            n_clean += 1
            clean_by[writer] += 1
            clean_by_bucket_writer[bucket][writer] += 1

    METADATA.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] stamped poison={n_poison} clean={n_clean} -> {METADATA}")
    print("[DONE] clean writer global:")
    for u in pool:
        c = clean_by.get(u, 0)
        print(f"  {u}: {c} ({100 * c / max(n_clean, 1):.1f}%) L={levels.get(u, 0):.2f}")
    print("[DONE] poison writer by bucket:")
    for b in sorted(poison_by_bucket_writer):
        total = sum(poison_by_bucket_writer[b].values())
        dist = ", ".join(
            f"{w}:{c}({100 * c / total:.0f}%)"
            for w, c in poison_by_bucket_writer[b].most_common(8)
        )
        print(f"  {b}: {dist}")
    print("[DONE] clean writer by bucket:")
    for b in sorted(clean_by_bucket_writer):
        total = sum(clean_by_bucket_writer[b].values())
        dist = ", ".join(
            f"{w}:{c}({100 * c / total:.0f}%)"
            for w, c in clean_by_bucket_writer[b].most_common()
        )
        print(f"  {b}: {dist}")


if __name__ == "__main__":
    main()
