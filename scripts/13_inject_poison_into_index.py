"""Install poison: inject FAISS/metadata, assign 50 Q per bucket, upload to that bucket only.

  python scripts/13_inject_poison_into_index.py --replace
  python scripts/13_inject_poison_into_index.py --place-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

ARTIFACTS = ROOT / "artifacts"
MANIFEST = ARTIFACTS / "poison" / "manifest.json"
DOCS_DIR = ARTIFACTS / "poison" / "docs"
ASSIGNMENT = ARTIFACTS / "poison" / "question_bucket_assignment.json"
ACCESS_MODEL = ARTIFACTS / "generated_access_model.json"
ACCESS_LIST = ARTIFACTS / "user_accessible_files.json"
METADATA = ARTIFACTS / "metadata.json"
FAISS_META = ARTIFACTS / "faiss_metadata.json"
_FAISS_ASCII = Path(os.environ.get("FAISS_ASCII_DIR", r"C:\temp\ref_rag_artifacts"))
FAISS_INDEX = _FAISS_ASCII / "faiss.index"

SEED = 42
PER_BUCKET = 50

spec = importlib.util.spec_from_file_location("s14", ROOT / "scripts" / "14_stamp_writer_levels.py")
s14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s14)


def _ensure_faiss_ascii():
    _FAISS_ASCII.mkdir(parents=True, exist_ok=True)
    src = ARTIFACTS / "faiss.index"
    if not FAISS_INDEX.exists() and src.exists():
        shutil.copy2(src, FAISS_INDEX)


def _sync_faiss_back():
    try:
        shutil.copy2(FAISS_INDEX, ARTIFACTS / "faiss.index")
        shutil.copy2(FAISS_META, _FAISS_ASCII / "faiss_metadata.json")
    except OSError as e:
        print("[WARN] could not sync faiss back to artifacts: %s" % e)


def strip_poison(vec_meta, access_meta):
    poison_names = {m["file_name"] for m in access_meta if m.get("is_poisoned")}
    clean_access = [m for m in access_meta if not m.get("is_poisoned")]
    clean_vec = [m for m in vec_meta if m.get("file_name") not in poison_names]
    print("[REPLACE] stripped %d poison docs; clean=%d" % (len(poison_names), len(clean_vec)))
    return clean_vec, clean_access, poison_names


def rebuild_index(vec_meta, model):
    texts = [m.get("content") or "" for m in vec_meta]
    print("[REPLACE] rebuilding FAISS for %d docs..." % len(texts))
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    return index


def inject(replace):
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bucket_info = man["shared_bucket"]
    provider = bucket_info["provider"]
    bucket = bucket_info["bucket"]

    vec_meta = json.loads(FAISS_META.read_text(encoding="utf-8"))
    access_meta = json.loads(METADATA.read_text(encoding="utf-8"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    old_poison_names = set()
    _ensure_faiss_ascii()

    if replace:
        vec_meta, access_meta, old_poison_names = strip_poison(vec_meta, access_meta)
        index = rebuild_index(vec_meta, model)
        faiss.write_index(index, str(FAISS_INDEX))
        _sync_faiss_back()
    else:
        index = faiss.read_index(str(FAISS_INDEX))

    existing_names = {m["file_name"] for m in access_meta}
    new_rows = []
    for doc in man["docs"]:
        fname = doc["file_name"]
        if fname in existing_names:
            print("[SKIP] already in index: %s" % fname)
            continue
        text = (DOCS_DIR / fname).read_text(encoding="utf-8")
        key = "%s:%s:%s" % (provider, bucket, fname)
        meta = {
            "file_name": fname,
            "global_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
            "provider": provider,
            "endpoint": "https://storage.googleapis.com",
            "parameters": {
                "iam": "google iam",
                "region": "asia-northeast3",
                "bucket": bucket,
            },
            "is_poisoned": True,
            "poison_id": doc["id"],
            "parent_id": doc.get("parent_id") or str(doc["id"]).rsplit("_v", 1)[0],
            "target_answer": doc["target_answer"],
            "target_question": doc["question"],
        }
        new_rows.append((text, meta))
        print("[ADD] %s" % fname)

    if new_rows:
        embs = model.encode([t for t, _ in new_rows], convert_to_numpy=True).astype(np.float32)
        index.add(embs)
        faiss.write_index(index, str(FAISS_INDEX))
        _sync_faiss_back()
        for text, meta in new_rows:
            vec_meta.append({
                "file_name": meta["file_name"],
                "global_uuid": meta["global_uuid"],
                "content": text,
            })
            access_meta.append(meta)
        FAISS_META.write_text(json.dumps(vec_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        METADATA.write_text(json.dumps(access_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[DONE] Injected %d poisoned docs into index" % len(new_rows))
    else:
        print("[INFO] nothing new to inject")
    return [m["file_name"] for _, m in new_rows], old_poison_names


def parent_key(meta):
    pid = meta.get("parent_id") or meta.get("poison_id") or meta.get("file_name") or ""
    return str(pid).rsplit("_v", 1)[0]


def assign_parents(parent_ids, ranked, seed=SEED, per_bucket=PER_BUCKET):
    rng = random.Random(seed)
    ids = sorted(set(parent_ids))
    rng.shuffle(ids)
    last = max(len(ranked) - 1, 0)
    return {pid: ranked[min(i // per_bucket, last)] for i, pid in enumerate(ids)}


def place():
    am = json.loads(ACCESS_MODEL.read_text(encoding="utf-8"))
    tiers = am["BUCKET_TIERS"]
    ranked = sorted(tiers, key=lambda b: int((tiers[b] or {}).get("tier_rank") or 99))
    levels = s14.gcp_user_levels(am)
    pool = s14.clean_pool(am)
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    poison_rows = [
        m for m in meta
        if m.get("is_poisoned") or "POISONED" in (m.get("file_name") or "")
    ]
    mapping = assign_parents([parent_key(m) for m in poison_rows], ranked)

    by_docs = Counter()
    by_q = defaultdict(set)
    for m in poison_rows:
        pid = parent_key(m)
        bucket = mapping[pid]
        m["parent_id"] = pid
        m["assigned_bucket"] = bucket
        m["provider"] = "gcp"
        params = dict(m.get("parameters") or {})
        params["bucket"] = bucket
        params.setdefault("iam", "google iam")
        params.setdefault("region", "asia-northeast3")
        m["parameters"] = params
        s14.stamp_meta_for_bucket(m, am, bucket, levels, pool)
        by_docs[bucket] += 1
        by_q[bucket].add(pid)

    METADATA.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for d in man.get("docs") or []:
            pid = d.get("parent_id") or str(d.get("id") or "").rsplit("_v", 1)[0]
            if pid in mapping:
                d["assigned_bucket"] = mapping[pid]
        MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")

    ASSIGNMENT.write_text(
        json.dumps(
            {"seed": SEED, "per_bucket": PER_BUCKET, "n_questions": len(mapping), "assignment": mapping},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    gcp_rbac = am["GCP_RBAC"]
    bucket_to_roles = defaultdict(list)
    for role, b in gcp_rbac.items():
        bucket_to_roles[b].append(role)
    poison_by_bucket = defaultdict(list)
    for m in poison_rows:
        b = (m.get("parameters") or {}).get("bucket")
        if b:
            poison_by_bucket[b].append(m["file_name"])
    all_poison = {fn for fns in poison_by_bucket.values() for fn in fns}
    access_map = json.loads(ACCESS_LIST.read_text(encoding="utf-8"))
    for user in am["USERS"]:
        name = user["name"]
        if user.get("full_access"):
            continue
        roles = set(user.get("gcp_roles") or [])
        entry = access_map.setdefault(name, {"files": []})
        files = {f for f in (entry.get("files") or []) if f not in all_poison}
        for b, fns in poison_by_bucket.items():
            if any(r in roles for r in bucket_to_roles.get(b, [])):
                files.update(fns)
        entry["files"] = sorted(files)
    ACCESS_LIST.write_text(json.dumps(access_map, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE] placement (disjoint Q, K variants together):")
    for b in ranked:
        print("  %s %s  questions=%d docs=%d" % (
            (tiers[b]["tier"] + "            ")[:12],
            b.split("-")[-1],
            len(by_q[b]),
            by_docs[b],
        ))


def upload():
    from google.cloud import storage

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mapping = {}
    if ASSIGNMENT.exists():
        mapping = (json.loads(ASSIGNMENT.read_text(encoding="utf-8")) or {}).get("assignment") or {}

    by_bucket = defaultdict(list)
    missing = []
    for doc in man["docs"]:
        fname = doc["file_name"]
        if not (DOCS_DIR / fname).exists():
            raise FileNotFoundError(DOCS_DIR / fname)
        bucket = doc.get("assigned_bucket") or mapping.get(
            doc.get("parent_id") or str(doc.get("id") or "").rsplit("_v", 1)[0]
        )
        if not bucket:
            missing.append(fname)
            continue
        by_bucket[bucket].append(fname)
    if missing:
        raise SystemExit("[ERROR] %d docs have no assigned_bucket" % len(missing))

    client = storage.Client.from_service_account_json(os.environ["GCP_ADMIN_KEY_PATH"])
    workers = int(os.environ.get("UPLOAD_WORKERS", "16"))
    for bucket_name in sorted(by_bucket):
        existing = {blob.name for blob in client.list_blobs(bucket_name)}
        pending = [fn for fn in by_bucket[bucket_name] if fn not in existing]
        print("[INFO] %s assigned=%d pending=%d" % (bucket_name, len(by_bucket[bucket_name]), len(pending)), flush=True)
        ok = err = done = 0
        if pending:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(client.bucket(bucket_name).blob(fn).upload_from_filename, str(DOCS_DIR / fn)): fn
                    for fn in pending
                }
                for fut in as_completed(futs):
                    done += 1
                    try:
                        fut.result()
                        ok += 1
                    except Exception as e:
                        err += 1
                        print("[ERROR] %s/%s: %s" % (bucket_name, futs[fut], e), flush=True)
                    if done % 50 == 0 or done == len(pending):
                        print("[PROGRESS] %s %d/%d uploaded=%d err=%d" % (
                            bucket_name, done, len(pending), ok, err
                        ), flush=True)
        print("[DONE] %s uploaded=%d skipped=%d err=%d" % (
            bucket_name, ok, len(by_bucket[bucket_name]) - len(pending), err
        ), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true", help="Strip old poison and rebuild FAISS first")
    parser.add_argument("--place-only", action="store_true", help="Skip inject; re-assign buckets and upload")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload to GCS")
    args = parser.parse_args()

    if not args.place_only:
        inject(args.replace)
    place()
    s14.main()
    if not args.no_upload:
        upload()


if __name__ == "__main__":
    main()
