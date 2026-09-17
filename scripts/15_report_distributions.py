"""Report current doc + writer distribution per bucket (metadata view + GCP blob view).

Writer/user levels use the same definition as TrustFilter: L = max L_doc
among buckets the person can read (not roles/5).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)
os.environ["SKIP_AWS"] = "1"

from src.retrievers.trust_filter import compute_bucket_openness, compute_user_levels

am = json.loads((ROOT / "artifacts" / "generated_access_model.json").read_text(encoding="utf-8"))
md = json.loads((ROOT / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
tiers = am["BUCKET_TIERS"]
lv = compute_user_levels(am, provider_scope="gcp")
openness = compute_bucket_openness(am, provider_scope="gcp")

order = sorted(tiers, key=lambda b: tiers[b]["tier_rank"])

clean = Counter()
poison = Counter()
writers = {}
for m in md:
    b = (m.get("parameters") or {}).get("bucket")
    if m.get("is_poisoned"):
        poison[b] += 1
    else:
        clean[b] += 1
        writers.setdefault(b, Counter())[m.get("written_by")] += 1

print("=== metadata view (what the index / TrustFilter sees) ===")
print(f"{'tier':<12}{'bucket':<8}{'L_doc':>7}{'clean':>8}{'poison':>8}")
for b in order:
    ldoc = 1.0 - float(openness.get(("gcp", b), 0.5))
    print(f"{tiers[b]['tier']:<12}{b.split('-')[-1]:<8}{ldoc:>7.3f}{clean[b]:>8}{poison[b]:>8}")
print(f"{'TOTAL':<12}{'':<8}{'':>7}{sum(clean.values()):>8}{sum(poison.values()):>8}")

print("\n=== clean writers per bucket (share, L_writer = max L_doc) ===")
for b in order:
    tot = sum(writers[b].values())
    dist = "  ".join(
        f"{w}(L={lv.get(w, 0):.3f}) {100 * c / tot:.0f}%"
        for w, c in sorted(writers[b].items(), key=lambda x: -x[1])
    )
    mean = sum(lv.get(w, 0) * c for w, c in writers[b].items()) / tot
    print(f"{tiers[b]['tier']:<12} n={tot:<6} meanL={mean:.3f}   {dist}")

print("\n=== poison writers ===")
pw = Counter((m.get("written_by"), (m.get("parameters") or {}).get("bucket")) for m in md if m.get("is_poisoned"))
for (w, b), c in pw.items():
    print(f"  {w} (L={lv.get(w, 0):.3f}) in {b.split('-')[-1]}: {c}")

# physical blobs
try:
    from google.cloud import storage

    client = storage.Client.from_service_account_json(os.environ["GCP_ADMIN_KEY_PATH"])
    print("\n=== physical GCP blobs ===")
    for b in order:
        bucket = client.bucket(b)
        n_poison = sum(1 for _ in client.list_blobs(bucket, prefix="hp_", max_results=100000)
                       if _.name.endswith("_POISONED.txt"))
        total = sum(1 for _ in client.list_blobs(bucket, max_results=100000))
        print(f"{tiers[b]['tier']:<12}{b.split('-')[-1]:<8} total_blobs={total:<7} poison_blobs={n_poison}")
except Exception as exc:  # noqa: BLE001
    print(f"\n[skip GCP listing] {exc}")
