# PoisonedRAG-style poison crafting

Source: *PoisonedRAG* (Zou et al.). We use the black-box construction on **real GCP IAM**.

## Goal

Inject a few texts \(P\) so that for attacker-chosen question \(Q\), RAG answers attacker-chosen \(R\).

| Condition | Meaning |
|-----------|---------|
| Retrieval | \(P\) in top-\(k\) for \(Q\) |
| Generation | with \(P\) in context, the LLM says \(R\) |

Black-box: \(S=Q\), \(P=Q\oplus I\). \(I\) is crafted so that \(Q\) + \(I\) yields \(R\).

## Cross-permission setting (this repo)

- 5 nested GCP buckets; attacker **user-2** (public only) vs victim **user-31** (all five)
- Script 13 assigns **50 disjoint questions per bucket**; all K variants stay in that bucket
- ACL stays on: poison is **in-scope** (the victim is allowed to read it)
- Writer stamp (script 14): `written_by` sampled among users who can access that bucket (read role as write proxy)

## Files

| Path | Role |
|------|------|
| `artifacts/poison/targets.json` | \((Q,R,I)\) |
| `artifacts/poison/docs/` | crafted \(P\) texts |
| `artifacts/poison/question_bucket_assignment.json` | 50 Q / bucket |
| `artifacts/poison/_archive/` | old expand snapshots |
| `scripts/12_craft_poisonedrag_docs.py` | build \(P=Q\oplus I\) |
| `scripts/13_inject_poison_into_index.py` | FAISS + metadata + upload |
| `scripts/14_stamp_writer_levels.py` | `written_by` / `writer_level` |
| `scripts/18_cloud_ladder_main_k5.py` | VAL/TEST Trust eval |
