# Permission-Aware RAG

Fork of the official implementation for  
**Permission-Aware RAG: IAM-Based Access Filtering in Multi-Resource Environments**,  
plus this fork's extension: **hierarchy Trust vs PoisonedRAG on real GCP**.

Script numbers follow the experiment order. See [`scripts/README.md`](scripts/README.md).

---

## Current experiment (GCP ladder)

- **32 users**, 5 nested GCP buckets (public → team → dept → restricted → private)
- Victim **user-31**, attacker **user-2**
- 50 disjoint poison questions per bucket; k = 5
- \(T = T_{doc} \times T_{hier}[\times T_{writer}]\), \(T_{hier}=1-\sigma((\mathrm{gap}-\delta_0)/k)\)
- \(L_{doc}=1-\mathrm{readers}/32\), \(L_{user}=\max L_{doc}\) the querier can read (user-31 ≈ 0.906)

**Finding so far:** on the 30-cell \((\delta_0,k)\) grid, product and logistic keep-sets are identical (VAL Poison@k 0.368, clean_keep 0.615), with or without \(T_{writer}\). Parameters are not identifiable; the filter is a ladder cut (drop public+team, keep dept+).

Main entry:

```powershell
conda activate permission-aware-rag
$env:PYTHONPATH = (Get-Location).Path
python scripts/18_cloud_ladder_main_k5.py
```

Do **not** run `17` as the main experiment — it is a helper module imported by `18`.

Results: `artifacts/results/` (tables) and `artifacts/results/figures/` (plots). One-page summary: `artifacts/results/formula_compare.md`.

---

## Repository layout

- `scripts/` — numbered pipeline (01–20)
- `src/` — `PermissionRetriever`, IAM adapters, `TrustFilter`
- `artifacts/` — access model, FAISS, poison, results (do not commit keys)
- `docs/` — GCP setup, Trust formulas, PoisonedRAG crafting

---

## Dataset

HotpotQA distractor split. Place it at `artifacts/hotpot_dev_distractor_v1.json`:

```bash
curl -Lo artifacts/hotpot_dev_distractor_v1.json https://hotpotqa.s3.amazonaws.com/hotpot_dev_distractor_v1.json
```

## Setup

Python 3.10+, Conda, a billed GCP project. AWS is optional (`SKIP_AWS=1` for the Trust path).

```bash
conda create -n permission-aware-rag python=3.10.18
conda activate permission-aware-rag
pip install -r requirements.txt
```

GCP: service account with Owner / Storage Admin; enable IAM API; download JSON key. Details: [`docs/gcp_setup.md`](docs/gcp_setup.md).

`.env` (never commit):

```env
OPENAI_API_KEY="sk-..."
GCP_PROJECT_ID="your-gcp-project-id"
GCP_ADMIN_KEY_PATH="./artifacts/gcp-admin-key.json"
GCP_BUCKET_LOCATION="asia-northeast3"
SKIP_AWS=1
```

---

## Pipeline

### A. Provision + index (once)

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/01_generate_access_model.py
python scripts/02_gcp_aws_resource_creator.py
python scripts/03_reconcile_and_verify_gcp_iam.py
python scripts/04_prepare_documents.py
python scripts/05_upload_resources.py
python scripts/06_generate_user_accessible_file_list.py
python scripts/07_generate_question_list.py
python scripts/08_run_ingestion_manager.py
```

### B. Original paper eval (optional, clean corpus)

```powershell
python scripts/09_run_permission_smoke.py
python scripts/10_run_quantitative_evaluation.py
python scripts/11_run_latency_evaluation.py
```

### C. Poison + Trust (current)

```powershell
python scripts/12_craft_poisonedrag_docs.py
python scripts/13_inject_poison_into_index.py --replace
python scripts/14_stamp_writer_levels.py
python scripts/15_report_distributions.py
python scripts/16_trust_ladder_audit.py
python scripts/18_cloud_ladder_main_k5.py
python scripts/19_plot_formula_compare.py
python scripts/20_plot_why_thirty_same.py
```

Trust implementation: `src/retrievers/trust_filter.py`.  
Formulas: [`docs/hierarchy_trust.md`](docs/hierarchy_trust.md).  
Poison craft: [`docs/poisonedrag_method.md`](docs/poisonedrag_method.md).

---

## License

Apache License 2.0. See [LICENSE-2.0.txt](./LICENSE-2.0.txt).

HotpotQA: https://github.com/hotpotqa/hotpot
