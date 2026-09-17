# scripts/

编号按实验顺序。当前主入口是 **`18_cloud_ladder_main_k5.py`**。不要把 `17` 当主实验跑。

PowerShell 每次先：

```powershell
conda activate permission-aware-rag
$env:PYTHONPATH = (Get-Location).Path
```

## A. 建 GCP 梯子环境

| 脚本 | 作用 |
|------|------|
| `01_generate_access_model.py` | 32 用户、5 层嵌套桶（public→private） |
| `02_gcp_aws_resource_creator.py` | 真建桶 / service account（`SKIP_AWS=1` 只做 GCP） |
| `03_reconcile_and_verify_gcp_iam.py` | 把 live IAM 对齐 access model，核对 L_doc |
| `04_prepare_documents.py` | HotpotQA → 文档 |
| `05_upload_resources.py` | 上传到云存储 |
| `06_generate_user_accessible_file_list.py` | 用户–文档 ACL 真值 |
| `07_generate_question_list.py` | 评测问题 |
| `08_run_ingestion_manager.py` | 嵌入 + FAISS |

## B. 原论文评测（可选，干净语料）

| 脚本 | 作用 |
|------|------|
| `09_run_permission_smoke.py` | 检索 + 权限过滤冒烟 |
| `10_run_quantitative_evaluation.py` | HotpotQA EM/F1 |
| `11_run_latency_evaluation.py` | IAM 检查延迟 |

## C. PoisonedRAG + Trust（当前实验）

| 脚本 | 作用 |
|------|------|
| `12_craft_poisonedrag_docs.py` | 生成毒文档 \(P=Q\oplus I\) |
| `13_inject_poison_into_index.py` | 注入 FAISS；每桶 50 题；上传该桶 |
| `14_stamp_writer_levels.py` | 按写权限代理打 `written_by` / `writer_level`（`13` 里也会调） |
| `15_report_distributions.py` | 各桶干净/毒/作者分布 |
| `16_trust_ladder_audit.py` | 不调 GCP，扫 \((\delta_0,k)\) 看 T 梯子 |
| `17_cloud_param_select_and_sim.py` | **工具库**，给 `18` import；`__main__` 是旧 GTR 协议 |
| **`18_cloud_ladder_main_k5.py`** | **主评测**：VAL 选 \(\tau\)，TEST Poison@k / clean_keep / ASR |
| `19_plot_formula_compare.py` | 换公式对比图 |
| `20_plot_why_thirty_same.py` | 30 格同一 keep-set 图 |

## 结果

- 数字：`artifacts/results/cloud_ladder_*.md`、`formula_compare.md`
- 图：`artifacts/results/figures/`
- 审计：`trust_ladder_audit.md`、`gcp_iam_openness_verify.md`

Threat：victim `user-31`，attacker `user-2`，k=5。乘积公式下 30 组 \((\delta_0,k)\) 评测一样；加不加 \(T_{writer}\) 也一样。
