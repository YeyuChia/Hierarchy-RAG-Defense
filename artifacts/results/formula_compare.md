# Formula swap — one-page comparison

Victim user-31, k=5. VAL = 125 poison Q + 40 clean Q. TEST = 125 poison Q + 40 clean Q.

Figures: `figures/formula_compare.png`, `figures/why_thirty_configs_same.png` (product grid).

## VAL (same questions, different T)

| formula | writer | 30-grid keep-set | selected Poison@k | selected clean_keep |
|---|---|---|---:|---:|
| ACL only | — | — | 0.611 | 1.000 |
| \(T=T_{doc}\times T_{hier}\) | off | 30/30 identical | 0.368 | 0.615 |
| \(T=\sigma(0.3T_{doc}+0.2T_{writer}+0.5T_{hier})\) | on | 30/30 identical | 0.368 | 0.615 |
| \(T=\sigma(0.5\max(T_{doc},T_{writer})+0.5T_{hier})\) | on | **two** sets | 0.368 (max-gap pick) | 0.615 |
| logistic-max, other cells | on | 22/30 cells | 0.304 | 0.530 |

Selected \((\delta_0,k)\) is always 0.35 / 0.08 under max-gap. Mid \(\tau\) vs \(L+\varepsilon\) does not change product or logistic keep-sets.

## TEST (product only)

| setting | writer | Poison@k | clean_keep | True ASR |
|---|---|---:|---:|---:|
| ACL, poison in assigned buckets | — | 0.603 | 1.000 | 63% |
| product Trust | on | 0.363 | 0.605 | 44% |
| product Trust | off | 0.363 | 0.605 | 44% |
| ACL, poison remapped to private | — | 0.603 | 1.000 | 63% |
| product Trust, poison remapped to private | on | 0.603 | 0.610 | 70% |

## Raw grids

- `cloud_ladder_val_nowriter.md`
- `cloud_ladder_val_logistic.md`
- `cloud_ladder_val_logistic_max.md`
- `cloud_ladder_main_k5.md`
