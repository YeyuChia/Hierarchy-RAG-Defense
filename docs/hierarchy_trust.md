# Hierarchy-aware trust

Filter retrieved docs using the **GCP access ladder**, after ACL. Not query–doc similarity.

## Pipeline

```text
retrieve top-k → ACL (may the user read?) → Trust (should they trust it?) → LLM
```

## Levels

From `artifacts/generated_access_model.json` (32 users, GCP-only when `SKIP_AWS=1`):

| Symbol | Meaning |
|--------|---------|
| `L_doc` | `1 - readers/\|USERS\|` for the chunk's bucket |
| `L_user` | `max L_doc` among buckets the querier can read (user-31 ≈ 0.906) |
| `L_writer` | same definition for `written_by` |
| `gap` | `max(0, L_user - L_doc)` |

Bucket ladder (readers / 32): public 0.031, team 0.406, dept 0.656, restricted 0.812, private 0.906.

## Score (current main)

\[
T_{\mathrm{hier}} = 1 - \sigma\bigl((\mathrm{gap}-\delta_0)/k\bigr)
\qquad
T = T_{\mathrm{doc}} \times T_{\mathrm{hier}} [\times T_{\mathrm{writer}}]
\]

\(T_{\mathrm{doc}}=L_{\mathrm{doc}}\). \(T_{\mathrm{writer}}=L_{\mathrm{writer}}\) when writer is on (`writer_mode=level`); omitted (=1) when off.

Keep if \(T \ge \tau\). VAL: \(\tau=(L+R)/2\) with \(L=\max T\) of open-tier poison (public+team) and \(R=\) median \(T_{\mathrm{clean}}\).

Code: `src/retrievers/trust_filter.py` (`combine_mode="mul"`, `hier_mode="sigmoid"`).

## What the grid does here

On the 30-cell \((\delta_0,k)\) grid, \(\tau\) sits in the empty band between team and dept. Product and logistic therefore drop buckets 1–2 and keep 3–5 for every cell. Adding \(T_{\mathrm{writer}}\) does not move team above \(\tau\). Logistic-max has a second, deeper cut; max-gap still picks the 0.368 / 0.615 operating point.

Treat this as a **discrete ladder cut**, not a learned sigmoid over \((\delta_0,k)\).
