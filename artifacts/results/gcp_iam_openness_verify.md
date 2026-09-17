# GCP IAM openness verification

- project: `permission-aware-rag-505023`
- n_users: 32
- querying_user: `user-31`
- all_match: **True**

| tier | bucket | JSON readers | live readers | JSON L_doc | live L_doc | match |
|------|--------|--------------|--------------|------------|------------|-------|
| public | `whsdeuye-gcp-bucket-1` | 31/32 | 31/32 | 0.031 | 0.031 | True |
| team | `whsdeuye-gcp-bucket-2` | 19/32 | 19/32 | 0.406 | 0.406 | True |
| dept | `whsdeuye-gcp-bucket-3` | 11/32 | 11/32 | 0.656 | 0.656 | True |
| restricted | `whsdeuye-gcp-bucket-4` | 6/32 | 6/32 | 0.812 | 0.812 | True |
| private | `whsdeuye-gcp-bucket-5` | 3/32 | 3/32 | 0.906 | 0.906 | True |
