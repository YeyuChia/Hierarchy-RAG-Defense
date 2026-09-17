# Ladder access model — main eval (k=5, TEST)

- Users: 32; victim=user-31; attacker=user-2
- Main: 50 disjoint questions per bucket; all K variants in that bucket
- VAL gap rule: separable 30/30; selected δ0=0.35 k=0.08 gap=0.2543 thr=0.1456
- VAL clean chunks above L: 61.6%

| setting | writer | thr | Poison@k | clean_keep | True ASR |
|---------|--------|-----|----------|------------|----------|
| ACL (main: low+mid) | - | - | 0.603 | 1.000 | 63% |
| Trust + writer (main) | True | 0.1456 | 0.363 | 0.605 | 44% |
| Trust no-writer (ablation) | False | 0.1456 | 0.363 | 0.605 | 44% |
| ACL (appendix: high) | - | - | 0.603 | 1.000 | 63% |
| Trust + writer (appendix: high) | True | 0.1456 | 0.603 | 0.610 | 70% |
