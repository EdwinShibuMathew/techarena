# V7 validation result

**Decision: not promoted.** The fixed recipe reproduces the previously reported
overall gains, but fails the five-day late-window nonnegative-change requirement.

| Cohort | Events | Global row AUC change vs V6 | Paired 95% interval |
|---|---:|---:|---|
| Primary Jan25–26 | 43,964 | +0.000242447044 | [+0.000150939096, +0.000335443808] |
| Secondary Jan26 | 35,138 | +0.000276540602 | [+0.000176846089, +0.000369603477] |
| Five-day Jan22–26 | 51,488 | +0.000267002885 | [+0.000155514621, +0.000369926261] |
| Five-day late 48–120h | 7,596 | **−0.000167595880** | Point comparison only |

[validation.json](validation.json) records the complete fixed protocol, source
and input hashes, deterministic whole-user bootstrap results, all four gate
decisions, and fresh CSV fingerprint verification. The original V6 CSV and
the generated V7 candidate match their existing configured SHA256 fingerprints.
Maximum event-sum error is 4.44e-16; within-event orders and ties are preserved.

This uses reused development dates, not an independent holdout. No model was
fitted, no parameters were tuned, and no competition upload was performed.
See [the reproduction commands](../../../experiments/v7-validation/RUN_V7.md).

**Note:** The original V7 candidate was said to be not promoted but when it was ran against huawei validation system, it was promoted in rank. 
