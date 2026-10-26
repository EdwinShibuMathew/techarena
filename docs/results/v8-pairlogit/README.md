> **Historical archive:** [PR #4](https://github.com/EdwinShibuMathew/techarena/pull/4), [reviewed source `3a90217`](https://github.com/EdwinShibuMathew/techarena/commit/3a902173cf111dcb748140a31293425ae66b2142).
> This recipe is closed and not submission-ready. Its historical execution tooling has confirmed review defects and is unsupported.
> Commands and validation claims below describe that revision. See the [archive and review dispositions](../v8-archive/README.md) and use the [shared workflow](../../ML_WORKFLOW.md) for new work.

# V8 PairLogit screen: not promoted

**The fixed recipe fails the primary confidence gate. V6 remains the benchmark;
no submission-ready V8 CSV or configuration was created.**

| Cohort | Events | AUC change vs V6 | Paired 95% interval | Required gate |
|---|---:|---:|---|---|
| Primary Jan25–26 | 43,964 | +0.000080285486 | [−0.000011608261, +0.000171144544] | **Failed:** lower bound is not positive |
| Secondary Jan26 | 35,138 | +0.000062360499 | [−0.000040755848, +0.000159492423] | Passed: point change is nonnegative |

The five-day stress test, final fit, and submission export were not run. The
positive point estimates do not justify promotion under the predeclared rule.
No parameters, seeds, weights, pair counts, links, or cohorts were swept.

The trial replaced only the dim8 seed42 combiner at weight 2/9. All five neural
models, 37 original features and four other combiners stayed fixed. Each CPU
ranker used 545 trees, depth 6, learning rate .05, L2 20, seed 42 and four threads.
Sixteen global negatives were sampled per positive with seed 3042, including
no-click events. The sigmoid slope was one; the intercept used fitting labels
only. Validation used 1,000 paired whole-user bootstrap draws with seed 2026.

Before fitting, both historical seed42 comparators reproduced exactly. All
five full-data V6 members and the canonical V6 CSV also reproduced exactly.
Preparation verified all 29 collected assets and 59 historical Python sources.
The source-manifest JSON needed an explicit LF/CRLF metadata adapter because
Git normalizes checkout line endings. This preserved all original source/model
hashes and the numerical replay tolerance. The initial pre-fit metadata failure
was preserved and charged to the compute budget; no ranker had been fitted.

Evidence:

- [Frozen experiment and commands](https://github.com/EdwinShibuMathew/techarena/blob/3a902173cf111dcb748140a31293425ae66b2142/experiments/v8-pairlogit/RUN_V8.md)
- [Preparation, checkpoint identities and baseline replay](preparation.json)
- [Screen, calibration, sampling and AUC pair decompositions](screen.json)
- [Descriptive cold-user, history-length and per-day diagnostics](diagnostics.json)
- [Decision and compute accounting](decision.json)

Diagnostics did not select a model, route users, or change promotion gates.
Dates are reused development data, not independent holdout evidence; intervals
are not adjusted for previous experimentation. No competition upload occurred.
