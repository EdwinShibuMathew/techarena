# V7.5 rejection audit and V7.6 design

This is a post-hoc investigation of a closed experiment. It does not fit a
model, rerun CUDA inference, export a new submission, reopen V7.5, or authorize
V7.6 training. Its comparator is the reconstructed V6/V7 recipe, whose outputs
are distinct from the lost historical artifacts. All reused dates remain
retrospective evidence.

Run with the original reconstruction environment:

```bash
/home/edwin/miniconda3/envs/techarena/bin/python experiments/v7.5-audit/audit.py run
/home/edwin/miniconda3/envs/techarena/bin/python experiments/v7.5-audit/audit.py verify
```

The controller freezes clean committed source, acquires the original shared
43,200-second ledger, reserves at most 1,200 seconds, and protects a 3,600-second
finalization reserve. A worker requires the current parent lease and exact
runtime. The controller publishes only after durable final accounting. Prior
charges and historical seals remain intact. Successful reruns verify and reuse
the same report without reserving more compute. A failed attempt remains charged;
any technical retry must preserve its evidence and cannot change the scientific
analysis or the closed outcome.

Inputs are bounded explicit task inventories, source archives, causal data
metadata, checkpoint hashes, combiner receipts, saved features/predictions, and
the existing CSV sidecars. Hashes are checked before and after analysis. The live
ledger is deliberately excluded from protected inputs because the audit appends
its own charge; an accounting prefix hash proves old attempts were preserved.

The analysis reproduces all six development comparisons and saved combiner
predictions, decomposes global AUC by negative-event type, examines within-event
ranking, row loss, Brier score, event score discrimination, subgroups, fixed-weight
member swaps, feature ranges, and PredictionValuesChange importance. A fixed
1,000-draw, seed-2026 whole-user bootstrap for A versus V7 measures uncertainty
only. Component/member swaps are diagnostic scores and cannot be selected or
exported. A sum of row scores is not a Bernoulli event probability.

`successor.py` freezes a bounded V7-anchored residual design, including causal
historical forecasts, inner user grouping, three divergence penalties, stopping
rules, both comparators, and all eight original gates. It contains no fitting
entry point. Failure of the inner selection stops the proposed future experiment;
outer results cannot trigger hyperparameter changes or a runner-up trial.

Required source checks include `tests/test_v75_audit.py` in the mandatory CPU
model-contract checker. Evidence-chain tests substitute a small corpus only;
their successful promotion decision is synthetic engineering validation, never
production promotion evidence.
