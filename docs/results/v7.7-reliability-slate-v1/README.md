# V7.7: measured development rejection

**No V7.7 candidate was promoted or exported.** All four preregistered variants
improved overall development AUC against both rebuilt anchors in both windows,
but every variant failed the required 48–72-hour comparison against rebuilt V6.
The campaign stopped at its frozen development rule. Primary, secondary,
five-day, historical floor checks and competition exports were not executed.

The [protocol](../../../experiments/v7.7-reliability-slate-v1/RUN_V7_7.md) fixes
the hypotheses, populations, folds, features, weighting, optimizer, selection,
twelve release checks and stopping rule. [Exact metrics](results.json),
[settled accounting](accounting.json) and [validation](validation.json) bind this
run to scientific source `68d7c9c5f7c44ab4edca5de674d02dd4f865ba07`.
The four source CI jobs passed before the private campaign in
[run 40787257509](https://github.com/EdwinShibuMathew/techarena/actions/runs/40787257509).

All values below are candidate minus comparator global row ROC AUC. The aged
development subset contains 4,065 first-user events from the Jan22–24 window.

| Variant | Jan22–24 vs V6 | Jan22–24 vs V7 | Jan24 vs V6 | Jan24 vs V7 | Aged vs V6 | Aged vs V7 |
|---|---:|---:|---:|---:|---:|---:|
| A: control | +0.00043060 | +0.00016257 | +0.00027794 | +0.00009511 | **−0.00033505** | −0.00002884 |
| B: reliability | +0.00045865 | +0.00019062 | +0.00032345 | +0.00014061 | **−0.00029840** | +0.00000780 |
| C: slate | +0.00037909 | +0.00011107 | +0.00034895 | +0.00016612 | **−0.00022404** | +0.00008216 |
| D: both | +0.00040936 | +0.00014134 | +0.00039609 | +0.00021326 | **−0.00018610** | +0.00012011 |

Reliability and slate evidence reduced the aged deficit relative to the control.
They did not recover the full difference between the fixed V7 anchor and V6.
For this cohort, V7 trails V6 by 0.00030620; D recovers 0.00012011, leaving the
0.00018610 deficit. This supports reconsidering the fixed anchor in a separately
declared future design. It does not justify choosing D, tuning this closed
campaign, relaxing the gate, or claiming improvement on unmeasured windows.
All dates are retrospectively reused; these are not independent holdout results.

The private preflight verified the dataset, runtime, CUDA, all pinned foundation
and protected assets, source snapshot and historical threshold source hashes.
Fifteen teacher jobs fitted 75 user-disjoint OOF CatBoost models. Eight bounded
heads converged. Ten original deployment combiners were reused after identity
checks. No neural model was retrained.

| Metadata day | Unique users/events | Fitting views | Effective original events |
|---|---:|---:|---:|
| Jan21 | 35,156 | 35,156 | 34,468.78 |
| Jan22 | 35,112 | 43,955 | 19,177.58 |

Each nonempty lag-day/six-hour stratum had equal weight, with every original
event's replicated total capped at one. All four candidates used identical
teacher predictions, normalization inputs, fitting weights and fixed negatives.

The campaign completed 40 accounted jobs and charged **858.8667 seconds**. The
original 43,200-second ledger now records **10,402.6125 seconds** across 363
attempts, none running; **32,797.3875 seconds remain**. No ledger was reset.
Re-running preflight, development, screen, stress, finalize, replay and run after
rejection was refused without changing the ledger or creating a new attempt.

Engineering validation passed 308 full-suite tests with one optional historical
artifact skip, plus 195 mandatory contracts with no skips. Tests cover real tiny
OOF/head fitting and CSV replay, twelve independent gate failures, source and
comparator evidence, fresh export retry after accounting failure, adoption of a
settled attempt with its original owner, frozen verification descriptor reuse,
and publication ordering. These synthetic release checks do not constitute a
private successful V7.7 export. The generic original V6 bundle check still lacks
its historical manifest; this run uses the explicitly rebuilt lineage.

The existing competition fallback remains
`artifacts/submissions/rebuilt_v6_submission.csv`, SHA256
`9320ca6465c5ce6576746c5fc8f00d13109e507d7a20d5d506a9e03f63fa2b85`.
Its bytes were included in the protected preflight inventory. It is the rebuilt
baseline, not V7.7 or an exact reproduction of the lost canonical V6 bundle.
No V7.7 CSV, configuration, readiness marker or competition upload was created.
