> **Historical archive:** [PR #4](https://github.com/EdwinShibuMathew/techarena/pull/4), [reviewed source `3a90217`](https://github.com/EdwinShibuMathew/techarena/commit/3a902173cf111dcb748140a31293425ae66b2142).
> This recipe is closed and not submission-ready. Its historical execution tooling has confirmed review defects and is unsupported.
> Commands and validation claims below describe that revision. See the [archive and review dispositions](../v8-archive/README.md) and use the [shared workflow](../../ML_WORKFLOW.md) for new work.

# V8 robust ensemble: A selected, rejected at primary gate

The revised development strategy selected composition A, updating dim16
seeds42/43 while retaining V6's original seed44. Primary promotion did not pass.
V6 remains the submitted benchmark, PR #4 remains draft, and there is no V8
ready configuration or submission CSV.

## Failure-aware development

The [frozen protocol](https://github.com/EdwinShibuMathew/techarena/blob/3a902173cf111dcb748140a31293425ae66b2142/experiments/v8-robust-ensemble/RUN_V8_ROBUST.md)
addresses three observed problems: weak transfer into the dominant early
five-day population, regression against no-click-event negatives, and previous
selection of a component by its average gain. It tests two fixed compositions:
A updates both dim16 members; B also uses the previously trained seed44 update.
No member weights, features or architecture were tuned.

Development uses only first-user Jan21 events with pre-Jan21 neural features.
Fold0 fits all five reference combiners before 08:00 UTC and validates 08:00–16:00;
fold1 fits before 16:00 and validates 16:00–24:00. Reference combiners are fitted
only on each fold's fitting block, not on the full day. The reference is a fresh
rolling V6-recipe ensemble, distinct from the cached promotion benchmark.

| Composition | Fold0 ensemble gain | Fold1 ensemble gain | Minimum |
|---|---:|---:|---:|
| A: dim16 pair | +0.000102899108 | +0.000123615320 | +0.000102899108 |
| B: dim16 pair + seed44 | +0.000094229285 | +0.000159499262 | +0.000094229285 |

A's no-click-negative contributions were +0.000051478618 and +0.000059383510;
B's were +0.000035206263 and +0.000067117089. Both qualified. A had the stronger
minimum fold even though B had the larger average. The selection receipt was
sealed before the new promotion screen. No promotion labels selected the
composition and no fallback was evaluated after failure.

Warm/cold, history, day and exact cross-group positive/negative-pair diagnostics
are included in the results. Group AUCs are descriptive and are not averaged to
reconstruct global AUC or used for routing. Exact pair contributions reconstruct
global change within 1e-12. All dates remain reused development evidence, not
independent holdout or selection-adjusted guarantees.

## Promotion outcome

| Window | V6 AUC | Selected A AUC | Change | Paired 95% interval |
|---|---:|---:|---:|---|
| Primary Jan25–26 | 0.818459253326 | 0.818470061493 | +0.000010808166 | [−0.000051946950, +0.000075899505] |
| Secondary Jan26 | 0.827127712250 | 0.827197755145 | +0.000070042895 | [−0.000004834098, +0.000145198289] |

The primary interval lower bound must be positive; it failed. The secondary
gate uses nonnegative point change and passed. All 1,000 paired draws were
valid in each cohort. These results show that the Jan21 development gains did
not produce a statistically supported primary improvement. They do not prove
which modeling mechanism caused the limited transfer.

## Integrity, compute and release limits

- [Preparation, reviewed assets and zero-error original-member replays](preparation.json)
- [Development results, diagnostics and complete artifact fingerprints](development.json)
- [Frozen composition receipt](selected.json)
- [Primary and secondary metrics and diagnostics](screen.json)
- [Stopping decision, training summary and compute accounting](decision.json)

Eight new dim16 prefix models completed eight event epochs from their original
epoch 196 preferences. The original mixed loss, fresh Adam, sampling seeds 3042/
3043, regularization, 37 features, combiner settings and member weights remain
fixed. Preferences were not retrained. Seed44's old epoch 260-derived prefix
outputs were verified for composition B; its closed standalone result was not
changed and its update was not used in selected A.

Both original dim16 members replayed with zero maximum error on primary,
secondary and five-day cohorts. All five full-data V6 members also replayed
with zero error and matched the canonical CSV fingerprint. Historical archives,
all prior trials, legacy sources and V6 configuration remain unchanged. Older
missing metadata stays absent, reconciled through reviewed hashes and original
reports; no metadata was invented.

Actual stress and finalize commands refused at the failed primary gate before
launching any new training. No new five-day comparison, full-data model, or
competition CSV was generated. The budget carries 748.931401111 prior seconds;
exact cumulative usage and stage-refusal verification are in the decision.
The original 14,400-second limit and process-group timeout handling remain.

`tools/predict_v8_candidate.py` verifies selection, complete inventories, all
promotion gates, saved member models and reused Jan25 combiners before release.
The successful multi-member release flow is covered synthetically, with real
neural and combiner save/reload tests and existing full-size CSV checks. A real
final V8 replay remains unrun because promotion failed. Atomic no-replace export
and publishing readiness metadata last are retained. Source CI does not promote
a model. All weights, cached arrays and predictions remain private and ignored.
