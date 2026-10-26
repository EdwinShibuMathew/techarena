> **Historical archive:** [PR #4](https://github.com/EdwinShibuMathew/techarena/pull/4), [reviewed source `3a90217`](https://github.com/EdwinShibuMathew/techarena/commit/3a902173cf111dcb748140a31293425ae66b2142).
> This recipe is closed and not submission-ready. Its historical execution tooling has confirmed review defects and is unsupported.
> Commands and validation claims below describe that revision. See the [archive and review dispositions](../v8-archive/README.md) and use the [shared workflow](../../ML_WORKFLOW.md) for new work.

# V8 residual event: B selected, rejected at primary gate

The broader, bounded comparison completed. B improved at all three development
origins, but lost AUC on the primary promotion window and failed its confidence
gate. This fixed recipe is closed. V6 remains the submitted benchmark, PR #4
remains draft, and no submission-ready V8 configuration or CSV was produced.

## Fixed modeling change

The [frozen protocol](https://github.com/EdwinShibuMathew/techarena/blob/3a902173cf111dcb748140a31293425ae66b2142/experiments/v8-residual-event/RUN_V8_RESIDUAL.md)
adds bounded event-wide and candidate-choice corrections to original V6 event
scores. Every original parameter, including preference embeddings, is frozen.
Both heads use 32 SiLU units and zero-initialized outputs; candidate corrections
are event-centered. Attention uses frozen item vectors and masked, recency-weighted
history. Eight AdamW epochs use lr .001, weight decay .0001, batch2048, gradient
norm5, the existing half-event/half-global-row-pair objective and .1 mean squared
total correction. No preference retraining or checkpoint selection occurs.

The adapter shares parameters across day types. Its cold-status input is masked
because that flag is constant zero in prefix fitting; cold users retain the
original ridge fallback. Exposure adjustment, 37 features, combiner parameters
and original ensemble weights are preserved. Changed-member combiners are refit.

## Development selection

For each origin, adapters fit only earlier events. Fresh reference and candidate
combiners fit first users in the first 8h; validation uses first users after 8h,
ending before Jan25. All ten candidates remain together. Available model ages
extend beyond 48h for Jan21 and Jan22; Jan24 has only the remaining 16h.

| Composition | Jan21 change | Jan22 change | Jan24 change | Eligibility |
|---|---:|---:|---:|---|
| A: event head on dim8 seed42 | −0.000006838704 | −0.000065255922 | +0.000007491925 | Rejected: origin regressions |
| B: both heads on dim8 seed42 | +0.000012948099 | +0.000059153058 | +0.000039583501 | Selected |
| C: both heads on all five members | +0.000018307792 | +0.000259494304 | −0.000094088960 | Rejected: Jan24 regression |

B's late-development changes were +0.000108789389 at Jan21 and
−0.000080497324 at Jan22. Their mean is positive; the eligibility rule requires
the mean, not every late-origin change, to be nonnegative. Its weighted
no-click-negative contributions were positive at all three origins. B was the
only eligible composition and was frozen before promotion. No fallback was
screened after its failure.

Jan22–24 overlaps dates in the historical five-day comparison, and all benchmark
dates have influenced earlier research. These are retrospective development
results, not independent confirmation or selection-adjusted confidence.

## Promotion result

| Window | V6 AUC | Candidate AUC | Change | Paired 95% interval | Gate |
|---|---:|---:|---:|---|---|
| Primary Jan25–26 | 0.818459253326 | 0.818446307610 | −0.000012945717 | [−0.000071759803, +0.000039411469] | Failed |
| Secondary Jan26 | 0.827127712250 | 0.827156295425 | +0.000028583175 | [−0.000032561827, +0.000093630479] | Point gate passed |

Both cohorts had 1,000 valid paired whole-user draws, seed 2026. Primary requires a
strictly positive interval lower bound; secondary requires a nonnegative point
change. Development gains again failed to transfer to the primary window.

## Integrity, stopping and replay

- [Preparation and zero-adapter reproduction](preparation.json)
- [All three development comparisons](development.json) and [frozen selection](selected.json)
- [Primary and secondary screen](screen.json)
- [Terminal decision, training hashes, refusal checks and budget](decision.json)

All 15 real zero-adapter probability checks had zero error, covering five members
on samples from each historical cohort, including available cold users. Newly
staged seed43 models/combiners reproduced all three full saved vectors exactly.
Full V6 inference reproduced the canonical CSV SHA256:
`c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c`.
All 29 V6 assets and original source fingerprints verified. The existing explicit
JSON LF/CRLF adapter is unchanged.

Twenty adapters completed exactly eight epochs: eighteen development adapters
plus B's pre-Jan25 and pre-Jan26 adapters. Every training process verified that
all original base parameters stayed unchanged. Actual stress and finalize
commands refused at the failed screen before further fitting or export. No
pre-Jan27 adapter, stress result, V8 ready configuration or submission exists.

The campaign used 813.772249 seconds including preparation, development, screen
and refusal verification. Cumulative compute is 1,974.359347 seconds of the
original 14,400-second cap. Remaining budget does not authorize retrying this
closed recipe. Original checkout edits, prior trials, frozen sources, V6 assets
and configuration remain intact.

`tools/predict_v8_residual.py` implements hash-bound, gate-checked replay with
saved adapters, validated Jan25 combiners, two independent candidate computations,
CSV validation and no-replace publication; ready configuration is last. Model,
causality, cache and refusal tests passed. Real final-release inference remains
unrun because promotion failed; the publication-order test uses stand-ins and
does not establish a real successful V8 release. No competition upload occurred.

Final regression checks: the model environment ran 152 tests with one expected
optional-artifact skip. CPU source tests passed 131 tests and 20 subtests, with
21 expected dependency/artifact skips. The suite covers zero identity, masked
history, causality, permutations, frozen training, save/reload, cache tampering,
selection, failed-gate refusal and publication order.
