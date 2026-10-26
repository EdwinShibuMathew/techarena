> **Historical archive:** [PR #4](https://github.com/EdwinShibuMathew/techarena/pull/4), [reviewed source `3a90217`](https://github.com/EdwinShibuMathew/techarena/commit/3a902173cf111dcb748140a31293425ae66b2142).
> This recipe is closed and not submission-ready. Its historical execution tooling has confirmed review defects and is unsupported.
> Commands and validation claims below describe that revision. See the [archive and review dispositions](../v8-archive/README.md) and use the [shared workflow](../../ML_WORKFLOW.md) for new work.

# V8 temporal search: selected B rejected at the primary gate

There is no V8 submission-ready configuration or CSV. The temporal development
search selected one winner, B, which failed its primary promotion gate. V6
remains the benchmark and PR #4 remains draft. No fallback candidate was
screened, and no five-day test, final competition-data fit, or export ran.

## Development selection

The [frozen protocol](https://github.com/EdwinShibuMathew/techarena/blob/3a902173cf111dcb748140a31293425ae66b2142/experiments/v8-temporal-combiner-search/RUN_V8_SEARCH.md)
compares three fixed recipes on first-user Jan24 events using pre-Jan24 neural
features. Fold 0 trains before 08:00 UTC and validates 08:00–16:00; fold 1 trains
before 16:00 and validates 16:00–24:00. Reference combiners are freshly trained
on each complete fitting block. Their validation events never enter fitting.

| Recipe | First fold component AUC change | Second fold component AUC change | Decision |
|---|---:|---:|---|
| A: 37 features, depth 4, L2 50 | +0.001061385034 | −0.000288451141 | Ineligible: negative second fold |
| B: 43 features, depth 4, L2 50 | +0.001185993162 | +0.000001967139 | Selected: highest qualifying mean |
| C: 37 features, depth 6, L2 20, separate shared logit correction | +0.000495673651 | +0.000583848743 | Qualified, lower mean |

These gains compare individual seed42 recipes with the fresh component
reference; they are not ensemble gains. All classifiers use 545 trees,
learning rate .05, seed 42, unit weights, and four CPU threads. C fits its base
on the first 75% of fitting events and its correction on the remainder, with
timestamp ties in calibration. Its standardization uses calibration events
only. No target outcome selected the winner. B's selection receipt was saved
and hash-bound before the new promotion screen. C was not screened afterward.

## Promotion outcome

| Window | V6 AUC | Selected B AUC | Change | Paired 95% interval |
|---|---:|---:|---:|---|
| Primary Jan25–26 | 0.818459253326 | 0.818448792190 | −0.000010461136 | [−0.000073408108, +0.000050981771] |
| Secondary Jan26 | 0.827127712250 | 0.827138621970 | +0.000010909720 | [−0.000063857112, +0.000086002488] |

The primary lower confidence bound must be strictly positive; it failed. The
secondary gate uses nonnegative point change and passed. Both original seed42
comparison members replayed with zero maximum error before fitting. All 1,000
whole-user bootstrap draws were valid in each window. All dates remain reused
development evidence, not independent holdout or selection-adjusted guarantees.

## Integrity and release limits

- [Preparation, baseline replay and source inventory](preparation.json)
- [Complete development results and artifact hashes](selection.json)
- [Frozen selected recipe](selected_recipe.json)
- [Promotion metrics and saved-model provenance](screen.json)
- [Stopping decision, compute carry-over and actual stage refusals](decision.json)

Preparation freshly replayed all five V6 members with zero maximum error and
verified the canonical V6 CSV fingerprint. All 29 original model/data assets
and frozen Python sources retain exact hashes. Existing V8 inputs/caches were
read only; newly created outputs belong to this search. The existing LF/CRLF
JSON metadata adapter is unchanged. Original `consolidate.py verify` succeeds
in the original checkout; V8's adapter verifies the clone's manifest binding.

Compute carries all 432.33501409900055 previous seconds against the original
14,400-second limit. The stopping decision records exact search and cumulative
usage. Engineering tests and human waits are excluded. Remaining compute does
not authorize changing this selection or screening a fallback after failure.

`tools/predict_v8_selected.py` binds full input/source/model/report inventories,
recomputes development selection and promotion gates, reloads the saved model
and optional correction, independently reconstructs the V6 ensemble, and uses
verified atomic no-replace CSV publication. The ready configuration is last.
The successful selected-release flow is tested synthetically, all three model
recipes have save/reload coverage, and the real serializer has full-size
synthetic CSV coverage. A real final selected V8 release was not exercised,
because promotion failed. Actual stress/finalize commands refuse before fitting.

All weights, cached arrays and predictions remain private ignored assets. The
source repository alone is not a model bundle. Neither prior closed V8 trial,
baseline configuration, legacy source, CI workflow, nor branch protection was
changed. A separate declared experiment is required for further modeling.
