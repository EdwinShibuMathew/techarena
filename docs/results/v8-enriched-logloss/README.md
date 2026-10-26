> **Historical archive:** [PR #4](https://github.com/EdwinShibuMathew/techarena/pull/4), [reviewed source `3a90217`](https://github.com/EdwinShibuMathew/techarena/commit/3a902173cf111dcb748140a31293425ae66b2142).
> This recipe is closed and not submission-ready. Its historical execution tooling has confirmed review defects and is unsupported.
> Commands and validation claims below describe that revision. See the [archive and review dispositions](../v8-archive/README.md) and use the [shared workflow](../../ML_WORKFLOW.md) for new work.

# V8 enriched Logloss: rejected at the primary gate

The single fixed successor did not pass promotion. V6 remains the benchmark,
PR #4 remains draft, and there is no `configs/v8.json` or V8 submission CSV.
No five-day test, final classifier fit, or competition upload was performed.

| Window | V6 AUC | Candidate AUC | Change | Paired 95% interval |
|---|---:|---:|---:|---|
| Primary Jan25–26, 43,964 events | 0.818459253326 | 0.818505581189 | +0.000046327863 | [−0.000019299361, +0.000108900308] |
| Secondary Jan26, 35,138 events | 0.827127712250 | 0.827177344401 | +0.000049632151 | [−0.000010838815, +0.000110530540] |

The primary interval lower bound must be strictly positive. The secondary gate
uses nonnegative point change and passed. Both original seed42 comparison
members replayed with zero maximum error before either classifier was fitted.
All 1,000 whole-user bootstrap draws were valid in both windows. These dates are
reused development data; the intervals are not independent holdout guarantees or
adjusted for previous model selection.

## Fixed recipe and preservation

The [predeclared protocol](https://github.com/EdwinShibuMathew/techarena/blob/3a902173cf111dcb748140a31293425ae66b2142/experiments/v8-enriched-logloss/RUN_V8_ENRICHED.md)
adds six relative event features to the original ordered 37 columns. The new
Logloss classifier has 545 trees, depth 6, learning rate .05, L2 20, seed 42,
four CPU threads, and unit row weights. Compatible saved sampling/quantization/
leaf parameters were frozen before fitting. It replaces only dim8 seed42 at
weight 2/9, without calibration, sharpening, event normalization, or neural
updates. No settings were changed after seeing the screen outcome.

The rejected PairLogit source, reports, original private input manifest, and
caches remain intact. Reused caches bind their checkpoint, cutoff, event order,
schema, input hashes, and generating source; successor outputs are separate.
Preparation freshly replayed all five protected V6 members, required the
canonical baseline CSV SHA256, and checked the cached evaluation features and
predictions against that replay.

The original bundle verifies with `python tools/consolidate.py verify` in the
original checkout. The cloned checkout has the already recorded JSON LF/CRLF
manifest serialization difference. V8 uses the existing narrow metadata adapter
while checking every frozen Python file and all 29 model/data assets exactly;
no baseline file or numerical tolerance was changed.

## Evidence and release limits

- [Preparation and source inventory](preparation.json)
- [Full screen metrics, model parameters and prediction hashes](screen.json)
- [Stopping decision, budget and real later-stage refusal checks](decision.json)

Compute carries forward all 370.28605625100136 prior seconds under the original
14,400-second limit. Exact successor and cumulative usage are in the stopping
decision. Engineering tests and human waits are excluded. Remaining budget is
not permission to retry this closed recipe or tune on these dates.

`tools/predict_v8_enriched.py` validates complete source/input/model/report
inventories and all gates, reloads the classifier, reconstructs V6 and enriched
features, and uses atomic no-replace CSV publication. The ready configuration is
published last. The successful orchestration is covered with a synthetic saved
classifier; the real serializer is tested with 529,390 synthetic row IDs.
This successful release path has **not** been exercised on a final competition
model because the screen failed. Actual `stress` and `finalize` commands were
verified to refuse before fitting. No deployment or model-readiness claim is
implied by green source CI.

Private assets, cached arrays, classifiers, and prediction files remain ignored.
Anyone replaying the evidence needs the exact private inputs named by the
manifest; tracked source alone is not the asset bundle. A new modeling attempt
requires a separately declared recipe and authorization; neither closed trial
should be edited or silently restarted.
