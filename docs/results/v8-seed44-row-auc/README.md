> **Historical archive:** [PR #4](https://github.com/EdwinShibuMathew/techarena/pull/4), [reviewed source `3a90217`](https://github.com/EdwinShibuMathew/techarena/commit/3a902173cf111dcb748140a31293425ae66b2142).
> This recipe is closed and not submission-ready. Its historical execution tooling has confirmed review defects and is unsupported.
> Commands and validation claims below describe that revision. See the [archive and review dispositions](../v8-archive/README.md) and use the [shared workflow](../../ML_WORKFLOW.md) for new work.

# V8 seed44 row ranking: rejected at the five-day gate

The primary, secondary and late-window checks passed, but the overall five-day
paired confidence interval includes zero. The fixed recipe is closed. V6
remains the submitted benchmark, PR #4 remains draft, and no V8 ready
configuration or submission CSV exists.

| Window | V6 AUC | Candidate AUC | Change | Paired 95% interval | Gate |
|---|---:|---:|---:|---|---|
| Primary Jan25–26 | 0.818459253326 | 0.818530740894 | +0.000071487568 | [+0.000021871825, +0.000124804568] | Passed |
| Secondary Jan26 | 0.827127712250 | 0.827199121538 | +0.000071409288 | [+0.000015179727, +0.000128538878] | Passed |
| Five-day Jan22–26 | 0.781819085895 | 0.781831397251 | +0.000012311356 | [−0.000049257533, +0.000077476257] | Failed |
| Late 48–120h | 0.739471381060 | 0.739573227787 | +0.000101846727 | Point comparison | Passed |

All 1,000 whole-user paired draws were valid in each bootstrap window. Dates are
reused development evidence, not independent holdout or selection-adjusted
guarantees. Positive point changes do not override the five-day interval gate.

## Fixed neural change

The [predeclared protocol](https://github.com/EdwinShibuMathew/techarena/blob/3a902173cf111dcb748140a31293425ae66b2142/experiments/v8-seed44-row-auc/RUN_V8_SEED44.md)
extends V6's existing mixed objective from seeds42/43 to seed44. Reuse reviewed
epoch 260 preference checkpoints; train exactly eight event epochs with seed44,
fresh Adam lr=0.0015, batch size 2048, anchor regularization .1 and effect
regularization .05. Use the unchanged half-event-CE/half-global-row-pair loss,
16 sampled negatives per positive, and separate pair RNG seed3044.

Five models completed training at the pre-Jan21/22/24/25/26 cutoffs. Training
uses fitting rows only, without validation-based checkpoint selection or early
stopping. The decision report records checkpoint hashes, prefix counts/cutoffs,
maximum fitting timestamps, preference fingerprints and final losses.

The original architecture, exposure adjustment, ridge fallback, 37 features,
545-tree depth6/L2=20 combiners, four other members, and 2/9 seed44 weight are
unchanged. No preference retraining, calibration, routing, weight search or
post-outcome changes occurred. The final policy would reuse the validated
Jan25 combiner, whose original hash matches V6's collected seed44 combiner.
No Jan26 combiner refit was introduced.

## Integrity and stopping evidence

- [Preparation and source/input provenance](preparation.json)
- [Primary and secondary screen](screen.json)
- [Five-day and late-window comparison](stress.json)
- [Stopping decision, five-model training summary and compute accounting](decision.json)

Before new training, the original seed44 short-window component AUCs reproduced
at tolerance 1e-12; the earlier-cutoff predictions matched the full saved seed44
vector at tolerance 1e-15. Short checks use scalar historical AUCs and reviewed
checkpoint/combiner hashes, not a claimed short-window vector identity. All five
full-data V6 members replayed with zero maximum error and matched the canonical
V6 CSV SHA256. All 29 V6 assets and frozen Python bytes were checked exactly;
the existing LF/CRLF-only source-manifest adapter is unchanged.

Older short seed44 checkpoints lack seed/cutoff annotations. Their missing
fields remain null; reviewed hashes, original source/fold lineage, matching
training reports and independently proven prefix examples establish the
recorded provenance. Their newly reviewed hashes are not presented as
historically signed hashes. The earlier-window hashes match the old training
report. Historical archives, all prior V8 trials, V6 configuration, legacy
sources and the original dirty checkout remain untouched.

The real `finalize` command refused at the failed stress gate before launching
the full-data training process. No pre-Jan27 event model, ready configuration,
final competition prediction, or upload was produced. The successful release
orchestration is tested synthetically, including a saved neural checkpoint and
Jan25 combiner reuse; real final release replay remains unrun.

This trial used 205.320158269 seconds including refusal verification. Total V8
compute is 748.931401111 seconds against the original 14,400-second cap. Remaining
compute does not authorize retrying or changing this closed recipe. Hard stage
timers terminate training process groups; timeout cleanup is regression tested.

`tools/predict_v8_seed44.py` requires a complete verified ready release, binds
source/input/model/report inventories, recomputes gates, independently replays
the V6 ensemble and new seed44 features, and publishes CSVs without replacing
inputs. Readiness metadata is last. Green source CI does not establish model
promotion. Weights, prefix caches and predictions remain ignored private assets.
