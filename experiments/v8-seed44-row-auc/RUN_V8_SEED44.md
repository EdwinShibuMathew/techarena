# V8 seed44 row-ranking extension

Hypothesis: V6's mixed event/global row-ranking objective, already used by
dim8 seeds42/43, improves seed44, whose original event model uses cross-entropy.
This is one fixed recipe, not a seed/objective/weight search. V6 remains the
comparator. All dates are reused development evidence; intervals are not
independent holdout or selection-adjusted guarantees.

## Frozen changes

Initialize only from reviewed seed44 epoch 260 preference checkpoints. Do not
retrain preferences or continue an already trained event8 checkpoint. Train
exactly eight event epochs with the original PreferenceModel/EventModel,
fresh Adam lr=0.0015, batch size 2048, seed44, anchor regularization .1, effect
regularization .05, and fitting-only original global-offset initialization.
Replace only event CE with the existing `mixed_row_auc_loss`: half eleven-way
CE plus half sampled global row-pair logistic loss. Pair sampling uses a separate
generator seeded 3044 and 16 negatives per clicked row, including no-click-event
negatives. Its original all-no-click-batch CE fallback is unchanged.

No validation labels select checkpoints, epochs, or schedules. Preserve the
original event-shuffle RNG stream and numerical settings. Model checkpoints
record seed, cutoff, source preference hash, objective, pair seed, fitting-index
hash, fitting count, maximum fitting timestamp, and final epoch8.

Keep the original 37-feature inference pipeline: eleven-outcome softmax,
exposure smoothing20/power.5, and prefix-only ridge cold-history fallback.
Combiners use Logloss, 545 trees, depth6, lr=.05, L2=20, seed42, four CPU threads,
unit weights, and compatible settings copied from the original seed44 combiner.
Replace only dim8 seed44 at weight 2/9: V6+(2/9)*(new44-old44). Retain the other
four members exactly; no clipping, normalization, sharpening, or calibration.

## Stages

| Stage | New neural prefixes | Combiner / target |
|---|---|---|
| Primary | pre-Jan24, pre-Jan25 | Fit Jan24; first-user Jan25–26 |
| Secondary | pre-Jan26, reuse pre-Jan25 meta features | Fit Jan25; first-user Jan26 |
| Five-day | pre-Jan21, pre-Jan22 | Fit Jan21; first-user Jan22–26 |
| Final | pre-Jan27 | Reuse validated Jan25 combiner; unlabeled evaluation |

The final policy matches V6: its collected seed44 combiner hash equals the
historical Jan25 combiner hash. There is no Jan26 combiner refit.

From the repository root in the existing CUDA model environment:

```bash
python experiments/v8-seed44-row-auc/runner.py prepare --source-root /home/edwin/projects/techarena-team
python experiments/v8-seed44-row-auc/runner.py screen
python experiments/v8-seed44-row-auc/runner.py stress
python experiments/v8-seed44-row-auc/runner.py finalize
```

Proceed only after a successful preceding command. Metric failure exits 2 and
closes the recipe. Integrity errors, training errors and timeouts preserve the
attempt and prohibit automatic retries. Do not remove journals or partial files.

Preparation reads only the authorized named historical manifests and seed44
assets, copies verified inputs into this trial, and proves prefix click examples
against fitting histories plus observed clicks. Short legacy checkpoints lack
seed/cutoff annotations: preserve these absent fields as null and reconcile
reviewed file hashes, per-fold source lineage and original training reports.
Earlier-window hashes also match the historical training report. Do not describe
newly reviewed short-window hashes as historically signed checkpoint hashes.

Before any new neural training, replay the original short seed44 models against
their recorded component AUCs, the earlier-cutoff seed44 predictions against the
full saved vector, and all five full-data V6 members against the canonical V6
CSV. Short historical AUC checks are scalar checks, not exact vector equality.
All new outputs stay in this experiment's ignored artifact directory; frozen
and historical files remain read-only.

## Compute and gates

Carry 543.611242842002 previous seconds against the original 14,400-second cap.
Stage caps: prepare 600, screen 3600, stress 3600, final 1800, verify 1800 seconds,
further bounded by remaining total time. A hard timer kills the entire training
process group on expiry and preserves the budget journal. Human waits and
engineering tests are excluded. No scratch preference training or new packages.

Require primary paired 95% CI lower > 0 and nonnegative secondary Jan26 change
before five-day fitting; then require five-day CI lower > 0 and late 48–120h change
>=0 before final training. Use 1000 whole-user paired draws, seed2026, and at
least 990 valid draws. A passed flag cannot override failed/incomplete evidence.

## Release

After all gates pass, save the new full-data event8 model and retain the already
validated Jan25 combiner. Reconstruct all five V6 members and the new seed44
features twice from saved assets. Require canonical V6 SHA256, existing
rtol=0/atol=1e-15 prediction tolerance, and identical independent V8 CSV hashes.
Validate 529,390 unique template-ordered IDs, schema, finite [0,1] probabilities.

```bash
python tools/predict_v8_seed44.py --device cuda --output artifacts/reproduced_v8.csv
```

Replay never trains a neural model or combiner. Complete inventories bind the
protocol, sources, inputs, new checkpoints, reused combiner, and gate reports.
Publish validated CSVs with no-replace hardlinks and publish the ready
`configs/v8.json` last. Keep PR #4 draft until real release replay and all three
required CI checks pass. No automatic merge or competition upload.
