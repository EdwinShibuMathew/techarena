# V8 residual-event campaign

Hypothesis: a bounded correction trained jointly with frozen V6 event scores
can improve global row ranking while retaining event click propensity. Comparator
is unchanged V6. Existing failed trials remain closed.

`residual_core.PROTOCOL` is the complete machine-readable recipe. Freeze its
source and input hashes before training. No preference fitting or base-model
updates; dim8 models retain their original epoch260 preference lineage and dim16
models their epoch196 lineage. Both adapter heads use 32 SiLU units, zero output
initialization and .5*tanh bounds. Candidate corrections are event-centered.
Use eight AdamW epochs, lr .001, weight decay .0001, batch2048, gradient norm5,
existing half-event/half-row-pair loss and .1 mean squared total correction.
All fitting labels and histories must precede their respective boundaries.

Three compositions, unchanged original ensemble weights:

- A: event-wide head on dim8 seed42.
- B: event-wide plus candidate head on dim8 seed42.
- C: both heads on all five members; reuse B's seed42 adapter exactly.

For each Jan21/Jan22/Jan24 origin, fit neural adapters strictly before the origin.
Fit fresh reference and changed-member combiners on first users in its first8h.
Validate first users after8h, ending before Jan25. Keep fitting and validation
feature calls separate because users can appear in both periods. Report model
age bands8–24/24–48/48–72/72–96h and warm/cold/history strata. No day-type-specific
adapter parameters; mask the unsupported cold-status input rather than
extrapolating from a feature that is always zero in fitting.

Eligibility: every origin aggregate gain nonnegative, mean gain positive, mean
available >=48h gain nonnegative, and mean weighted no-click-negative contribution
nonnegative. Choose highest minimum origin gain; ties1e-12 prefer A,B,C. Freeze
winner before promotion; no fallback. Jan22–24 overlaps the reused five-day dates.
These are retrospective development results, not independent confirmation or
selection-adjusted intervals.

Run in the existing techarena CUDA environment:

```bash
python experiments/v8-residual-event/runner.py prepare --source-root /home/edwin/projects/techarena-team
python experiments/v8-residual-event/runner.py development
python experiments/v8-residual-event/runner.py screen
python experiments/v8-residual-event/runner.py stress
python experiments/v8-residual-event/runner.py finalize
```

Advance only after success. No eligible composition, failed integrity/gate or
timeout closes this recipe. Primary paired95% lower>0; Jan26 point>=0; five-day
paired95% lower>0; late48–120h point>=0. Use1000 paired whole-user draws, seed2026,
at least990 valid. Preserve exact cohorts and original V6 prediction vectors.
Primary/secondary fitting uses the original Jan24/Jan25 first-user cohorts;
five-day fitting uses Jan21. Final adapters use pre-Jan27 original V6 scorers and
all prefix events, retaining each validated Jan25 combiner without refitting.

Carry all existing compute from the robust campaign receipt. Cumulative cap14400s;
phase caps600/6000/1800/1800/2400/600 for preparation/development/screen/stress/
finalization/verification. Hard timers terminate training process groups.

Only an eligible release permits:

```bash
python tools/predict_v8_residual.py --device cuda --output artifacts/reproduced_v8.csv
```

Verify all asset/source hashes, exact V6 CSV identity, zero-adapter reproduction,
independently loaded candidate replay twice, all529390 IDs/order/probabilities and
CSV fingerprints. Publish without replacement; ready configuration last.
No competition upload or merge. Keep PR4 draft unless every model and engineering
check passes. Source CI alone does not establish improvement.
