# Fixed V7 promotion validation

Hypothesis: fixed power sharpening at gamma 1.12 improves global row ROC AUC
against unchanged V6 while preserving event sums and within-event ordering.
Sharpening cannot improve within-event rank: its effect is on comparisons
between rows from different events. No parameters, model weights or seeds are
tuned. Dates are reused development data, not independent holdout evidence;
intervals are not adjusted for prior selection.

The runner stages only the three known, hash-locked prediction files listed in
`ARCHIVES`. Reading these exact historical files requires an explicit exception
to the repository's `local/` restriction. It never walks archives. Originals
are read-only; staged arrays and generated CSVs remain ignored by Git.
The active training cache, evaluation row IDs, example submission, and the
fingerprinted V6 CSV must also be present. Use the existing model environment;
this CPU validation requires no fitting or GPU and installs no packages.

```bash
python experiments/v7-validation/validate_v7.py prepare --source-root /path/to/original/repository
python experiments/v7-validation/validate_v7.py run
```

`prepare` freezes the recipe, source/input hashes, and copies of the approved
cohorts before scoring. It requires an unused staging directory. `run` checks
the frozen hashes before and after computation, verifies first-user cohorts
against training labels/timestamps, and uses 1,000 paired whole-user multinomial
bootstrap draws (seed 2026, 95% percentile intervals).

Promotion requires primary Jan25–26 interval lower > 0, secondary Jan26 point
change >= 0, five-day Jan22–26 interval lower > 0, and nonnegative change on the
five-day late 48–120 hour subset. A failure ends this fixed recipe; do not sweep
gamma or regenerate a comparator. All gates are recorded in the report.

The report at `docs/results/v7-validation/validation.json` also records a fresh
V7 CSV fingerprint check against canonical V6. Existing reports, staged inputs,
and generated outputs are preserved. This is evidence for admin review, with
no automatic PR merge, model fitting, or competition upload.
