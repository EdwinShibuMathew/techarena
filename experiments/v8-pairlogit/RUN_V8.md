# Fixed V8 PairLogit trial

Hypothesis: replacing only V6's dim8 seed42 Logloss combiner with a globally
sampled PairLogit combiner improves global row ROC AUC. Its weight stays 2/9.
All five neural checkpoints, original 37 features, four other combiners,
exposure adjustment and Ridge fallback formulas remain fixed. No V7 sharpening,
neural retraining, weight tuning, or competition upload is performed.

The recipe in `v8_core.PROTOCOL` is frozen before fitting: 545 CPU trees, depth 6,
learning rate .05, L2 20, seed 42, four threads, and compatible saved settings.
Each positive has sixteen uniformly sampled negative rows (seed 3042), without
replacement per positive, across the fitting day including no-click events.
One CatBoost group permits cross-event pairs. The unit-slope sigmoid intercept
is fitted only on that day's predictions/labels using 80 bisections to match
its row CTR. Tree counts, sampling, link, weights and seeds are never swept.

## Inputs and temporal rules

Run from an isolated checkout of the repaired code, in the existing `techarena`
environment with the verified RTX 4070. `prepare` copies the exact active V6
bundle and the report-named historical assets from `--source-root`; it never
walks archives. This task explicitly authorizes these manifest-listed reads
despite the general `local/` restriction. Original inputs remain untouched.

| Case | Fitting day / feature prefix | Target / feature prefix |
|---|---|---|
| Primary | Jan24 / before Jan24 | First user Jan25–26 / before Jan25 |
| Secondary | Jan25 subset of primary features | First user Jan26 / before Jan26 |
| Five-day | Jan21 / before Jan21 | First user Jan22–26 / before Jan22 |
| Final | Jan26 first-user features from the secondary cache | Unlabeled evaluation / before Jan27 |

Meta cohorts are derived deterministically from the raw cache; they are not
claimed to be recovered archive split files. Prefix positive-click multisets
are proved against fitting-user initial histories and observed prefix clicks,
retaining repeats and excluding future-only users. Every feature cache binds
checkpoint, cutoff, data/source hashes, event order and feature schema.

Before either screen fit, both old seed42 members must replay at rtol=0,
atol=1e-15. The historical NPZ `candidate` is V6; `baseline` is V5. Five-day
seed42 comes from `candidate_members[0]`. Identity replacement must preserve
V6 exactly. Preparation also replays all five full-data V6 members and the
canonical V6 CSV SHA256. Any mismatch closes preparation before a ranker fit.
The bundle's source-manifest binding permits only LF/CRLF serialization of the
same JSON bytes, to account for Git's checkout normalization. Every historical
Python source, model and asset still requires its original exact SHA256; the
line-ending adapter and both manifest fingerprints are recorded explicitly.

## Commands and budget

```bash
python experiments/v8-pairlogit/runner.py prepare --source-root /path/to/original/repository
python experiments/v8-pairlogit/runner.py screen
python experiments/v8-pairlogit/runner.py stress
python experiments/v8-pairlogit/runner.py finalize
```

Use the existing environment; GPU stages need OS-level GPU access. No package
installation is part of the experiment. Outputs under ignored
`artifacts/experiments/v8-pairlogit/` and compact reports under
`docs/results/v8-pairlogit/` are preserved. Existing phase attempts are not
automatically retried. A failed gate returns exit code 2 and forbids advancing.

Compute limits, excluding implementation work and human waits: 30 minutes
prepare, 60 minutes screen, 60 minutes stress, 60 minutes final fit, and 30
minutes independent verification; 14,400 seconds total. The persistent budget
journal accounts for completed/failed phases, with a hard process timeout
during expensive calls. Timeouts preserve partial outputs and do not produce
a ready configuration. Do not alter the recipe to fit a smaller budget.

## Promotion and submission

Evaluate global row AUC, half-credit for ties, against unchanged V6. Use 1,000
paired whole-user bootstrap draws, seed 2026, 95% percentile intervals, at
least 99% valid draws. Primary interval lower must be >0 and secondary point
change >=0 before stress. Five-day interval lower must be >0 and the pooled
late 48–120h point change >=0. Per-day and pair decompositions are diagnostics,
not routing or selection rules. Dates are reused development data, not an
independent holdout; intervals are not selection-adjusted.

Only after all four gates pass, fit the same recipe on honest Jan26 meta data.
Save the ranker and training-only intercept, generate a draft, then independently
reconstruct features and all five V6 members from saved assets and reload the
new ranker. Both V8 CSVs must have identical SHA256 before publication at
`artifacts/submissions/v8_candidate.csv` and creation of `configs/v8.json` with
`submission_ready=true`. Existing different files and input aliases are refused.
The release binds model, link, validation reports, code and baseline hashes.

After a successful release, replay without tree fitting:

```bash
python tools/predict_v8.py --device cuda --output artifacts/reproduced_v8.csv
```

The file must have 529,390 unique row IDs aligned to the example submission,
finite probabilities in [0,1], and the frozen V8 fingerprint. No clipping,
automatic merge, or portal upload is supported. A failed recipe leaves V6
as the benchmark and produces no submission-ready V8 release.
