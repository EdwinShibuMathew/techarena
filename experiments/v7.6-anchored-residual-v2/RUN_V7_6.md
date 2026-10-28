# V7.6 bounded V7-anchored residual, revision 2

One focused recipe tests whether causal stale-prefix graph views can improve
global row AUC without the cross-event regression observed in V7.5. The prior
design and closed experiments remain unchanged. Comparisons use rebuilt V6 and
V7, not missing historical vectors. Reused dates remain retrospective evidence.

The fixed 13-column input and 16-unit tanh residual retain a ±0.25 log-odds
bound, exact identity on uncovered/empty-history events, and exact endpoints.
Raw dates, prefix ages/counts and an event-sum constraint are excluded. The
objective combines row BCE, minibatch global-row pairs, and Bernoulli anchor KL.
Hidden weights use Xavier uniform gain one; the output starts at zero. Training
uses CPU float64, 32 epochs, AdamW lr0.001/wd0.01, 2,048-event batches, and seeds
2026/5026/6026 for initialization/order, pairs, and view sampling. No early stop.
Normalization is fitting-only, event-balanced over views, with scale floor1e-6
and standardized features bounded to ±5; output probabilities are never clipped.

Existing neural checkpoints are reused. Six-hour forecast blocks start Jan20
12:00 and end Jan26 00:00. Every allowed pair has 6h<=d-c<=48h, d<=block start,
and age<168h. The unchanged 37-feature combiner fits [c,min(c+24h,d)); graph and
prediction neural prefixes are d. Each event retains total weight one: one
uniformly sampled view per epoch. No duplicate view is an independent label.
The first fitting record per user in the preceding 48h is used.

Inner validation uses Jan21 12–18 and18–24, stable five-user-group exclusion,
and freshest/oldest regimes. Three fixed penalties0.1/1/10 must have nonnegative
point changes for every block/regime/anchor and positive mean versus each anchor.
Select the maximum worst paired95% lower bound, ties choosing larger penalty
within1e-12. Bootstrap1000/seed2026/min990. Failure closes the experiment before
outer preparation. Otherwise freeze one penalty and source.

Four independent residual states end Jan22/23/25/26. Jan22 serves development
and five-day testing; Jan26 serves secondary testing and final inference.
All model and comparator hashes freeze before outer scoring. Development must
pass both windows and its aged subset against both anchors. Screening requires
primary CI lower>0 and secondary delta>=0 versus both. Stress requires five-day
CI lower>0 and48–120h delta>=0 versus both. Each failed stage is terminal.

The shared original43,200-second ledger carries all prior charges. No new
ledger, scratch neural training, cap extension, or scientific retry is allowed.
Jobs reserve durably before launch; training cap900s, other cap1200s; protect
3600s for finalization. Workers require the owning lease and exact frozen source
and runtime. Completed task claims, source, outputs and accounting are bound.
Only identical fully validated non-export work can be recovered after technical
failure. Fresh exports always execute again rather than relabeling cached work.

Before preflight, test/commit the code, pass its CI, and use the existing
source-readiness packager to produce
`artifacts/v7_6-engineering/source-readiness.json` for that commit. Execute with
the original `techarena` environment and CUDA access for feature extraction:

```bash
python experiments/v7.6-anchored-residual-v2/runner.py preflight
python experiments/v7.6-anchored-residual-v2/runner.py prepare-inner
python experiments/v7.6-anchored-residual-v2/runner.py select
python experiments/v7.6-anchored-residual-v2/runner.py prepare-outer
python experiments/v7.6-anchored-residual-v2/runner.py freeze
python experiments/v7.6-anchored-residual-v2/runner.py develop
python experiments/v7.6-anchored-residual-v2/runner.py screen
python experiments/v7.6-anchored-residual-v2/runner.py stress
python experiments/v7.6-anchored-residual-v2/runner.py finalize
python experiments/v7.6-anchored-residual-v2/runner.py verify
```

`run` performs the full ordered campaign; `status` reports compact progress.
`replay` performs fresh standalone inference after two staged exports or after
publication. Finalize runs two exports plus that staged replay, then paid
artifact-backed verification. Each final run regenerates neural features and
graph, checks member probabilities within1e-15, and produces the same official
529,390-row CSV. Task-validated vectors must match CSV probabilities. Publication
follows durable final accounting, preserves existing output, and writes a new
V7.6 configuration/checksum/receipt. Competition upload is manual. If any gate
fails, no V7.6 ready configuration or candidate submission is published; the
verified rebuilt-V6 fallback remains available.

Source CPU tests may skip unavailable optional dependencies. Mandatory model
contracts must include every V7.6 test without skips. Synthetic fitting/export
tests establish engineering behavior only and never establish real promotion.
