# V7.4 implementation validation — research not started

**Not promoted; no competition submission produced.** The graph model and full
V6/V7 promotion workflow are implemented, but this checkout lacks the verified
trained bundle, chronological prefix assets, processed caches and authoritative
compute ledger. The scientific campaign has not started. No candidate has been
selected and no promotion AUC or confidence interval has been measured.

Implementation revision: `8636f6803d2c9112513c3d8df672be5eb0d82378`, based on
merged main `329d79cdd279dcf4f8c0359a15815cb42d2e5026`.
The [frozen protocol](../../../experiments/v7.4-history-graph-v1/RUN_V7_4.md)
declares the hypothesis, two variants, chronology, selection and stopping rules.

## Implemented behavior

- Frozen V7 plus a sparse, distinct-user co-click graph using observed warm-up
  histories and prefix clicks. Graphs exclude self-links, future clicks and
  non-clicked impressions; repeated snapshots do not inflate support.
- Two predeclared 24h/72h history-decay variants. A single bounded learned
  coefficient changes candidate ranking while preserving V7 event click mass.
  Empty or constant-evidence events retain V7 scores exactly in mixed batches.
- A separate schema-v4 context interface and explicit `v6_v7_four_gate` policy.
  All four original checks must pass against each anchor. Older schemas and the
  V7.3 upload-trial policy retain their original behavior.
- Frozen-source fitting, independent lineage/normalization checks, checksum-bound
  graph caches, independent uncached final graphs, and graph-bound supported
  replay evidence. Accounting still precedes public publication and readiness.
- Read-only prerequisite checks refuse missing/replaced inputs or absent,
  unfinished, unreconciled accounting before creating any research phase claim.

## Local validation

| Check | Result |
| --- | --- |
| Locked CPU full regression suite | 183 passed; 1 optional artifact skip |
| Required CPU model contracts | 70 passed; zero skips |
| Lightweight source profile | 153 passed; 31 dependency/artifact skips |
| V7.4-specific required tests | All 18 passed |
| Locked dependency consistency | Passed |
| Source integrity | All 59 frozen files verified |
| Source package and checksum verification | Passed |

The required model-contract runner includes every V7.4 test and rejects skips.
Only the lightweight source profile skips SciPy/model-dependent tests. The
locked suite's single optional skip is the absent private V7 candidate CSV.

New tests use actual graph construction, an independent distinct-user oracle,
finite-difference checks of both derivatives, actual scalar optimization and
serialized reload. They cover causal cutoffs, warm-up/repeat handling, opaque
encodings including int32 IDs, sparse caps/ties, endpoint/mass/permutation
properties, cache corruption, independent metadata expectations, all eight gate
failure paths, modified receipts/chronology, accounting failures and terminal
stopping. The actual synthetic CLI campaign completes preparation, development,
screening, five-day testing, two uncached final graphs and matching 480-row CSVs.
An uninformative graph closes the synthetic recipe before later phase claims.

These synthetic baselines and datasets are orchestration/model-contract evidence.
The publication control fixture substitutes successful numerical evidence to
isolate readiness behavior. Neither establishes actual V7.4 performance or exact
private V6/V7 reproduction. All existing frozen source and V6/V7 configurations
remain unchanged; no historical result or receipt was rewritten.

## Restoration and next execution

The supplied train.zip, evaluation.zip and example_submission.csv matched their
declared original fingerprints and were restored under ignored data/raw paths.
The original Competition Folder remains intact and ignored. Raw data alone does
not replace trained checkpoints, prefix references or the accounting ledger.

The read-only checker currently returns `ready_for_prepare: false`. No executable
research experiment.json, research ledger, phase claim, V7.4 ready configuration
or submission CSV has been created. The last historical reported usage is
4,459.699446460001 of 14,400 seconds; current authoritative usage is unverified
until the original ledger is restored. Do not recreate or reset it from that
historical number, and preserve one authoritative compute owner on restoration.

After restoring the exact private inputs and reconciling the ledger, run the
prerequisite checker and freeze/commit the research specification, then follow
the protocol's gated commands. Stop at the first failure. Promote V7.4 only
after actual eight-gate acceptance and independent private final/replay checks.
Competition upload remains manual.

The [validation record](validation.json) binds local checks, source-package
identity and the explicit unstarted research status to the implementation
revision. GitHub CI results belong to the actual revision tested remotely.
