# Project context and engineering lessons

Paths below are relative to the repository root. Read current
[repository instructions](../../../../AGENTS.md),
[shared ML workflow](../../../../docs/ML_WORKFLOW.md) and
[CI policy](../../../../docs/CI_RELEASE.md) for maintained contracts and commands.
This reference captures non-obvious lessons, not a replacement implementation.

## Benchmark and boundaries

TechArena ChaiScript predicts clicks for ten candidates per event, with at most
one click. The official metric is global row ROC AUC, not mean per-event AUC.
V6 is the submitted benchmark: three dim8 members, two dim16 members and CatBoost
combiners, covering 52,939 evaluation events / 529,390 candidate rows. Historical
public AUC is 0.819451. Its canonical CSV SHA256 is:

```text
c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c
```

Use `configs/v6.json` for the actual recipe. Preserve `legacy/`,
`configs/source_manifest.json` and `configs/v6.json`. New experiments belong in
`experiments/`; compact evidence belongs in `docs/results/`. Private weights,
arrays, CSVs and archives remain ignored. Do not scan `local/`, `data/` or
`artifacts/`; access necessary assets through explicit verified manifests and
supported scripts. Do not print raw binary or tabular assets into context.

## Scientific decisions

Declare the hypothesis, V6 comparator, candidate order/selection rule and stopping
rule before promotion. Use the same chronological refit/transfer pattern for
development and promotion: separate metadata-fitting and target neural prefixes,
strictly permitted fitting aggregates, whole ten-candidate events and exact row
ordering. Evaluation labels must not reach fitting or feature extraction.

The historical promotion contract uses first-user Jan25–26 primary events,
nonnegative Jan26 secondary change and the fixed five-day Jan22–26 comparison.
Primary and five-day paired 95% CI lower bounds must be strictly positive;
secondary and late 48–120h point changes must be nonnegative. Bootstrap is 1,000
whole-user draws, seed 2026, with at least 990 valid draws. Use current protocol
validation for exact cutoffs and cohort contracts. Reused dates are retrospective,
not independent holdouts or selection-adjusted guarantees. Do not tune to the
public leaderboard or retry a scientifically rejected recipe.

Checkpoints retain their true dimension, seed, epoch, cutoff and lineage. The
historical preference initializations use dim8 epoch 260 and dim16 epoch 196;
never change metadata to fit a shared expectation or use later-prefix weights.
Verify the actual initialization assets and current protocol before training.

## Integrity and execution lessons

- **Independent checkpoint expectations:** adapter code receives defensive deep
  copies of metadata and nested candidate parameters. Compare saved metadata
  with independently owned expected values. Reusing an adapter-mutated dictionary
  as the validator's expected value allowed invalid cutoffs/dimensions/epochs.
- **Immutable evidence:** bind a run to committed source, an immutable source
  snapshot, explicit input hashes and a frozen selection decision. Replay old
  evidence from its recorded revision/snapshot. Do not rewrite old hashes to
  make changed source appear compatible, or import chains of predecessor
  campaign workspaces for a new trial.
- **Compute ownership:** reserve allowance durably before launch, use an exclusive
  shared lock, and terminate workers/process groups on timeout or controller
  death. A phase denied the budget lock must remain unclaimed and launch no
  worker. Claim only after acquisition/reservation; preserve retry protection
  once claimed. Direct workers require an active controller lease and relevant
  gate prerequisites. Preserve cumulative usage across research run IDs.
- **Readiness last:** validate predictions and independent saved-model/CSV replay,
  then successfully finish accounting before publication. An accounting-write
  failure must not leave a readiness marker. Readiness must also be revalidated
  during supported replay; file existence alone proves nothing.
- **Atomic publication:** use unique staging files in the destination filesystem.
  JSON metadata uses atomic replacement with latest-successful-write semantics;
  CSV release publication preserves an existing destination. A failed write
  must not destroy the previous valid artifact. Keep these policies distinct.
- **Cache identity:** containment is necessary but insufficient. Bind reusable
  features to extractor/helper source, checkpoint bytes, fitting data/labels,
  cutoff, target order, feature schema and numerical runtime/device/precision.
  Validate checksums, shapes and dtypes before a hit. Preserve probability
  precision separately from intentional float32 features; widening an old
  rounded array cannot recover lost rankings. Independent final extraction
  bypasses the cache.
- **Line endings:** the source-manifest JSON may accept only verified LF/CRLF
  representation equivalence, recording declared and checkout hashes. Continue
  validating frozen source files against their exact bytes; this is not generic
  JSON equivalence or permission to edit locked files.

## Validation levels

| Evidence | What it establishes |
|---|---|
| Source tests, integrity checks and archive verification | The tested committed source can be packaged without changing frozen baseline bytes or tracking private assets |
| Required CPU model contracts and synthetic workflow | Actual synthetic fitting, gradients, causality, checkpoint reload, gates and failure handling; required contracts must run without skips |
| Exact private V6 replay | All five members match references within the enforced tolerance (historically <=1e-15), and the CSV matches the canonical SHA256 |
| Candidate promotion and independent final replay | The candidate passes every declared scientific gate and produces validated, reproducible output |

Read the current workflow and dependency lock rather than assuming an old test
count or environment still applies. The source-only suite may have legitimate
optional skips; a required model check that skipped core behavior is not a pass.
Use the separate locked CPU test environment; preserve a working private CUDA
inference environment. No level above substitutes for another.

## Review and archive practice

When archiving rejected work, preserve machine-readable reports byte-for-byte,
record original and archived hashes, annotate historical summaries explicitly,
and pin historical protocol links to the reviewed revision. Excluding obsolete
entrypoints does not repair or certify their old source. Keep recorded scientific
outcomes distinct from tooling defects and from unverified accounting guarantees.
Old green CI only establishes the checks and revision actually exercised.
