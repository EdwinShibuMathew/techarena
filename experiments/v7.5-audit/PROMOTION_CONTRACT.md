# Future promotion evidence contract

`ml_workflow.promotion.verify_promotion(root, evidence)` returns `eligible`,
`blockers`, recomputed comparisons and eight numerical gates. It publishes
nothing. A future publisher must require `eligible is True` and bind the exact
evidence hash into its release receipt. `metric_gates` is diagnostic and cannot
authorize a release. Existing closed V7.5 publication code is not retrofitted or
waived by this contract.

The descriptor is schema version 1. Every named file is a root-relative regular
path in `artifacts: {path: sha256}`; symlinks and traversal are rejected. These
identities must be frozen by the experiment controller before evaluation, not
invented by a publisher after seeing results. The verifier checks consistency
and recomputes metrics; it is not an authenticity service for arbitrary supplied
manifests or a replacement for trusted source/accounting ownership.

Required descriptor fields:

| Field | Contract |
|---|---|
| `schema_version`, `purpose`, `model_role` | `1`, `research`, `candidate`; fallback and diagnostic roles cannot promote |
| `candidate`, `baseline_lineage`, `comparators` | Frozen winner; `rebuilt_v6_v7_recipe`; exact V6/V7 identity hashes |
| `bootstrap` | `resamples:1000`, `seed:2026`, `minimum_valid:990` |
| `run_manifest` | Inventoried frozen manifest described below |
| `development` | `winner`, `passed:true`, `report` containing original development results |
| `source_record`, `source_directory` | Snapshot record and directory containing `source.tar.gz` and `source/` |
| `engineering_receipt` | `kind:source_readiness`, snapshot `git_commit`, checks `cpu_tests`, `model_contracts`, `source_integrity`, `source_archive`, all `success` |
| `accounting_record`, `budget_genesis` | Settled original ledger and its genesis, not a new budget |
| `data_metadata` | Pinned reconstructed data metadata with original arrays adjacent |
| `predictions` | `primary`, `secondary`, `five_day` paths to accounted prediction NPZs |
| `example_submission` | Pinned official submission template |
| `exports` | At least two independently generated `{csv, receipt}` pairs |

The manifest contains `candidate`, `baseline_lineage`, `comparators`, nonempty
`models:{path:sha256}`, `source_sha256` (snapshot archive), `data_sha256`
(metadata file), `protocol`, `references:{primary,secondary,five_day:path}` and
`owner_prefix`. The protocol declares `release_policy:v6_v7_four_gate`,
`candidate_order` and the causal `development` cases. The development winner is
recomputed by the original dual-anchor selection rule, rather than trusting a
passed flag. The references must be independently frozen comparator vectors,
bound by the manifest before candidate scoring.

Each prediction NPZ contains `indices`, `rowids`, `labels`, `candidate`,
`baseline`, `v7`, and scalar `V6_identity`/`V7_identity` strings. First-user indices,
labels, IDs, order, and fixed cohort counts are recomputed from the pinned full
training corpus. Reference predictions must match the frozen vectors within
absolute tolerance 1e-15. Probabilities must be finite and in [0,1]. Primary and
five-day paired whole-user 95% lower bounds must be strictly positive against
both anchors. Secondary and 48–120-hour late differences must be nonnegative
against both. Missing evidence fails; identity scores cannot pass strict gates.

Each export receipt binds `candidate`, `fresh_neural_features:true`,
`fresh_graph:true`, `source_sha256`, `manifest_sha256`, `model_sha256` (canonical
hash of the model dictionary), `csv_sha256`, and a distinct `compute_owner`.
Each owner must belong to the manifest generation and have a completed budget
reservation. CSV files must be physically distinct, byte-identical, and pass
the frozen official validator with all 529,390 rows in official order. Freshness
claims must originate in the accounted worker that actually performs inference;
writing a new receipt around cached predictions is not independent inference.

The original ledger must reconcile all attempt charges, contain no running
reservation, retain its genesis identity, and stay within 43,200 seconds.
Source tests, synthetic fixture successes, CSV validity, and public leaderboard
results each have narrower meaning than scientific promotion. No automatic
competition upload is part of this verifier.
