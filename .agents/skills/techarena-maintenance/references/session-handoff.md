# Dated session handoff

Status last checked **2026-10-26 23:05 UTC** against GitHub. The checkout's base
was main commit `0d6888f6f861c5b6b0ca5b5b3deefdaaae65499d`. Refresh live Git/PR/CI
and verified budget state before acting; this file is history, not a task queue
or authorization to repeat past operations.

## User intent and workspace preservation

The user wants a genuinely improved submission-ready V8, reliable ML engineering
and reviewable repository changes. The conversation separately authorized
workflow infrastructure, PR fixes and archival work; none established a promoted
V8. Do not turn a documentation/maintenance request into a new modeling campaign.

The original checkout contained unrelated modified experiment/prediction files
and untracked research directories. Prior work used isolated source checkouts
and preserved those edits. Inspect the current tree: do not assume those changes
still exist, overwrite them, or hardcode their old machine-specific paths.

## Completed workflow work

[Merged PR #17](https://github.com/EdwinShibuMathew/techarena/pull/17), head
`ffa73f5356598b8db51865608c3bb2b41f924e31`, merged as
`06c8cbdf4348b4047cac774fae3fd1032054acbe`, introduced the standalone shared
workflow, verified feature reuse and required CPU model contracts. It addressed
the work tracked in issues #9–#16; verify individual issue closure independently.

Its [review](https://github.com/EdwinShibuMathew/techarena/pull/17#issuecomment-6428253067)
found two valid regressions, both fixed before merge:

- `2062654`: keep independent checkpoint expectations and deep-copy adapter
  metadata/candidate parameters. Tests rejected mutated cutoff, dimension and
  preference epoch for both dimension roles and preserved valid receipts.
- `ffa73f5`: acquire/reserve the budget before writing the phase claim. Tests
  proved that contention leaves no claim/worker and permits a later start, while
  retaining retry protection for phases already claimed.

The four CI checks were `cpu-tests`, `model-contracts`, `source-integrity` and
`release-readiness`. The updated protection configuration describes the desired
policy; do not infer the live GitHub protection settings from that file.

[Workflow evidence](../../../../docs/results/workflow-validation/README.md) records
an exact private CUDA V6 replay from fresh LF source: all 29 assets verified,
all five member differences zero, canonical CSV SHA256 matched. The
[review-fix evidence](../../../../docs/results/pr17-review-fixes/README.md) records
129 passing locked-CPU tests with one optional artifact skip and 20 required
contracts with zero skips. These counts and the CUDA proof belong to their
recorded sources; they are not a fresh replay of later changes.

## Rejected V8 work and archival decision

[PR #4](https://github.com/EdwinShibuMathew/techarena/pull/4) is **closed without
merging**. Its preserved head is
`3a902173cf111dcb748140a31293425ae66b2142`. The user chose to preserve evidence and
retire its outdated tooling rather than modernize the duplicated execution code.
Its original branch/commits were retained.

[Merged PR #18](https://github.com/EdwinShibuMathew/techarena/pull/18), head
`b5ae2a7a4e84c23d189f7ad9ffc278241cf70de5`, merged as
`0d6888f6f861c5b6b0ca5b5b3deefdaaae65499d`, preserves 26 JSON reports byte-for-byte
and six summaries with only archive notices and commit-pinned protocol links.
The [archive](../../../../docs/results/v8-archive/README.md) and
[manifest](../../../../docs/results/v8-archive/manifest.json) are the sources for
those historical measurements and their provenance.

| Closed campaign | Recorded stopping gate |
|---|---|
| PairLogit | Primary confidence gate failed |
| Enriched Logloss | Primary confidence gate failed |
| Temporal combiner search, selected B | Primary confidence gate failed |
| Seed44 row ranking | Five-day confidence gate failed despite passing screen/late checks |
| Robust ensemble, selected A | Primary confidence gate failed |
| Residual event, selected B | Primary confidence gate failed |

The [PR #4 review](https://github.com/EdwinShibuMathew/techarena/pull/4#issuecomment-6429021053)
found unsafe accounting under controller death/overlap, readiness published
before accounting succeeded, and a standalone seed44 trainer bypassing gates
and budgeting. PR #18 **excluded** those old execution paths; it did not repair
or certify the historical runners. Future work uses the merged shared runtime.

PR #18 added a final-accounting-write-failure regression: publication is not
called, no readiness marker exists, and the durable reservation remains charged.
Numerical finalization/publication in that test use stand-ins. Its recorded
validation was 134 passing locked-CPU tests with one optional private-artifact
skip and 21 required contracts with no skips. No private CUDA replay or research
training was rerun for that archive. All four GitHub CI checks passed.

## Research continuation boundaries

The latest archived decision records **1,974.359347462001 seconds** of the
original **14,400-second** cumulative research cap. Preserve that historical
carry and reconcile it with newer verified usage; do not reset the ledger or
assume this remains the current consumed total. Recorded old totals are not
proof that the historical accounting was safe under crashes or overlap.

As of this snapshot, no V8 candidate passed every gate, and no submission-ready
V8 configuration or CSV was produced. New scientific work needs its own declared
hypothesis and reviewed adapter using the shared workflow, correct prefix assets
and existing promotion gates. Synthetic CI validates the workflow, not a measured
improvement over V6. The skill itself does not select a new recipe, permit a
closed one to restart, or authorize training, merging or competition upload.
