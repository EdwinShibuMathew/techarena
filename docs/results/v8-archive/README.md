# Archived V8 campaigns from PR #4

These six campaigns are closed research evidence. None produced a candidate that
passed every promotion gate, and none produced a submission-ready V8. V6 remains
the submitted benchmark. The dates were repeatedly reused; the intervals are
retrospective and are not independent or selection-adjusted guarantees.

The archive comes from [PR #4](https://github.com/EdwinShibuMathew/techarena/pull/4)
at [reviewed commit `3a90217`](https://github.com/EdwinShibuMathew/techarena/commit/3a902173cf111dcb748140a31293425ae66b2142).
The [provenance manifest](manifest.json) records original and archived hashes for
all 32 reports. The 26 JSON reports are unchanged byte-for-byte. The six Markdown
summaries have only an archive banner and a protocol link pinned to that commit;
their measurements, decisions and original validation claims are preserved.

| Campaign | Recorded stopping decision | Evidence |
|---|---|---|
| PairLogit | Primary confidence gate failed | [Screen and decision](../v8-pairlogit/README.md) |
| Enriched Logloss | Primary confidence gate failed | [Screen and decision](../v8-enriched-logloss/README.md) |
| Temporal combiner search | Selected B; primary confidence gate failed | [Selection and decision](../v8-temporal-combiner-search/README.md) |
| Seed44 row ranking | Primary/secondary/late checks passed; five-day confidence gate failed | [Screen, stress and decision](../v8-seed44-row-auc/README.md) |
| Robust ensemble | Selected A; primary confidence gate failed | [Development and decision](../v8-robust-ensemble/README.md) |
| Residual event | Selected B; primary confidence gate failed | [Development and decision](../v8-residual-event/README.md) |

## Review dispositions

The [tech lead review](https://github.com/EdwinShibuMathew/techarena/pull/4#issuecomment-6429021053)
identified three valid defects in the historical tooling. That tooling is
excluded from this archival change; the old source has not been repaired or
certified for execution.

| Finding | Historical consequence | Supported replacement |
|---|---|---|
| P1: budget accounting on controller death and overlapping phases | A worker could outlive its controller without being charged; overlapping journals could overwrite each other | PR #17's shared runtime reserves compute before launch, enforces exclusive ownership and terminates orphaned workers |
| P1: readiness before verification accounting completes | A failed ledger write could leave a ready configuration despite failed finalization | The shared controller publishes only after successful accounting; a fault-injection regression verifies that an accounting-write failure prevents publication |
| P2: direct seed44 trainer bypasses gates and budgeting | A standalone trainer could fit despite failed stress and no active budget | New work uses the shared controller's phase prerequisites and active worker lease; the historical standalone CLI is excluded |

These defects concern tooling guarantees. They do not change the recorded
scientific decisions or establish that any reported metric is numerically
wrong. Old passing tests and replay claims describe their recorded revisions;
they do not validate the excluded execution paths against this review.

## Budget and future work

The latest decision records **1,974.359347462001 seconds** against the original
**14,400-second** cumulative cap. Keep that carry; archiving does not reset the
research budget or authorize retrying a rejected recipe. These are recorded
historical totals, not proof that controller-death or overlap accounting was
safe. Reconcile any demonstrable unrecorded real work before another campaign.

Use the [merged shared ML workflow](../../ML_WORKFLOW.md) for a separately
reviewed new hypothesis. Reusable model ideas remain inspectable at the pinned
source revision; this archive imports no model code, trainer, predictor or
implementation-dependent test from PR #4. Existing V6 files, private assets,
closed decisions and earlier validation evidence remain unchanged. No new
research training, private CUDA replay, competition submission export or upload
was performed. Automated contracts may fit disposable synthetic models.

## Retirement sequence

The repository admin reviews and merges this evidence-preservation PR. After
that merge, close PR #4 as superseded and retain its branch and commits. Until
then it remains draft, linked to the replacement. Historical mentions of PR #4
remaining draft inside the copied reports describe the original timeline.

## Archival validation

[Fresh validation evidence](validation.json) records verification against the
pinned Git source for all 32 original report hashes. The locked CPU suite passed
134 tests with one optional private-artifact skip; all 21 required model/workflow
contracts passed without skips. Source-only checks passed 124 tests with 11
optional skips. Dependency consistency passed. The regression exercises real
budget accounting and a failing ledger write; numerical finalization and
publication are stand-ins, so this is not successful V8 release evidence.
