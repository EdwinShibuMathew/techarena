# V7.6 measured result: rejected at inner selection

**V7.6 was implemented and its complete inner campaign ran. It is not promoted
or submission-ready.** All three declared regularization penalties improved
over rebuilt V6, but each regressed against rebuilt V7 in required validation
regimes. Frozen selection returned no winner and closed the recipe before outer
preparation. No V7.6 candidate CSV, release configuration or readiness receipt
exists. The existing verified rebuilt-V6 competition CSV remains available.

## Executed design

The revised recipe retains the V7-anchored, 13-feature, 16-unit tanh residual
bounded to ±0.25 row log-odds. It adds causal stale-prefix views, fitting-only
normalization with total weight one per event, and grouped historical validation.
It excludes raw age/date/global prefix counts and does not preserve event score
sums. Empty-history/uncovered events and exact endpoints retain the anchor.

Training used the frozen 32 epochs, AdamW learning rate 0.001/weight decay 0.01,
CPU float64, 2,048-event batches, BCE/global-row-pair/anchor-KL objective and
seeds 2026/5026/6026.
No neural checkpoints were trained, replaced or relabeled. No hyperparameter was
changed after seeing validation results. All dates remain retrospective evidence.

The scientific source was **12521005a8484e5b534fdf205285c2d00b843ed7**.
All four [source CI jobs](https://github.com/EdwinShibuMathew/techarena/actions/runs/40773371592)
passed before the campaign. Actual production execution completed:

| Work | Completed jobs |
|---|---:|
| Verified preflight | 1 |
| Causal neural-feature extraction | 65 |
| Fixed-recipe 37-feature anchor combiners | 15 |
| Historical forecast views | 14 |
| Residual fits: three penalties × two origins × five user folds | 30 |
| Paired-bootstrap selection | 1 |
| Total | 126 |

Each residual fit excluded its validation user group from prior residual fitting
records. The Jan21 12:00 origin used about 28,000 fitting events per fold; the 18:00
origin used about 31,000. Each model completed 32 epochs and saved validation
predictions for both the freshest and oldest eligible prefix regimes.

## Why selection failed

The frozen rule required nonnegative point changes in every origin/regime against
both anchors and a positive mean against each. Eligible penalties would have been
ranked by their worst paired 95% confidence lower bound, ties favoring stronger
regularization. Each of the 24 comparisons used 1,000 whole-user draws with seed 2026;
all draws were valid. None of the three penalties was eligible.

| Penalty | Jan21 12–18 fresh Δ vs V7 | Jan21 12–18 stale Δ vs V7 | Jan21 18–24 fresh Δ vs V7 | Jan21 18–24 stale Δ vs V7 |
|---|---:|---:|---:|---:|
| 0.1 | +0.000155212 | −0.000237185 | −0.000043490 | −0.000198138 |
| 1 | +0.000154499 | −0.000218897 | −0.000036453 | −0.000187736 |
| 10 | +0.000121776 | −0.000075192 | −0.000006727 | −0.000093890 |

Gains against V6 ranged from +0.001012030 to +0.003567826, but those do not waive the
V7 comparisons. For penalty 10, the late stale interval versus V7 was
[−0.000214016,+0.000024961]. That uncertainty crosses zero; the predeclared
nonnegative point requirement still fails. Penalty 10 is reported as the strongest
declared regularization, not as a selected or deployable candidate.

[Exact results](results.json) preserve all comparisons, source/protocol identities,
prediction hashes and the terminal decision. The outer development windows,
primary, secondary, five-day and late promotion checks were **not run**. None of
the eight outer gates can be claimed passed or failed from this inner result.
No runner-up, post-hoc blend, comparator waiver or leaderboard tuning followed.

## Engineering and accounting

The full local suite passed **264 tests**, with one optional missing historical
V7 CSV skip. All **151 required model contracts** passed without skips, including
26 V7.6 tests. Tests exercise real residual fitting, independent loss/gradient
oracles, zero-initialization gradients, deterministic replay, user/time boundaries,
event weighting, source/claim binding, reserve checks under the ledger lock,
accounting failure, and fresh independent synthetic CSV exports. Downstream
production inference/publication was not exercised because the scientific
prerequisite failed; its synthetic tests do not establish model promotion.

The campaign charged **920.847724322979 seconds** to the original 43,200-second
ledger. Cumulative usage is **9,543.745763106032 seconds**, leaving
**33,656.25423689397 seconds**. All 197 prior attempts remain unchanged, followed by
126 completed V7.6 jobs. No job has unsettled accounting. The [sealed ledger](accounting.json)
has SHA256 `5c621588a848912b5103f1aa724ac1028a790c249241ba6c891a0fd5eb21fd87`.

Actual `prepare-outer`, `develop`, `screen` and `finalize` commands all refused
with “Inner selection failed; V7.6 is closed.” They added no claims or charges:
the ledger hash, 323 attempts and 126 claims stayed unchanged. The
[validation record](validation.json) records these checks.

Existing competition file: `artifacts/submissions/rebuilt_v6_submission.csv`,
529,390 official rows, SHA256
`9320ca6465c5ce6576746c5fc8f00d13109e507d7a20d5d506a9e03f63fa2b85`.
The original baseline verification still passes. This is a rebuilt-V6 fallback,
not a V7.6 output or a reproduction of the lost historical submission. No
competition upload occurred. Future research requires a separately declared
hypothesis; this closed recipe must not be retuned on its inspected dates.
