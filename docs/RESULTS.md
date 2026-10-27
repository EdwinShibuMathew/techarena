# Experiment register

I keep our experiment results here so we can compare changes without repeating
work. The original compact JSON reports are in `docs/results/` where available.
I treat a rejected recipe as evidence about that configuration, not proof that
its entire model family is ineffective.

## Submitted benchmark

V5 public: 0.819402. V6 public: 0.819451. Last reported V6 rank: 27; tenth-place
score in that snapshot: 0.826331. These are historical, not live standings.

## Research candidate: V7 (fixed power sharpening; not promoted)

| Evaluation | Metric / Change vs V6 | Paired 95% Bootstrap Interval | Gate Status |
|---|---:|---|---|
| **Primary (Jan25–26, 43,964 events)** | **+0.0002424470** | **[+0.0001509391, +0.0003354438]** | **PASSED** (Lower > 0) |
| **Secondary (Jan26, 35,138 events)** | **+0.0002765406** | **[+0.0001768461, +0.0003696035]** | **PASSED** (Nonnegative) |
| **Five-Day Stress (Jan22–26, 51,488 events)** | **+0.0002670029** | **[+0.0001555146, +0.0003699263]** | **PASSED** (Lower > 0) |
| **Five-Day Late (48–120h, 7,596 events)** | **−0.0001675959** | Point comparison only | **FAILED** (Change < 0) |

- **Method**: Within-event power sharpening ($\gamma=1.12$) preserving exact event click sums.
- **Submission Artifact**: `artifacts/submissions/v7_candidate.csv` (529,390 rows).
- **Candidate SHA256**: `9aebf08df53b9b1d0661479151c37118859831b3b4c9d371d533c3bf203b141a`.
- **Status**: Not promoted. Overall gains reproduce, but the late-window check fails.
- **Reproduced evidence**: [fixed protocol, source/input hashes, metrics and fingerprint verification](results/v7-validation/validation.json).
- Sharpening preserves within-event ordering; gains concern global row comparisons across events. These are reused development dates, not independent holdout evidence. No model fitting or competition upload was performed.

Our successful steps included preference learning, event training, exposure
adjustment, cold-history fallback, CatBoost combiners, longer preference
training, lower-rate fine-tuning, dimension/seed diversity and the small V6
global row-ranking update.

## V7.1 entropy-conditioned allocation (not promoted)

The fixed two-parameter postprocessor used user-grouped cross-fitted V6 scores,
retaining event score mass and within-event ranking. It passed both development
cases, primary/Jan26 screening and the five-day confidence gate. It failed the
late-window point gate; the recipe is closed with no final export.

| Gate | AUC change vs V6 | Paired 95% interval | Decision |
|---|---:|---|---|
| Primary Jan25–26 | +0.000253392017 | [+0.000135593223, +0.000375645545] | Passed |
| Secondary Jan26 | +0.000299762461 | [+0.000178194607, +0.000414059705] | Passed |
| Five-day Jan22–26 | +0.000208877797 | [+0.000023714297, +0.000373496981] | Passed |
| Late48–120h | −0.000483522408 | Point comparison | Failed |

No promotion-based tuning, fallback, cohort routing or new neural training was
performed. Finalization refused at the failed stress gate. New compute was
915.106481503s; cumulative research usage is 2,889.465828965005s of 14,400.
The dates remain reused retrospective evidence. See the
[protocol, source provenance and terminal decision](results/v7.1-entropy-sharpening-v1/README.md).

## V7.2 joint calibration (rejected during development)

All three predeclared calibration variants improved the two aggregate development
windows, but all regressed on the Jan22–24 aged 48–72-hour subset. No candidate
qualified, so the campaign stopped before promotion. All four promotion gates
remain unchanged and were not evaluated. No submission CSV was produced.

| Variant | Jan22–24 change | Jan24 forward change | Aged change |
|---|---:|---:|---:|
| Event correction | +0.000666948308 | +0.000474150119 | −0.000152856957 |
| Joint constant | +0.000726594348 | +0.000585951991 | −0.000084809510 |
| Joint entropy | +0.000720001722 | +0.000573566195 | −0.000087419660 |

The controller refused further stages before phase claims. New compute was
507.804453513 seconds; cumulative usage is 3,397.270282478001 of 14,400 seconds.
See the [frozen method and exact evidence](results/v7.2-joint-calibration-v1/README.md).

## V7.3 upload trial (accepted against V7)

The selected entropy-plus-mass correction on fixed V7 passed the user-approved
trial criteria. Primary and five-day paired 95% lower bounds against V7 were
strictly positive; Jan26 point change was nonnegative. Late change is reported
as risk and did not block this explicitly approved trial policy.

| Window | AUC change vs V7 | Paired 95% interval |
|---|---:|---|
| Primary Jan25–26 | +0.000081636115 | [+0.000026788450, +0.000136279733] |
| Secondary Jan26 | +0.000127886096 | [+0.000089680508, +0.000165693243] |
| Five-day Jan22–26 | +0.000161523553 | [+0.000038298539, +0.000272903676] |
| Late 48–120h | −0.000252453117 | Diagnostic point comparison |

The original V6 late gate still fails; full scientific promotion is not claimed.
Two independent final exports and a separately budgeted supported replay matched
exactly. The validated 529,390-row CSV was delivered to the project submissions
folder and Windows Downloads. New compute was 1,062.429163982 seconds; cumulative
usage is 4,459.699446460001 of 14,400 seconds. Reused dates remain retrospective,
not independent or selection-adjusted evidence; no hidden competition score is
claimed. See the [exact method, evidence and CSV fingerprint](results/v7.3-v7-anchor-v1/README.md).

## Archived V8 campaigns

[PR #4's preserved evidence](results/v8-archive/README.md) records six closed
campaigns. None passed every promotion gate or produced a submission-ready V8.
These results use repeatedly reused dates and are retrospective evidence.

| Campaign | Recorded stopping gate |
|---|---|
| PairLogit | Primary confidence interval lower bound not positive |
| Enriched Logloss | Primary confidence interval lower bound not positive |
| Temporal combiner search, selected B | Primary confidence gate failed |
| Seed44 row ranking | Five-day confidence gate failed despite passing screen and late checks |
| Robust ensemble, selected A | Primary confidence interval lower bound not positive |
| Residual event, selected B | Primary confidence gate failed |

The archive records all three valid tooling review findings and their supported
replacements in the merged shared workflow. Historical execution entrypoints
are excluded. The recorded cumulative compute is 1,974.359347462001 seconds of
14,400; archival work does not reset accounting or reopen rejected recipes.

## Recent rejected candidates versus V6

| Fixed experiment | Primary AUC change | Primary paired 95% interval | Jan26 change | Decision |
|---|---:|---|---:|---|
| exposure-matched preference, 85/15 expert blend | -0.0001047702 | [-0.0002941105, +0.0000950952] | +0.0001325543 | reject; no five-day run |
| time-bilinear seed42 replacement | -0.0000960311 | [-0.0001498893, -0.0000362835] | -0.0000791266 | reject; no five-day run |

Both passed reproduction and preservation checks. Neither produced a submission.

Earlier rejected/unsupported recipes: seen-history blending; joint combiner;
24 event epochs; dimension4; pooled-history correction; dim8-only history
integration; standalone attention history expert; all-event equal-user combiner
training; global implicit ALS blend; ALS cold-user routing; past-exposure
aggregate features. See the full handoff and historical plan for details.

## How I want us to choose the next experiment

I want us to audit the implementation and existing evidence before another trial.
We should rank at most three directions, then declare one hypothesis,
configuration, comparator, promotion gate and stopping rule. We keep new
experiments inside this repository and use the positive primary paired-interval /
nonnegative secondary gate followed by the fixed five-day comparison, while
acknowledging adaptive date reuse. We don't select weights against the public
leaderboard or relabel reused dates as an independent holdout.
