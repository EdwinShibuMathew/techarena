# V7.5 result and verified competition submission

**The competition CSV is delivered. V7.5 was rejected during development.**
The recommended file is the verified rebuilt-V6 baseline, as requested for the
case where no V7.5 candidate passes every eligibility and promotion check.

File: `artifacts/submissions/rebuilt_v6_submission.csv`

```text
9320ca6465c5ce6576746c5fc8f00d13109e507d7a20d5d506a9e03f63fa2b85
```

The CSV has **529,390 rows**, the official header and example-submission IDs and
order, unique IDs, and finite probabilities in [0,1]. Two independent neural
inference/export runs and a third supported CLI replay produced matching CSV
hashes; member predictions also matched within 1e-15. A checksum and receipt sit
beside the file. The receipt explicitly identifies a rebuilt baseline, not a
promoted V7.5 or a reproduction of the missing historical V6 submission.

## Scientific result

V7.5 fitted new 47-feature CatBoost combiners, adding graph affinities, centered
affinities, item support, history/edge coverage and continuous prefix-age/size
features to the original 37 columns. The graph construction, neural schedules,
ensemble weights and 545-tree combiner parameters were fixed before fitting.
There was no hard age/date routing or public-leaderboard selection.

The three frozen variants were A (current-prefix fitting, gamma1.12), B (pooled
prefix fitting, no sharpening), and C (B's fitted models, gamma1.12). Pooled
training used real older-prefix predictions on each fitting day's first-user
cohort, with each event's replica weights summing to one. Models and graphs
predated the fitting labels. No additional post-fit calibrator was fitted.

| Variant | Jan22–24 gain vs rebuilt V6 | Gain vs rebuilt V7 | Jan24 gain vs rebuilt V6 | Gain vs rebuilt V7 |
| --- | ---: | ---: | ---: | ---: |
| A: current prefix, sharpened | +0.000306011 | +0.000037990 | +0.000036400 | **−0.000146431** |
| B: pooled ages | **−0.000154184** | **−0.000422206** | **−0.000334133** | **−0.000516965** |
| C: pooled ages, sharpened | +0.000158569 | **−0.000109453** | **−0.000175803** | **−0.000358635** |

The windows contain 47,957 and 35,040 first-user events. Their reference V6
vectors matched the preserved V7.4 reconstructed comparator within 1e-15.

| Variant | Aged 48–72h gain vs rebuilt V6 | Gain vs rebuilt V7 |
| --- | ---: | ---: |
| A | +0.000365523 | +0.000671726 |
| B | +0.000144517 | +0.000450720 |
| C | **−0.000139256** | +0.000166948 |

The aged subset contains 4,065 events. A recovered V7.4's aged regression, but
failed the forward comparison against rebuilt V7. B and C also failed declared
eligibility criteria. The frozen maximin selection returned no winner.

Primary/Jan26 screening, five-day testing and candidate finalization did not run;
no promotion confidence intervals were measured. There is no V7.5 ready
configuration or candidate submission. The recipe is closed without retuning,
runner-up screening or waiver of a comparator. All dates are reused,
retrospective evidence, not independent holdout or selection-adjusted guarantees.

## Assets, accounting and verification

The campaign verified and reused 25 inherited neural prefix models and trained
20 additional models: five full-prefix members for baseline delivery and five
members at each of Jan20 06:00, Jan20 12:00 and Jan23 00:00 UTC. The total available
neural inventory is therefore 45 models. Weights, raw data and prediction arrays
remain local and ignored by Git.

All **125 V7.5 jobs settled successfully**, using **3,509.8107641260576 seconds**
(58.50 minutes). Cumulative usage is **8,558.097210590051 / 43,200 seconds**.
The original limit and charges were preserved. V7.4's terminal accounting was
sealed, and its report still reproduces its original 5,048.286446463993 seconds.
The lost older historical ledger was not recreated.

Baseline training/export source: `ba2ac0389c64413982e0620fe162981d6ce775b6`.
Scientific source: `cf46c0900fab039a291d1c6516add0afac8f5bbe`.
[Source CI](https://github.com/EdwinShibuMathew/techarena/actions/runs/40725030908)
passed all four jobs. The required runner includes 95 model contracts, including
13 V7.5 contracts. Tests cover graph-feature oracles, causal replicas, weighting,
future-checkpoint refusal, all eight gate failures, accounting failure,
independent baseline replay, and synthetic end-to-end graph/combiner/bootstrap
and CSV behavior. Synthetic neural features and row counts are fixtures; those
tests do not establish scientific promotion. All 59 frozen source files remain
verified and historical baseline configurations remain unchanged.

After rejection, actual screening and finalization commands refused without
new claims or ledger changes. The delivered CSV's checksum sidecar passed.
No competition upload or automatic merge was performed.

- [Exact metrics, model hashes, source and accounting](results.json)
- [Recommended CSV and sidecar locations](delivery.json)
- [Engineering and terminal-guard evidence](validation.json)
- [Frozen experiment protocol](../../../experiments/v7.5-history-age-v1/RUN_V7_5.md)

Verify the existing delivery without fitting:

```bash
python experiments/v7.5-history-age-v1/campaign.py verify
```

To independently regenerate the baseline under the remaining shared budget:

```bash
python experiments/v7.5-history-age-v1/campaign.py replay-baseline
```

This replay verifies the existing file; it does not reopen V7.5 candidate research.
