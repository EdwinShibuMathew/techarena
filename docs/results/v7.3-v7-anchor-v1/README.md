# V7.3: accepted for the user-approved V7 upload trial

**Upload-ready under the explicitly approved V7-relative trial policy.** The
selected entropy-plus-mass correction improved primary and five-day AUC versus
fixed V7 with strictly positive paired 95% lower bounds. Jan26 also improved.
Late-window performance regressed and remains a documented risk. This is not
full promotion under the original four V6 gates; no competition upload or hidden
leaderboard improvement is claimed.

| Window | V7 AUC | V7.3 AUC | Change vs V7 | Paired 95% interval |
|---|---:|---:|---:|---|
| Primary Jan25–26 | 0.818701700370 | 0.818783336484 | +0.000081636115 | [+0.000026788450, +0.000136279733] |
| Secondary Jan26 | 0.827404252852 | 0.827532138948 | +0.000127886096 | [+0.000089680508, +0.000165693243] |
| Five-day Jan22–26 | 0.782086088781 | 0.782247612334 | +0.000161523553 | [+0.000038298539, +0.000272903676] |
| Late 48–120h | 0.739303785180 | 0.739051332063 | **−0.000252453117** | Point comparison; diagnostic |

Each interval used 1,000 valid paired whole-user bootstrap draws, seed 2026.
Primary/five-day required lower>0 and secondary required point change>=0.
Dates have been reused throughout research: these are retrospective measurements,
not independent holdout or selection-adjusted proof, and do not guarantee an
improved hidden competition score.

## Model and frozen selection

The [protocol](../../../experiments/v7.3-v7-anchor-v1/RUN_V7_3.md) fits a small
event-wide correction on fixed V7 gamma=1.12 scores. Candidates learn either
standardized-log-mass correction alone or entropy-plus-mass correction. Five
user-grouped cross-fitting folds provide fitting scores, excluding each held
user from temporary combiner training. Normalization uses fitting data only.
Bounds, optimizer, row-BCE/pairwise objective, L2 and seeds were frozen before
the run. Original V6 neural scorers, preferences, released combiners and
ensemble weights remain unchanged. No new neural model was trained.

| Candidate | Jan22–24 gain vs V7 | Jan24 forward gain vs V7 | Minimum |
|---|---:|---:|---:|
| Mass | +0.000160979875 | +0.000106883931 | +0.000106883931 |
| Entropy plus mass | +0.000160917466 | +0.000113926917 | **+0.000113926917** |

Both qualified; the frozen maximin rule selected entropy plus mass before
screening. Its aged-development change was −0.000449659653; the approved trial
policy recorded that diagnostic without blocking selection. There was no
post-acceptance tuning, runner-up promotion, date/age routing or public-score
selection. Previous V7.1/V7.2 decisions remain closed and unchanged.

The final model reuses the validated Jan25 calibrator with coefficients
c=−0.018350114739393953 and d=−0.021652154941973815. It preserves within-event
order and ties while changing event-wide scores. The fitting data and saved
normalization are bound to independent lineage and parameter expectations.

## V6 comparison and release semantics

All original V6 measurements remain present. Primary change was +0.000324083158,
Jan26 +0.000404426697 and five-day +0.000428526439, passing their original checks.
Late change was −0.000420048997, failing the original fourth gate. The trial
configuration therefore explicitly records upload_ready true, submission_ready
false and promotion_supported false. Existing/default policies still enforce
all four V6 gates; this new policy does not reinterpret old failed reports.

## Artifact, replay and accounting

The delivered file is `artifacts/submissions/v7_3_trial.csv`, also copied to
`C:\Users\Edwin\Downloads\v7_3_trial.csv` with checksum and receipt sidecars.
It contains 529,390 rows with official IDs/order, required columns and finite
probabilities in [0,1]. Its SHA256 is:

```text
0e48fad60f1d3bf92f7ddbfb4117725e9ccb10ad2432fa8b43f7996e51a5ef7b
```

Scientific source was frozen at `ce1eef8fbbbbbe8bc97c808c40ea50d0e9f5e933`.
Two fresh V6 replays matched the canonical CSV and every member vector exactly;
both fixed V7 anchor exports matched their canonical SHA256. Two independent
candidate computations and CSV exports matched exactly. Final accounting
completed before publication. A separately budgeted supported CLI replay then
reproduced the same new CSV hash, with another exact V6/V7-anchor reproduction.
The delivered Linux/Windows CSVs and sidecars were checked byte-for-byte.

Saved V6 vectors reproduced with zero maximum error on every measured cohort
with a historical reference. No within-event order or tie changed and all
scored probabilities were finite and in range. One hundred temporary fold
combiners, five development reference combiners and six calibrators were fitted.
Feature/OOF caching recorded 87 requests, 43 hits and 44 misses. No original model
asset or frozen source file changed.

New compute including the independent supported replay was 1,062.429163982
seconds; cumulative usage is 4,459.699446460001 of the original 14,400 seconds.
All phases finished within their 1,200-second caps and retained the existing
carry identity. Locked CPU validation passed 155 tests with one optional artifact
skip; all 42 required model contracts passed without skips. Source CI establishes
engineering checks separately from the measured scientific acceptance above.

- [Preparation/source identity](preparation.json)
- [Exact development report](develop.json), [screen](screen.json), [stress](stress.json)
- [Independent finalization](finalize.json) and [supported replay](supported-replay.result.json)
- [Decision, comparison metrics, integrity and complete accounting](decision.json)
- [Published trial configuration](../../../configs/v7_3_trial.json)

Reports, seals and ledgers are preserved byte-for-byte from the immutable run.
Weights, fitting arrays and submission CSVs remain private and untracked.
