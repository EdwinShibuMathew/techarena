# V7.2 bounded joint calibration

This is a new three-candidate campaign, not a restart of V7.1. V6 is the
comparator. All four scientific gates remain unchanged. No competition upload
or merge is automatic. Original neural models, released combiners, preference
embeddings and ensemble weights are frozen. No new neural model is trained.

Hypothesis: learning small event-wide confidence corrections together with
conditional-score corrections can improve global row AUC across model ages.
Set S=sum(p), q=p/S, H=-sum(q*log(q))/log(10),
r=log(max(q,1e-12))-mean(log(max(q,1e-12))) and
L=(log(max(S,1e-12))-fitting_mean)/max(fitting_std,1e-6).
The map is sigmoid(logit(p)+(a+b*(H-.5))*r+c*(H-.5)+d*L).
Zero-mass events use uniform q; zero/one endpoints remain exact. Zero coefficients
return the input exactly. Exported probabilities are never clipped. Event mass
may change. Within-event order and ties must be unchanged, otherwise scoring fails.
No timestamp/age, cold-status or cohort routing is used by the adapter.

Exactly three candidates, in tie order: event (c,d); joint (a,c,d);
entropy_joint (a,b,c,d). Inactive coefficients are zero. Bounds [-.25,.25],
identity initialization, float64 L-BFGS-B maxiter 200, ftol 1e-12, gtol 1e-8, maxls 50.
Fit half mean-row BCE plus half mean sampled global pair-logistic loss plus
.01*sum(active_coefficients**2). Sample 16 negative rows per positive using
seed 2026, including negatives from no-click events. Loss-only endpoint logits
use probabilities 1e-15 and 1-1e-15; exported endpoints stay unchanged.
Non-convergence closes the campaign, without changing optimization settings.

Five deterministic user-grouped folds retain all ten candidates and exclude
every held-out user's events from temporary combiner fitting. Temporary CatBoost
copies retain original frozen parameters, 545 trees and no early stopping.
Normalization and coefficients fit only the declared metadata day, never target
labels. Save candidate parameters, normalization, coefficients and independent
lineage expectations together. Reuse only checksum-verified compatible caches.

| Case | Fitting day | Target prefix | Target end (exclusive) |
|---|---|---|---|
| jan22_long | Jan21 | Jan22 | Jan25 |
| jan24_forward | Jan22 | Jan24 | Jan25 |
| primary | Jan24 | Jan25 | Jan27 |
| secondary | Jan25 | Jan26 | Jan27 |
| five_day | Jan21 | Jan22 | Jan27 |

Dates are January 2025 UTC. Eligibility requires both development gains >=0,
their mean >0, and the Jan22_long late 48–72h gain >=0. Rank eligible candidates
by their minimum of these three gains; ties within 1e-12 use the declared order.
Seal one winner before promotion. If none qualifies, stop. Never screen a
runner-up after promotion failure.

Promotion requires primary paired 95% CI lower>0, secondary point gain>=0,
five-day paired 95% CI lower>0, and late 48–120h point gain>=0. Bootstrap 1000
whole-user draws, seed 2026, >=990 valid. Dates are reused retrospective evidence,
not independent or selection-adjusted validation. No public leaderboard tuning.

Continue the existing shared ledger and its original carry identity, verifying
the V7.1 terminal decision (2889.465828965005 seconds consumed at declaration).
The total cap remains 14400 seconds; every phase is capped at 1200 seconds.
No ledger copy/reset or historical receipt edits. Old runs replay through their
own immutable source snapshots after shared runtime changes.

```bash
python experiments/v7.2-joint-calibration-v1/prepare_spec.py
# Commit all source and experiment.json before preparation.
python tools/run_experiment.py prepare --run-id v7.2-joint-calibration-v1 --spec experiments/v7.2-joint-calibration-v1/experiment.json
python tools/run_experiment.py develop --run-id v7.2-joint-calibration-v1
python tools/run_experiment.py screen --run-id v7.2-joint-calibration-v1
python tools/run_experiment.py stress --run-id v7.2-joint-calibration-v1
python tools/run_experiment.py finalize --run-id v7.2-joint-calibration-v1
python tools/run_experiment.py verify --run-id v7.2-joint-calibration-v1
```

Stop at the first failure. No blending, fallback, sweep, gate change or
post-promotion tuning. After all gates pass, reuse the saved Jan25 calibrator,
run two fresh canonical V6 replays and independently compute both candidate CSVs.
Require identical arrays/hashes, 529390 rows and official IDs/order/columns.
Complete accounting before publishing configs/v7_2.json and
artifacts/submissions/v7_2.csv. Independent supported replay:

```bash
python tools/predict_postprocessor.py --release v7_2 --output artifacts/submissions/v7_2_replayed.csv
```

The ready configuration, checksum and receipts establish reproducibility and
passing declared gates, not guaranteed improvement on hidden competition labels.
