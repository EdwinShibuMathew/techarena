# V7.3 V7-anchored upload trial

This is a new campaign with an explicitly approved upload-trial policy. V7.1
and V7.2 remain closed and unchanged. Original V6 models, released combiners,
preferences and ensemble weights remain frozen. No new neural training or
competition upload is automatic.

Hypothesis: a small event-wide residual learned on fixed V7 improves global row
ROC AUC beyond V7's gamma=1.12 sharpening alone. V6 metrics remain scientific
reference measurements; upload acceptance compares directly with V7. The fixed
V7 identity is bound to its configuration, validation report and canonical CSV
SHA256 9aebf08df53b9b1d0661479151c37118859831b3b4c9d371d533c3bf203b141a.

First apply the original fixed V7 map to each V6 probability vector. For that
anchor p, set S=sum(p), q=p/S, H=-sum(q*log(max(q,1e-12)))/log(10), and
L=(log(max(S,1e-12))-fitting_mean)/max(fitting_std,1e-6).
Output sigmoid(logit(p)+c*(H-.5)+d*L). Zero/one anchor endpoints stay exact;
zero coefficients return the anchor exactly. Zero-mass events use uniform q.
No exported clipping, model-age feature, cohort routing or learned sharpening.
Ordering and ties must be unchanged.

Two candidates, in tie order: mass (learn d, c=0), entropy_mass (learn c,d).
The four-slot saved coefficient format retains a=b=0. Coefficients are bounded
to [-.25,.25]. Use float64 L-BFGS-B at identity, maxiter 200, ftol 1e-12, gtol 1e-8,
maxls 50. Fit half mean-row BCE plus half mean global pair-logistic loss plus
.01*sum(active_coefficients**2). Sample 16 negative rows per positive with
seed 2026, including no-click-event negatives. Endpoint logits use loss-only
1e-15 stabilization. Non-convergence is terminal; no optimizer changes.

Five deterministic user-grouped folds keep all ten candidates together and
exclude held users from temporary combiner fitting. Preserve the exact original
CatBoost parameters, 545 trees and no early stopping. Apply the fixed V7 map
to cross-fitted V6 scores before fitting normalization and coefficients. Target
labels never reach fitting. Independently check saved normalization and lineage.

| Case | Fitting day | Target prefix | End, exclusive |
|---|---|---|---|
| jan22_long | Jan21 | Jan22 | Jan25 |
| jan24_forward | Jan22 | Jan24 | Jan25 |
| primary | Jan24 | Jan25 | Jan27 |
| secondary | Jan25 | Jan26 | Jan27 |
| five_day | Jan21 | Jan22 | Jan27 |

Dates are January 2025 UTC. Development eligibility requires both overall gains
against V7 >=0 and their mean >0. Select the largest minimum gain; ties within
1e-12 use the declared order. Aged development is diagnostic, not an eligibility
criterion. Freeze one winner before acceptance evaluation; no runner-up after
failure.

Upload acceptance: primary paired 95% lower>0 versus V7, secondary Jan26 point
gain>=0 versus V7, and five-day paired 95% lower>0 versus V7. Use 1000 whole-user
draws, seed 2026, >=990 valid draws. Report late 48–120h changes against V6 and V7,
without blocking this explicitly approved trial. Record all four original V6
gate outcomes separately. Reused dates are retrospective evidence, not
independent holdout/selection-adjusted proof or a hidden leaderboard guarantee.

The explicit release_policy=v7_upload_trial does not change omitted/default
V6 promotion behavior or historical reports. Published trials have upload_ready
true, submission_ready false and promotion_supported false. Preserve these
semantics in verification and supported replay.

Continue the existing shared ledger at 3397.270282478001 seconds consumed. Keep
its original carry identity and 14400-second cap; every phase has 1200 seconds.
Cache reuse requires compatible identities and complete checksum verification.
Freeze committed source and specification before budgeted preparation:

```bash
python experiments/v7.3-v7-anchor-v1/prepare_spec.py
# Commit all source and experiment.json.
python tools/run_experiment.py prepare --run-id v7.3-v7-anchor-v1 --spec experiments/v7.3-v7-anchor-v1/experiment.json
python tools/run_experiment.py develop --run-id v7.3-v7-anchor-v1
python tools/run_experiment.py screen --run-id v7.3-v7-anchor-v1
python tools/run_experiment.py stress --run-id v7.3-v7-anchor-v1
python tools/run_experiment.py finalize --run-id v7.3-v7-anchor-v1
python tools/run_experiment.py verify --run-id v7.3-v7-anchor-v1
```

Stop at the first failure; never retune or export an unqualified candidate.
After acceptance, reuse the saved Jan25 calibrator. Perform two fresh canonical
V6 replays, export/check the fixed V7 anchor SHA256, and independently calculate
the new CSV twice. Require identical arrays/CSV hashes, 529390 official rows,
correct columns/order and finite scores in [0,1]. Complete final accounting before
publishing configs/v7_3_trial.json and artifacts/submissions/v7_3_trial.csv.
Replay via `python tools/predict_postprocessor.py --release v7_3_trial` with a
fresh --output destination. Deliver the verified artifact, checksum and receipt
to the original project's submissions folder and Windows Edwin Downloads,
without replacing V7 or uploading to the competition.
