# My TechArena research notes

I keep these notes from my WSL runs and leaderboard snapshots as of 2026-10-24.
They record the experiment sequence, including preparation and decisions at
each stage. Earlier pending steps describe their status at the time; they are
not a new instruction to repeat completed or rejected experiments. Our current
setup and reproduction steps are in [the README](../README.md).

I run experiments in WSL and review the evidence before choosing the next one.
These notes are a research record, not a promise of an optimal model or score.

## Current baseline

Time-bilinear trial completed and rejected. Primary V6 0.8184592533 ->
0.8183632223, change -0.0000960311, paired interval
[-0.0001498893, -0.0000362835]. Secondary change -0.0000791266 to
0.8270485856. Both global point changes and every exploratory time-bucket
point change were negative. V6 replay and combiner reload checks passed;
four retained members were exact; V5/V6 files unchanged. No CSV or upload.
Do not run the five-day comparison or full-data fitting for this recipe.
The compiler's _POSIX_C_SOURCE redefinition warning did not stop this run:
real CPU/CUDA model and loss self-tests passed, followed by completed fitting
and reporting. Keep the working environment unchanged for this warning alone.
V6 remains my strongest recorded public submission (0.819451).

Earlier prepared screen (now completed and rejected): time_bilinear_model.py / run_time_bilinear_trial.py.
New workspace techarena-time-bilinear-trial; actions self-test/setup/train/
evaluate. Hypothesis: V6 has a time-independent user-item dot product plus
separate additive user/item context effects. Add u^T Delta[time_bucket] v,
with four zero-initialized 8x8 matrices (256 shared parameters). Matrices are
pooled across weekday/weekend so the pre-weekend prefixes can fit every new
context; original eight additive context effects remain unchanged. Use the
existing seed42 preference epoch260 checkpoints, the V6 mixed-objective event8
updates and all original penalties, plus 0.05*mean(Delta^2). Zero initialization
consumes no RNG and starts at the original model. No preference retraining.

Train three prefix event models, fit two honest 545-tree/37-feature combiners,
and replace only dim8 seed42 at its original 2/9 weight. All other four V6
member probabilities and all ensemble weights are reused exactly. The usual
exposure correction and ridge cold-history fallback stay in place; a private
inference function loads the new architecture without modifying V5/V6 globals.
Setup replays V6 and freezes inputs plus protected V5/V6 artifacts. Require a
positive primary global paired interval and nonnegative Jan26 gain before the
same fixed recipe's five-day paired comparison versus V6. No full-data fitting,
weight search, CSV or upload is included. Dates remain reused retrospective
development data; intervals are not selection-adjusted.

Research motivation: user-item-context tensor factorization, not a claimed
replication or promised gain: https://doi.org/10.1145/1864708.1864727 .
Local independent NumPy score/gradient oracles, exact source-update AST checks
and synthetic orchestration passed: atomic setup, three stage handoffs, honest
two-day meta fits, four retained members exact, completed resume and protected
V6/prediction tamper rejection. GPU/training/CatBoost and Windows symlinks were
stand-ins. WSL self-test checks real CPU/CUDA zero-model identity/RNG, score
oracle and finite-difference gradients, inactive users/day-type pooling, then
runs the generated event8 trainer and cold/empty-history pipeline on a tiny
fixture. At that stage, I still needed to run the actual CUDA handoff, V6 replay
and real fitting in WSL.

I completed the exposure-matched preference trial in WSL and failed the
declared promotion gate. Against V6: primary 0.8184592533 -> 0.8183544831,
change -0.0001047702; paired 95% percentile interval
[-0.0002941105, +0.0000950952]. Secondary 0.8271277123 -> 0.8272602666,
change +0.0001325543. The primary interval crosses zero, so the secondary
point gain does not rescue the fixed candidate. Correctness passed: V6
replay differences zero, saved expert combiners reproduced, all five V6
members retained, V5/V6 files unchanged. No submission or upload occurred.
Do not advance this fixed 85/15 candidate to the five-day comparison or
full-data training, and do not perform a post-hoc blend-weight sweep. This is
an unsupported near-tie, not proof that all exposure-matched methods fail.
V6 remains my best recorded public submission at 0.819451.

My latest recorded public result: V6 candidate uploaded, AUC 0.819451,
rank 27 in the supplied screenshot. V5 was 0.819402: improvement 0.000049.
Screenshot tenth is 0.826331, leaving a 0.006880 public gap. Other teams'
improvements explain why a higher score accompanies a worse rank. This is a
snapshot, not a verified live cutoff or a prediction of private ranking.

My full-data V6 checks passed in WSL: 529,390 unique example IDs
in order, valid probabilities, both validation replays exact, three retained
members reused exactly, and protected V5 files unchanged. Candidate SHA256:
c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c.
V6 is the current best reported public submission; retain V5 and V6 artifacts.
Earlier preparation notes below describe the state when they were written.

Earlier bounded screen (completed and rejected, results above): an independent
exposure-matched negative-sampling preference expert. Current preference
training samples 31 uniform item comparisons; this hypothesis instead samples
31 items using q(i)=(past candidate displays(i)+20)/(total displays+20*n_items),
with every count strictly before the relevant prefix cutoff. Warm-up display
frequencies are unavailable, so this is an explicitly imperfect approximation.
Retain the existing preference positive examples and dimension8 seed42,
196+64 preference schedule, and the V6 seed42 mixed-objective event8 recipe.
Disable the post-hoc exposure adjustment for this expert (power0) as part of
the coherent new recipe; retain its raw/pipeline feature slots and refit its
545-tree/37-feature combiner honestly on the preceding meta day. This tests
representation learning and exposure treatment together, not a one-factor
attribution experiment. Keep the five V6 members untouched. The single planned
candidate is 85% V6 plus 15% new expert, on every candidate row. No sampler,
blend-weight or exposure-power sweep after viewing outcomes.

First reproduce cached V6 in both existing development cases. Require a
positive global primary paired-gain interval and nonnegative Jan26 gain,
then compare the identical candidate against the saved V6 five-day predictions
using the earlier-cutoff protocol. Require the existing overall/late-window
gates and all correctness checks before full-data fitting or submission.
All these dates remain reused retrospective development data; intervals are
not selection-adjusted. This is a reasoned next hypothesis, not an established
best model or a promised large gain. Do not expand to more seeds before the
initial fixed integration passes. Research context for sampling choices:
https://doi.org/10.1145/3522672 ; benchmark findings do not establish lift here.

Prepared exposure_matched_preference.py and run_exposure_matched_trial.py for
the separate techarena-exposure-matched-trial workspace. CLI self-test/setup/
train/evaluate. Setup replays V6's two historical cases, reconstructs V5 and
verifies the full V6 manifest, then freezes existing inputs and protected V5/V6
artifacts. New weighted preference models are trained from scratch at Jan24,
Jan25 and Jan26, with original minibatch updates preserved by AST comparison
apart from the sampling expression. Stage196 saves Adam state; fine197-260
retains it at LR0.003 and resets the declared sampling seed1042. Generated
training sources check frozen hashes. The event8 updates are the successful
V6 mixed-objective source unchanged; checkpoint saves gain provenance metadata
and atomic writes. Three prefix sampling tables use an exact alias sampler
for efficient independent replacement draws from smoothed display counts.

Private inference-function namespaces disable only the expert's exposure
adjustment, preserving original V5/V6 module globals. Two new 545-tree expert
combiners use honest Jan24/Jan25 meta features; a fixed 85/15 blend compares
against V6. The report includes 300 whole-user paired resamples and the original
primary/secondary gate. All output is confined to the new trial; no submission
is created. Completed setup/training/evaluation can be verified and resumed.

Local numeric/source checks and synthetic orchestration passed: strict cutoff,
future-query independence, exact alias mass, original update ASTs, V6 cache
reconstruction, transactional setup, all nine stage handoffs, honest two-day
meta fitting, completed resume and protected-file/prediction tamper rejection.
GPU/training/CatBoost and Windows symlinks were stand-ins locally. The WSL
self-test checks actual CPU/CUDA sampling and row-ranking gradients, executes
all generated stages on a tiny fixture using the full fixed epoch schedule,
and checks compatible cold/empty history inference. At that stage, I still needed
to run real training and reproduce the historical predictions in WSL.

Five-day global row-ranking check completed and passed: overall V5 0.7817113214
-> candidate 0.7818190859, gain 0.0001077645, paired interval
[0.0000231960, 0.0001935632]. Late 48-120h gain 0.0002384174 was nonnegative;
its separate interval crossed zero, which does not fail the declared late
point-estimate gate. All five daily point gains were positive. All five original
baseline members replayed exactly after the report-hash reconciliation, and
new combiners reproduced after reload. Reused dates/selection limitations remain.

I decided to proceed with full-data training and candidate CSV creation while
keeping V5 intact; he continues to execute commands in WSL and upload himself.
Prepared row_auc_final_utils.py and run_row_auc_final.py. CLI actions:
self-test, setup, train, build. New workspace techarena-row-auc-final. Reuses
the full-data fine-tuned preference epoch260 checkpoints for seeds42/43 and
fits only two event8 models using the frozen mixed loss. The generated trainers
are adapted from the successful Jan26 sources; an AST check requires the entire
optimizer update block, pack function and model classes to remain identical.
Validation loading/scoring is removed, and all 686,824 labeled fitting events
are used before the Jan27 cutoff. Training inputs contain no evaluation data.

The build uses the short trial's existing dim8_seed42_jan25.cbm and
dim8_seed43_jan25.cbm, with no combiner fitting or weight search. It reproduces
both short-window validation cases before evaluation prediction. The other
three V5 cached evaluation members are reused exactly. It reconstructs the
existing V5 probabilities from all five cached members, checks every original
V5 CSV row/probability by rowid and its recorded SHA, then freezes V5 CSV,
prediction caches, metadata and old full checkpoints. All source files are
verified again before and after export. Candidate output:
techarena-row-auc-final/outputs/techarena_submission_v6_candidate.csv, with
candidate_submission_checks.json and candidate_submission_manifest.json.
No upload or change to V5 artifacts is performed. New model/prediction/CSV
hashes and query-cache fingerprints support resume and changed-file detection.

Local checks passed: both final trainer adaptations parse and preserve the
tested update block; exact CSV alignment works with a differently ordered
example. Synthetic final lifecycle covered V5 reconstruction, frozen setup,
two separate fits, retained members, export/manifest verification, completed
and interrupted-export resume, protected V5 bytes and CSV tamper rejection.
CUDA/training/inference/prior-study checks and Windows symlinks were stand-ins
in that local workflow. I still needed to complete full training and validation
replay in WSL at that stage. The candidate is an incremental update, not a predicted
top-ten score; no new public result has been reported.

Five-day setup initially stopped at the cached training-report SHA comparison,
before creating the new trial or training. Likely cause found in the earlier
baseline trainer: every train action rewrites training_results.json with a new
runtime, even when checkpoints are reused, making prior cached report hashes
stale. This cause is not assumed from the traceback alone. Updated
run_row_auc_five_day.py requires direct prediction replay for every member
whose cached report hash differs, using current checkpoint, prefix data,
original dimensional pipeline and saved combiner. Prediction tolerance stays
rtol=0/atol=1e-15; combiner hashes and all original raw/model/source checks stay
strict. The script records both report references and replay differences, binds
the verification to a fingerprint of all frozen input hashes, and rechecks those
bytes before accepting the new setup. Later evaluation reuses only that exact
frozen proof. No original report/cache hash is overwritten or relabeled.

Local regression passed for a runtime-only rewritten report, exact replay of
all five members/both dimensions, rejection of missing/stale proof, changed
dim8/dim16 predictions and changed combiner bytes. GPU inference was a stand-in.
I needed to copy the corrected script and rerun self-test/setup/train/evaluate;
no deletion or rerun of the original baseline trainer is needed.

The short global row-ranking trial passed its declared development gates:
primary V5 0.8183722315 -> candidate 0.8184592533, gain 0.0000870218,
paired interval [0.0000260274, 0.0001542581]; secondary gain 0.0001336187
to AUC 0.8271277123. Old-member reproduction differences were zero and new
combiners reproduced after reload. Dates are reused and intervals are not
selection-adjusted; this supports the next stress test, not promotion.

Prepared run_row_auc_five_day.py for my WSL run. It reuses the completed
techarena-v5-long-window preference epoch260 checkpoints for dim8 seeds42/43
at Jan21 and Jan22, and copies the successful short trial's generated trainers.
Only monitoring-split names and checkpoint cutoff metadata change; restoration
checks require all other trainer text to stay identical. Four event8 fits and
two 545-tree/37-feature combiners are created in techarena-row-auc-five-day.
Meta labels are first-user Jan21 events; targets are the same 51,488 first-user
events across Jan22-26 as the saved baseline. Seed44 and both dim16 predictions
and all original V5 weights are reused exactly. No preference/full-data training
or submission export. Both earlier studies' source/cache/checkpoint hashes and
the saved five-member baseline are verified before the new fitting.

Frozen five-day gate: overall global paired gain interval entirely positive,
pooled 48-120h (Jan24-26) global AUC change nonnegative, and all reproduction
checks pass. Use 1,000 paired whole-user draws, identical for both models;
report overall/late intervals, daily point results and exploratory groups.
The late point gate guards against an early-window gain hiding a late decline;
daily/group outputs are not candidate-selection rules. Passing supports a
full-data review, not automatic promotion or a qualification guarantee.

Local source/self-tests and a synthetic end-to-end lifecycle passed, covering
four exact recipe adaptations, old evidence validation, temporal pairing,
four event fits/two honest meta fits, serialized predictions, retained members,
ties/explicit-duplication bootstrap oracle, report/cache resume and checkpoint
tamper detection. CUDA/training/CatBoost and Windows symlinks were stand-ins.
A Windows/WSL line-ending mismatch found in testing was fixed by comparing
normalized Python text while freezing each input's original bytes separately.
At that stage, I still needed to train the models and reproduce V5 predictions in WSL.

Next prepared experiment: row_auc_objective.py / run_row_auc_trial.py changes
the event-stage data objective for dim8 seeds42/43 to 0.5 original eleven-way
event cross entropy + 0.5 sampled global row-pair logistic ranking loss.
Each clicked fitting row samples 16 unclicked rows from the fitting minibatch,
including other users and no-click events. Pair scores are log odds of the
original softmax candidate marginals, computed using an explicit sum over the
other outcomes for numerical stability. A separate generator seeded seed+3000
preserves the original event-shuffle random stream. This is a ranking surrogate,
not exact AUC optimization or standard per-user BPR. Pairwise-ranking research
motivates the hypothesis, not a prediction of improvement here:
https://arxiv.org/abs/1205.2618

Reuses V5's dim8 preference epoch260 checkpoints; original architecture,
event8 schedule, Adam .0015, anchor .1 and effect .05 regularization,
global-offset initialization, 37 feature pipeline and 545-tree combiner recipe
are unchanged. The sole candidate replaces only dim8 seeds42/43; seed44 and
both dim16 members retain their exact predictions and all V5 weights remain.
Six new event fits and four new combiners live in techarena-row-auc-trial.
No full-data training, submission export, weight search or extra epochs.
Use the same primary positive global paired interval and nonnegative secondary
global AUC gate, followed by a five-day paired test before promotion.

Local numeric checks passed with a NumPy-backed tensor stand-in against an
independent explicit-sum oracle, including ordinary/extreme scores and finite
differences. All six source adaptations preserved model classes, regularization
and final-epoch guards. A synthetic filesystem lifecycle tested setup/frozen
hashes, six trainer calls, four honest meta fits, baseline reconstruction,
unchanged three members, model reloads, bootstrap, resume and changed-output
detection. GPU/autograd/CatBoost and Windows symlinks were stand-ins locally.
The WSL self-test checks real CPU/CUDA autograd against finite differences,
then runs the generated trainer and original inference on a tiny cold/empty
history fixture before the real experiment.

Past-exposure feature trial completed and failed: primary change -0.000006354
with paired interval [-0.000070690, 0.000053979], secondary -0.000003463.
All old-member predictions reproduced exactly and new model serialization
passed. This is an unsupported near-tie, not a meaningful proven regression.
No five-day test or submission; close the fixed feature/blend candidate.

Previously prepared experiment: run_exposure_trial.py with exposure_features.py adds
16 explicit past-impression summaries to each V5 combiner (53 total). They
cover prior user-item displays/clicks/non-clicks, time since last exposures,
recent-24-hour user/item activity and smoothed click rates, and user/item
six-hour-bucket counts/rates pooled across day types. Existing V5 neural
models already learn from non-clicks in event training; this tests explicit
conditional summaries, not previously unused negative labels. All aggregates
and smoothing priors use only observations strictly before each prefix cutoff.
Targets query these frozen tables; they do not update them. No impression is
treated as explicit dislike, and warm-up impressions are unavailable.

Ten new 545-tree combiners use the original first-user meta events on Jan24
and Jan25, original settings and five member weights. The sole candidate is
50% unchanged V5 + 50% augmented ensemble. No feature subset/weight search is
planned. Reproduce every old member prediction first, check new serialization,
and require positive primary global paired interval plus nonnegative Jan26
global change before an earlier-cutoff five-day paired comparison. All new
artifacts live in techarena-exposure-trial; no full-data fitting or submission.

Local checks passed: all 16 features match an independent row-by-row oracle;
future labels cannot alter them; exact cutoff, recent-window boundary, empty
recent activity, unknown keys, repeated targets and fitting/target order are
covered. A synthetic filesystem run checked three cutoffs, five members, ten
meta fits, saved-model predictions, fixed blend, bootstrap, resume and changed
output detection. GPU/CatBoost were stand-ins in that local workflow; I still
needed to run actual fitting and reproduce predictions in WSL. The general
motivation for examining observed non-clicks is discussed in Lin, Kamar and
Horvitz, "Signals in the Silence":
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/LinKamarHorvitzCorrected.pdf
Its results do not establish improvement for this synthetic competition.

The ALS cold-history-only routing follow-up also failed: primary global
change +0.000042261 with interval [-0.000008606, 0.000082021], secondary
-0.000013498. Correctness passed, performance did not. Do not run five-day
checks or submissions for either ALS integration, and do not search more
ALS weights or subgroups on these same reused dates.

Latest follow-up: the fixed 85% V5 / 15% ALS global blend failed. Primary
change -0.000229729 with paired interval [-0.000457797, 0.000064277];
secondary change -0.000246723. Correctness and saved-model reproduction
passed. Do not run its five-day comparison or create a submission.

Previously prepared: check_als_cold_routing.py tested ONE new routing hypothesis from
the existing cached predictions. Use the same 15% ALS weight only when the
user has zero fitting events before the case cutoff and nonempty target
history. All other probabilities stay exactly V5. This is a post-hoc follow-up
motivated by the cold-history subgroup point gain, not independent evidence.
It performs no training, GPU work, threshold/weight search or export. It
reconstructs V5 from all five saved members, reproduces the failed global ALS
blend, verifies source/checkpoint hashes, and bootstraps GLOBAL row AUC over
all first-user events so cross-group comparisons are included. Only a wholly
positive primary paired interval and nonnegative secondary global change
advance this routing rule to an earlier-cutoff five-day paired test. Keep V5
until that next gate passes. The rule affects only a small part of the final
population; it is not expected to close the full top-ten gap by itself.
Local checks passed: numeric self-test against an independent pairwise AUC
oracle (including tied scores and explicit bootstrap event duplication), and
a synthetic filesystem run covering both windows, five-member reconstruction,
old ALS blend reproduction, source hashes, resume and output tamper detection.
The actual competition run has now completed and failed as recorded above.

- Team: ChaiScript; screenshot score 0.819402, rank 21.
- Public gain over V4: 0.000803.
- Gap to 0.82: 0.000598. To exceed 0.82, the gain must be greater than this.
- Leader in the screenshot: 0.831991; gap 0.012589.
- Tenth place in the screenshot: 0.826112; gap 0.006710.
- My qualification objective: finish in the top ten.
- The Phase 1 deadline I was working toward: November 3, 2026, end of day.
  The deadline's official timezone and the qualification selection rule still
  need to be checked in the competition portal. Plan completion before that day.
- V5 has three fine-tuned dim8 seeds (42, 43, 44), each with weight 2/9,
  and the original two dim16 seeds (42, 43), each with weight 1/6.
- V5 development AUC: 0.8183722315263706 on the two-day primary check;
  0.8269940935263441 on the January 26 secondary check.
- Keep V5's CSV, individual predictions, encoder files, checkpoints, combiners,
  protocol and manifest as the reproducible reference.

## Objective and constraints

The objective is top-ten qualification, not simply passing 0.82. Tenth place is
a moving cutoff; 0.826112 is only the latest screenshot's reference. No fixed
public score guarantees a qualifying private rank. Improve the information
used for prediction, rather than add more replicas of the same model. Run
each model idea through defined stages. The final competition ranking uses private global row AUC;
the public and private sets are random partitions of evaluation rows. The
supplied INSTRUCTIONS.md limits submissions to ten per day.

I use my existing techarena conda environment in WSL. Hardware: RTX 4070
Laptop GPU with 8 GB VRAM, about 15 GiB WSL RAM. No new downloads are planned
for the first experiment; NumPy, SciPy, PyTorch and CatBoost are already available.
Keep experiments in a separate WSL project folder and leave V5 intact.

## 1. Establish the failure pattern and feasibility

Use the cached V5 out-of-time predictions and labels. Report global AUC,
counts, click rates and V5-minus-V4 gains by:

- First target day and elapsed time from the fitting cutoff.
- Seen versus unseen users, fitting activity 0 / 1-5 / 6-15 / 16+ events.
- Empty versus nonempty history, history length, and last-click age.
- Repeat versus nonrepeat candidates and candidate exposure.

Keep the original first-event-per-user selection. Use whole-user/event paired
resampling when comparing changes. Check the sampling variability before
treating a small group's AUC as evidence for a new model.

Add a small item-pair coverage diagnostic. Count support from fitting clicks
only and distinguish pairs supported by one user from pairs supported by
several users. Separate identical-item repeats from cross-item relationships.
This tells us whether explicit item-pair counts are plausible extra features;
it does not by itself establish predictive value.

Use the existing preference_training.npz click records as the starting source.
Do not count the same history twenty times because it appears in repeated
impression events. Preserve genuine repeated clicks. Do not fit aggregate
statistics from target histories or target labels.

Deliverable: one diagnostic report with the largest credible error groups,
pair coverage, data sizes, and the chosen first model configuration.

## 2. Main new hypothesis: learn a history-dependent correction

Audit decision (2026-10-24): V5's improvement over V4 is supported on both
development checks. Primary other-user pair coverage is 22.92% for at least
one supporting user but only 0.97% for at least three. Evaluation short-history
frequency is 16.37%, versus 9.27% on Jan26, and evaluation's 90th-percentile
horizon is 70.25 hours. Prioritize a learned history model that shares parameters
across items; explicit pair counts remain a possible auxiliary signal.

The first implementation is now frozen in pooled_history_model.py and
run_history_trial.py: trainable 8-dimensional history item embeddings initialized
from the frozen dim8 seed42 event model, mean/recent/last pooling, 24-hour
recency decay, a 32-unit hidden layer, zero-initialized output correction,
neutral correction for empty histories, eight epochs, Adam at 0.001, seed2026,
anchor regularization0.02 and correction regularization0.01. The resulting
candidate probability is the 38th feature in each of the five combiners.
V5 ensemble weights and 545-tree combiner settings stay fixed.

Question: can the latest history improve a candidate's score beyond the
learned user profile and the current hand-built history summaries?

Start with a small history component on top of the frozen, fine-tuned dim8
seed42 event model in each fitting prefix. The component sees the candidate,
the supplied history items and their ages. It learns an adjustment to the
candidate logits. Preserve the eleven-outcome event objective: no click or
one of the ten candidates. Train on complete fitting events, including no-click
events, using only each event's earlier history.

Use the existing item vectors to initialize or supply the history representation.
Start with compact pooling and recency information. Candidate-dependent
attention is a possible single follow-up if diagnostics justify it. Record
whether item vectors are frozen or trainable before the first fit; this choice
must not drift between folds or between local and final training.

Initialize the final correction layer to zero. At initialization, assert that
scores match the frozen base model. This is a correctness check, not a claim
that later training cannot reduce AUC. Empty histories should produce a
well-defined neutral history correction, and supplied histories should allow
the component to operate for users absent from fitting data.

Freeze the architecture, optimizer, learning rate, regularization, training
length, sampling seed and integration rule before inspecting its validation AUC.
Use one neural seed for this first architectural test. Avoid a broad search.

Deliverable: new neural checkpoints and out-of-time scores for the Jan24,
Jan25 and Jan26 fitting prefixes, with provenance and epoch-zero checks.

## 3. Integrate new information while retaining the V5 ensemble

Use the new model's candidate probabilities as one additional feature in each
of the five existing per-member combiners. Keep the original 37 features,
the 545-tree combiner settings, and the five V5 ensemble weights. This compares
V5 with V5 plus the same new information, without also replacing the ensemble
by one weaker standalone member or reintroducing the failed joint combiner.

Combiner fitting features must come from neural models trained strictly before
the meta-training day. Jan24 features fit the primary combiners; Jan25 features
fit the Jan26 secondary combiners. No combiner may train on its own evaluation
labels. No inference-time rule or probability blend is selected on the public
leaderboard.

Screen on the current development protocol:

- Primary: first event per user across Jan25-26; combiners fit on Jan24.
- Secondary: first event per user on Jan26; combiners fit on Jan25.
- Paired whole-event bootstrap: 300 resamples, seed 2026.
- Advance only if the primary gain interval is entirely positive and the
  secondary gain is nonnegative. Inspect major cohorts for large regressions.

These dates have been reused many times. The intervals describe the tested
comparisons and do not correct for the history of model selection.

## 4. Require a longer future-window stress check before submission

For a candidate that passes the cheaper screen, freeze it and construct a
five-day target window, Jan22-26. Select each user's first event in that whole
window, rather than one first event per day.

- Fit neural models and global statistics strictly before Jan22 to produce
  target-window features.
- Fit combiners on first events from Jan21 using neural features from models
  fitted strictly before Jan21.
- Rebuild the V5 baseline under these same earlier fitting cutoffs. The
  baseline needs all five members with their original respective schedules.
- Evaluate the new history component under the same cutoffs and compare paired
  predictions. The absolute AUC should not be compared directly with the old
  Jan26 number because fitting data and target population differ.
- Target features may use each target event's supplied prior history, including
  clicks after the global fitting cutoff but before that target event. Global
  fitted statistics and model updates must remain before the fitting cutoff.
- Require a positive paired gain interval on this stress check before promotion.
  If it fails, diagnose; do not repeatedly tune on the stress window and call
  it an untouched holdout.

This remains retrospective evidence. All available dates have influenced prior
development in some form; the stress check is not a new independent dataset.

## 5. Promote one frozen candidate

Only after the checks pass, fit the new component on all training events,
reuse the selected out-of-time-trained combiners, reproduce the saved
development predictions, and export the next version with the validated CSV
writer. Verify IDs, shape, probabilities and the actual five-member weights.

Make one public submission. Record its score alongside the exact protocol and
retain V5 for comparison. Do not project a local AUC gain directly onto public
AUC or predict a finishing rank from the development checks.

## Schedule to the November 3 deadline

These are work windows and priorities, not runtime estimates. Available daily
time has not yet been specified. I run the WSL
commands and upload selected submissions myself.

| Dates | Priority | Exit criterion |
| --- | --- | --- |
| Oct24-25 | Preserve V5, run the group/pair-coverage audit, establish the earlier-cutoff V5 reference for the five-day stress window | Baseline reproducible; failure groups and data support understood |
| Oct26-28 | Test the compact history correction; allow one justified candidate-dependent history variant; run a sparse pair-feature experiment only if its coverage supports it | At least one candidate shows a supported improvement or the hypotheses are explicitly rejected |
| Oct29-31 | Confirm the strongest candidate on the longer window; test complementarity between the best genuinely different signals | A frozen candidate improves beyond V5 on the main development checks and the stress window |
| Nov1-2 | Freeze architecture and weights; complete full-data training, prediction reproduction and a selected public submission | Reproducible strongest supported submission, with checkpoint and CSV backups |
| Nov3 | Confirm the chosen final submission and acceptance status under the portal's actual selection rules | Correct selected/accepted artifact before the official cutoff |

Start the cheap history-model screen while the broader baseline is being
prepared if waiting for all earlier-cutoff baseline models would delay useful
work. Apply the longer-window promotion gate before another public submission.
Do not spend the final day on a new unverified architecture.

If the first history hypothesis fails, reserve the next test for an alternative
with a concrete reason to add information: candidate-dependent history weights,
or sparse item relations when coverage is adequate. More seeds remain a
finishing step after a stronger model is established, not the main route to
closing the current 0.006710 public gap. Set each trial's configuration and
integration before inspecting its outcome; avoid an open-ended parameter sweep.

## Scope and stopping decisions

- First round: one history architecture, one seed and one integration rule.
- If the initial screen fails, use its training and group diagnostics to decide
  whether one targeted repair is justified. Do not sweep unrelated models.
- If the history component has no supported benefit, revisit explicit sparse
  item-pair features only when their coverage diagnostic supports a trial.
- My stated deadline is November 3 EOD; confirm its timezone and how
  the qualifying/final submission is selected in the portal. Keep the workload
  modular until I know how much time I can spend each day. If little time remains,
  prioritize a reproducible strongest submission and a single cheap screen
  over rebuilding a large evaluation suite.
- A full 50,000-by-50,000 float64 matrix alone is 20 GB (about 18.6 GiB), before
  solver workspaces. Dense EASE is therefore not the next implementation for
  the current 15 GiB WSL setup. This is a resource choice, not an accuracy claim.

## Research basis

### October 24 experiment update

Expanded-combiner results: correctness checks and exact feature/prediction
reproduction passed. Primary AUC change was -0.0000823, paired interval
[-0.0002379, +0.0000648]; secondary +0.0000881. The primary gate failed, so
this remains an unsupported near-tie and does not advance to the five-day check.

Next prepared experiment: implicit_als_model.py and run_als_trial.py learn an
independent 16-dimensional factorization using confidence-weighted implicit
least squares. Configuration: confidence 1+40*log1p(pair click count), L2 0.1,
30 alternating iterations, seed2028, factor dot scores scaled by8 before the
original fixed eight-epoch event stage. Normal equations accumulate sparse
edges in chunks; no full users-by-items matrix is allocated. New first-user
545-tree/37-feature combiners are fitted on Jan24 and Jan25. The single fixed
integration is 85% original V5 plus15% ALS expert on all rows. Existing V5
members stay intact. Primary positive paired interval and nonnegative Jan26
change are required before an earlier-cutoff five-day test. This is a new
representation objective, not a promised leaderboard gain. CPU algebra/source
and synthetic orchestration checks passed; actual CUDA solves and an
end-to-end preference/event/feature self-test are included for my WSL run.

Matched comparison completed: later-cutoff V5 AUC 0.758884 versus 0.733914
on the same 3,531 late-arriving events, gain 0.024970 with paired interval
[0.018383, 0.031005]. Both days, seen users, unseen users with history and
short histories improved. Fitting activity was unchanged for matched users;
the result therefore concerns changes in the shared fitted pipeline, not new
observed events for these particular users. It does not isolate embeddings,
statistics or combiner fitting, and this subset is unusually cold (41.6%).
The full-data V5 already uses the full seven days; this diagnostic gain is not
an available additional leaderboard gain from merely repeating training.

Recommended next hypothesis: expand combiner training from first events per
user to all events in the same eligible meta-training day, using event weights
inversely proportional to that user's event count on the day. Keep the five
neural models, 37 features, combiner settings and V5 ensemble weights fixed.
Generate meta features strictly from models/statistics fitted before that day,
and handle repeated cold-user targets per event rather than overwriting one
user embedding with multiple target histories. Primary: fit on Jan24, score
first-user Jan25-26. Secondary: fit on Jan25, score first-user Jan26. Only a
supported primary paired gain and nonnegative secondary change advance to
the existing five-day check (combiner trained on Jan21). This is a test of
better use of existing supervised data, not a proven consequence of the
matched-cutoff result. No new models have been trained and no submission files
have changed; I continue to run experiments in WSL.

Prepared implementation: expanded_combiner_features.py and
run_expanded_combiner_trial.py. The extension keeps the original 37 formulas
and fits the original ridge encoder on prefix clicks, but scores each cold
event using its own inferred profile without mutating user embeddings. Event
weights are inverse fitting-day user counts, rescaled to mean row weight one;
all ten candidate rows inherit their event's weight. WSL self-test compares
both dimensions against original one-event inference, including repeated
cold users and empty histories. The trial freezes source/checkpoint hashes,
checks first-event feature equivalence and original member predictions, fits
ten new 545-tree combiners, and applies the original primary/secondary gate.
It supports resuming completed member results and displaying an existing final
report. At that stage, I still needed to run the GPU self-test and real fitting
in WSL; local numeric, adaptation and synthetic orchestration checks passed.

I completed the five-day V5 baseline: AUC 0.781711 on 51,488 first-user
events, with bootstrap interval [0.779203, 0.784061]. Later-day cohorts contain
more unseen users and short histories, so daily AUC declines cannot be read as
isolated time decay. Prepared compare_v5_matched_events.py for the 3,531 Jan25-26
events common to the Jan22-cutoff baseline and Jan25-cutoff V5 predictions.
It reconstructs both cached ensembles, aligns labels and indices, verifies
unchanged fitting activity for matched users, and computes 1,000 paired user
bootstrap resamples with exploratory group intervals. This measures the change
between more recently fitted pipelines on identical targets, while retaining
the confounding between additional data, parameter fitting, exposure statistics
and combiner fitting day. No fitting, model changes or submission occur.

The separate candidate-attention expert also failed its fixed integration:
primary AUC change -0.001169, secondary -0.001280, and the primary paired
interval wholly negative. Its unseen-with-history subgroup point gain is
exploratory and does not justify promotion or a post-hoc routing rule.

Prepared next: run_v5_long_window.py reconstructs the unchanged V5 recipe at
two earlier cutoffs. Meta features use neural models fitted before Jan21 and
combiner labels from Jan21; stress features use models fitted before Jan22,
scored on first events per user across Jan22-26. It creates ten neural fits and
five 545-tree combiners in techarena-v5-long-window, retaining the five original
member weights and all 37 feature formulas. Prefix click examples exclude
future-only users and are checked as a multiset against raw fitting histories
and labeled fitting clicks. The report establishes a diagnostic baseline; it
has no promotion/qualification score threshold and generates no submission.
Differences by horizon mix user-cohort differences with time effects, and
comparison with the existing shorter-window AUC also changes fitting-set size.

The pooled correction failed both main checks. Its all-five integration lost
0.002112 primary AUC, and the fixed dim8-only follow-up lost 0.000503 with a
wholly negative paired interval. Neither advances to the longer-window check.
V5 remains the current submission baseline (my recorded public AUC 0.819402).

The next prepared, not yet executed experiment is a standalone candidate-specific
attention history expert. It initializes item vectors from the corresponding
earlier-cutoff dim16 seed42 preference model, trains using only fitting events,
and uses no user-ID embedding or V5 prediction as a neural input. A separate
14-feature calibration model is fitted on the preceding meta-training day.
The only planned integration is 85% unchanged V5 plus 15% expert for nonempty
histories; empty-history predictions remain exactly V5. Fixed sampled-click
pretraining is 16 epochs, followed by 8 event epochs; calibration is 545 trees.
The same positive primary paired interval and nonnegative secondary change are
required before a longer-window stress check. This is another retrospective
screen on reused dates, not independent confirmation or a qualification promise.

These papers motivate hypotheses; their benchmark results do not establish
that a method will improve this competition.

- Sarwar et al., item-based collaborative filtering: item relationships can be
  used to score items against a user's previous interactions.
  https://www.ra.ethz.ch/CDstore/www10/papers/519/
- Kabbur et al., FISM: factorized item relationships offer a compact model for
  sparse interaction data.
  https://chbrown.github.io/kdd-2013-usb/kdd/p659.pdf
- Zhou et al., Deep Interest Network: history relevance can depend on the
  candidate being scored.
  https://arxiv.org/abs/1706.06978
- Steck, EASE: useful item-based reference; its standard solution uses a full
  item-item matrix and matrix inversion.
  https://arxiv.org/html/1905.03375v1
