# V7.2 joint calibration: rejected during development

**Not submission-ready.** None of the three predeclared candidates passed the
aged-development eligibility check. The campaign stopped before promotion;
none of the four promotion gates was evaluated or changed. No V7.2 configuration,
submission CSV or competition upload exists.

| Candidate | Jan22–24 AUC change | Jan24 forward change | Aged 48–72h change | Eligibility |
|---|---:|---:|---:|---|
| Event correction | +0.000666948308 | +0.000474150119 | −0.000152856957 | Failed |
| Joint constant sharpening | +0.000726594348 | +0.000585951991 | −0.000084809510 | Failed |
| Joint entropy sharpening | +0.000720001722 | +0.000573566195 | −0.000087419660 | Failed |

The development windows contain 47,957 and 35,040 first-user events; the aged
subset contains 4,065 events. All changes compare global row ROC AUC with V6.
Eligibility requires nonnegative changes in both windows and the aged subset,
with a strictly positive mean across the two overall windows. No candidate
qualifies, so no winner, promotion bootstrap interval or ready release exists.

## Method and interpretation

The [frozen protocol](../../../experiments/v7.2-joint-calibration-v1/RUN_V7_2.md)
tested a bounded event-wide confidence correction, with optional constant or
entropy-conditioned conditional-score correction. Unlike V7.1, event sums could
change; within-event order, ties and zero/one endpoints stayed intact. The three
variants shared the same fitting data, user-grouped cross-fitted V6 scores and
reference combiners. Normalization was fitted only on the metadata day and
checked independently before target scoring.

All five original neural scorers, released combiners and ensemble weights stayed
unchanged. Fifty temporary fold combiners and five development reference
combiners were fitted, using the frozen original recipe. Six calibrators
converged under the declared optimizer settings. There was no new neural or
preference training, cohort routing, public-leaderboard selection, post-outcome
parameter change or fallback candidate.

Both aggregate development windows improved for every variant, but aged
development regressed for every variant. Aggregate gains therefore did not
establish eligibility. This result does not prove why aged transfer failed.
The reused dates remain retrospective evidence, not independent or
selection-adjusted validation. In particular, these are development results,
not measured primary, secondary, five-day or late-promotion passes.

## Provenance, integrity and accounting

- [Preparation and source identity](preparation.json)
- [Exact development report](develop.json), [seal](develop.seal.json) and
  [completed accounting](develop.ledger.json)
- [Terminal decision, refusal checks and numerical integrity](decision.json)

Scientific source was frozen at `dfd1361eee5411f431cac645da3c23b82a7356ed`.
The public report, seal and ledger match the immutable private evidence
byte-for-byte. The source archive and manifest hashes were verified. All
declared assets and 59 frozen source files verified before research.

The saved V6 reference vector reproduced with zero maximum error on Jan22–24
for all three candidates. Jan24 forward used a fresh five-member V6-recipe
reference shared across candidates with identity/hash checks. Every scored
probability was finite and in [0,1]; no within-event ordering or tie changed.
Event sums changed intentionally under schema-v3, rather than weakening
V7.1's schema-v2 mass-preservation requirement.

Feature/OOF caching recorded 81 requests: 59 hits and 22 misses. Preparation used
29.342358254 seconds and development 478.462095259 seconds, both within the
1,200-second phase cap. New scientific compute was 507.804453513 seconds;
cumulative usage is 3,397.270282478001 seconds of the original 14,400-second cap.
The original carry identity and all earlier reservations remain unchanged.

The actual `screen`, `stress` and `finalize` commands each refused with
`develop gate failed; recipe is terminal`, before claiming a phase or starting
further fitting. The release replay CLI refused the missing configuration
before inference. Remaining budget does not reopen this closed campaign.

Before research, the locked CPU suite passed 148 tests with one optional private
artifact skip, and all 35 required model contracts passed without skips. Tests
cover real calibration/reload, normalization, aged eligibility, shared reference
reuse/corruption, and actual synthetic CSV export/replay. Synthetic validation
does not establish successful private final-release execution; that path remains
unrun for V7.2 because development failed. Source CI also does not establish
scientific promotion.
