# V7.4 from-raw reconstruction — rejected at development

**All 25 chronological prefix models were rebuilt and trained. V7.4 was not
promoted:** both declared graph variants regressed on aged events versus rebuilt
V6, so neither qualified for promotion screening. No full-prefix final models,
release configuration or submission CSV were produced.

The user authorized scratch reconstruction and a separate 43,200-second budget
after confirming that checkpoints, prefix assets, processed caches and the
original ledger were lost. The original raw archives and example submission
matched the declared fingerprints. Reconstruction verified 686,824 training
events, 52,939 evaluation events, 54,456 users, 50,000 items and 1,009,707 full
preference examples. Five fixed-schedule members were trained at each of Jan21,
Jan22, Jan24, Jan25 and Jan26. Adam and RNG state are saved for technical recovery.

Comparators are **newly fingerprinted V6/V7 recipe reconstructions**, not a byte
reproduction of the missing historical vectors. Historical configurations,
frozen sources, outcomes and accounting were preserved. The lost original ledger
was not recreated; its last reported usage remains unverified.

## Measured development results

The ensemble used the original 37 features, fixed 545-tree combiners and ensemble
weights. Graph coefficients were fitted from five user-grouped out-of-fold
combiner predictions. V7 uses the fixed gamma of 1.12. Both graph variants preserve
its event click mass.

| Variant | Window | Events | AUC gain vs rebuilt V6 | AUC gain vs rebuilt V7 | Aged gain vs rebuilt V6 |
| --- | --- | ---: | ---: | ---: | ---: |
| 24h | Jan22–24 | 47,957 | +0.000292034 | +0.000024013 | **−0.000244640** |
| 72h | Jan22–24 | 47,957 | +0.000296728 | +0.000028706 | **−0.000214726** |
| 24h | Jan24 | 35,040 | +0.000197137 | +0.000014306 | — |
| 72h | Jan24 | 35,040 | +0.000204773 | +0.000021941 | — |

The aged subset contains 4,065 first-user events at least 48 hours after the
Jan22 target cutoff. Both variants improved this subset versus rebuilt V7, but
the declared selection rule requires non-negative gains against **both** anchors.
Overall gains cannot override the negative aged comparison against rebuilt V6.
The frozen maximin selection returned no winner and closed the recipe.

Primary, Jan26, five-day and finalization stages were not run. No promotion
confidence intervals were measured. These reused dates are retrospective
development evidence, not independent holdout evidence. The 24h/72h recipe is
closed; there is no runner-up, weight retuning or fallback after this decision.

## Accounting and engineering checks

The separate ledger charged **5,048.286446463993 seconds** (84.14 minutes) of the
approved 43,200 seconds. All 71 preparation, training and evaluation jobs settled
successfully. Scientific rejection is recorded as a completed development phase.
Subsequent screening and finalization commands refused; they created no claims
and left the ledger hash unchanged.

At scientific source `49cc3677855678356ac00a48156af0b4442703a3`:

- Locked CPU regressions: **195 passed**, one optional private-artifact skip.
- Required model contracts: **82 passed**, zero skips, including 12 reconstruction
  contracts for chronology, real graph scoring, fixed combiners, user folds,
  cache/provenance checks, exact technical recovery and durable accounting.
- Source integrity: all **59 frozen files** verified; source packaging passed.
- [GitHub Source CI](https://github.com/EdwinShibuMathew/techarena/actions/runs/40608973998)
  passed all four jobs.

[results.json](results.json) records exact metrics, runtime, raw/input/model
hashes, training/scientific source commits, budget lineage and the terminal
decision. [validation.json](validation.json) records engineering checks and the
real terminal-command refusals. Trained assets remain local under ignored
`artifacts/rebuild/`; the public source contains no weights or raw data.

The [protocol](../../../experiments/v7.4-rebuild-v1/RUN_REBUILD.md) and controller
preserve this generation. Inspect it with:

```bash
python experiments/v7.4-rebuild-v1/run.py status
python experiments/v7.4-rebuild-v1/report.py
```
