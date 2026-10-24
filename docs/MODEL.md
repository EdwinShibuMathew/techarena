# Current model: V6

Predict a click probability for every candidate row. Each event contains ten
candidates and at most one click; the neural event model includes an eleventh
no-click outcome. The official metric is global row ROC AUC.

## Members and weights

| Member | Preference epochs | Event objective | Weight |
|---|---:|---|---:|
| dimension8 seed42 | 196 + 64 lower-rate | half event CE / half global row-pair logistic | 2/9 |
| dimension8 seed43 | 196 + 64 lower-rate | half event CE / half global row-pair logistic | 2/9 |
| dimension8 seed44 | 196 + 64 lower-rate | original event CE | 2/9 |
| dimension16 seed42 | 196 | original event CE | 1/6 |
| dimension16 seed43 | 196 | original event CE | 1/6 |

All event stages use eight epochs. Preference learning uses clicked examples and
31 uniform sampled comparison items. Fine-tuning retains Adam moments.

Inference applies the original 11-outcome softmax, exposure smoothing20/power0.5,
and ridge history-profile fallback for unseen users with nonempty history. Each
member has a 37-feature, 545-tree CatBoost combiner. The ensemble preserves the
original floating-point averaging order.

Review `legacy/train_final_models.py`, `legacy/row_auc_objective.py`,
`legacy/build_submission.py`, `legacy/submission_features.py` and
`legacy/run_row_auc_final.py`. These are the exact historical sources; the new
prediction wrapper makes model/data locations relative to this repository.

## Reported evidence

| Evaluation | Events | V6 AUC |
|---|---:|---:|
| first-user Jan25–26; pre-Jan25 neural models, Jan24 combiner | 43,964 | 0.8184592533262289 |
| first-user Jan26; pre-Jan26 neural models, Jan25 combiner | 35,138 | 0.8271277122502162 |
| first-user Jan22–26; pre-Jan22 models, Jan21 combiner | 51,488 | 0.7818190858952342 |
| reported public leaderboard | hidden subset | 0.819451 |

The five-day gain over V5 was approximately 0.0001078, with a paired interval
approximately [0.0000232, 0.0001936]. Dates have been used repeatedly for model
development. These intervals are not adjusted for adaptive selection.

## Reproduction contract

V6 has 529,390 evaluation rows. Its submitted CSV SHA256 is:

```text
c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c
```

The portable wrapper checks every member against saved predictions before
checking the ensemble and export. It accepts a relocation only if the new CSV
is byte-identical to the submitted V6. Until that test passes in Edwin's WSL,
the wrapper is prepared code, not verified reproduction of the real artifacts.
