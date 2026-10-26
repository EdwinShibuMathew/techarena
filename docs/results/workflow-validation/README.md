# Integrated ML workflow validation

[Compact evidence](validation.json) covers issues #9–#16 using the standalone
runtime on `1f359f8`. Subsequent report-only commits preserve that source snapshot.
No V8 candidate was trained or promoted, no leaderboard score is claimed, and
closed campaign decisions and private caches were preserved.

- Locked Python 3.12 CPU environment: 126 passed, one optional private-artifact
  skip in 26.42 seconds. Required model/workflow contracts: 17 run, zero skips,
  21.70 seconds. Dependency consistency passed.
- A deliberately detached preference forward pass, patched only in memory,
  caused both dimension gradient contracts to fail and the required checker to
  exit 1. The source package also refuses missing/failed model-check results.
- A public synthetic campaign completed prepare, develop, screen, stress,
  finalize and verify. Its 480-row CSV matched an independent uncached replay.
  Its explicit `submission_ready: false` prevents confusing workflow validation
  with competition readiness. Real Torch fits and CatBoost save/reload were used.
- CPU integration covers terminal scientific failure, blocked downstream stages,
  changed inputs, unrelated current-checkout maintenance, source snapshots,
  controller death and charged compute reservations. Pure contract tests cover
  every primary/secondary/five-day/late promotion failure condition.
- Cache tests cover cross-request reuse, changed extraction dependencies,
  labels, cutoffs, row/schema order, runtime settings, corruption, partial writes,
  concurrent producers and exact probability dtype preservation. Separate
  concurrent V6 export producers both reached shared receipt publication.
- A fresh LF source fixture verified all 29 existing private assets, replayed all
  five V6 members on the existing CUDA environment with **zero** differences,
  and reproduced canonical SHA256
  `c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c`.
  The receipt records the original CRLF manifest hash, checkout LF hash and
  `lf-crlf-only` adapter. Receipt revalidation also passed.

Large outputs, private assets, caches and raw logs remain ignored. The repository
admin should review the integrated PR and apply the updated protection policy
once the four GitHub Actions checks are green. Nothing here merges PR #4 or
closes the tracked issues automatically.
