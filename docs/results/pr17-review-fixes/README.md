# PR #17 review fixes

Both findings in the [tech lead review](https://github.com/EdwinShibuMathew/techarena/pull/17#issuecomment-6428253067)
were reproduced against `ea2806f` and addressed. [Fresh validation](validation.json)
binds the exact implementation and regression-test hashes used for this follow-up.

- **P1:** checkpoint expectations remain independently owned by the runner;
  adapters receive deep copies of metadata and nested candidate parameters.
  Altered cutoff, dimension and preference epoch are rejected for dim8/dim16
  roles without a success receipt. Valid saved metadata remains accepted and
  its receipt stays correct even if the adapter subsequently mutates its copies.
- **P2:** the budget lock and reservation precede the immutable phase claim.
  A contended phase leaves no claim, launches no worker and does not change the
  held ledger. The same phase succeeds after lock release, including the full
  synthetic fit/gate/export/replay campaign. Subsequent retries remain blocked;
  existing scientific-failure, timeout and controller-death contracts still pass.

Locked Python 3.12 CPU checks: **129 passed, one optional private-artifact skip**
in 23.33s. Required model/workflow checks: **20 passed, zero skips** in 21.46s.
Source-only checks: 119 passed, 11 optional skips and 88 passing subtests.
Dependency consistency and verification of the existing 29-file asset bundle
also passed. Frozen V6 files have no changes.

No new CUDA replay was performed for these runner-only fixes. Earlier canonical
V6 replay evidence, historical reports, closed decisions and original workspace
changes were preserved. No research candidate was trained or promoted.
