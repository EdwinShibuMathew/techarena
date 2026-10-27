# PR #21 replay review fixes

Implementation commit: `16b72b373a2125d6080eea350987283e92671c84`, based on
reviewed PR head `14eca06c0d6f7628ba3bc86fae9ebe41a82bc1a8`.

Both findings in the [review comment](https://github.com/EdwinShibuMathew/techarena/pull/21#issuecomment-6452123127)
were reproduced and fixed. Supported replay now defaults to
`artifacts/replays/<release>.csv`, avoiding the protected published hard link.
The controller publishes the destination only after final shared accounting and
replay-ledger persistence succeed, and writes the ready replay result last.

The maintained bridge verifies source snapshots before importing their modules,
then uses an exec transition into a fresh interpreter with frozen scientific
imports. It preserves the controller-owned PID, active lease, watchdog and hard
timeout. Frozen inference and policy verification remain intact. Snapshot workers
that publish directly are never invoked. Existing manifests, scientific results,
release configurations and replay receipts are unchanged.

## Validation on the implementation commit

| Check | Result |
| --- | --- |
| Locked CPU regression suite | 165 passed, 1 optional private-artifact skip |
| Required model contracts | 52 passed, zero skips |
| Pinned CPU dependency consistency | Passed |
| Source integrity | All 59 frozen files verified |
| Frozen paths versus reviewed PR head | Unchanged |
| Source release package and checksums | Passed |

The optional skip is `test_v7_submission_integrity_if_present`, because
`artifacts/submissions/v7_candidate.csv` is absent. The required model-contract
runner includes all ten new replay tests and permits no skips.

Replay regressions cover both default CLIs after actual hard-link publication,
protected hard-link/symlink aliases, final accounting and replay-ledger write
failures, publication ordering, worker failure and timeout, staged artifact
tampering, matching/different existing outputs, publication races and final result
write failure. Subprocess fixtures exercise real snapshot/runtime hash checks,
lease validation, export/CSV validation and controller-death termination. A
between-preflight-and-launch tampering test proves altered snapshot code is
rejected before it executes.

The frozen subprocess fixtures declare a numerical inference substitute before
sealing their source. They use synthetic twenty-row data and real CSV validation;
they do not establish exact private V6/V7/V7.3 inference. Other existing CPU model
contracts continue to exercise real fitting, gradients and checkpoint reload.

No private inference, new research campaign, competition upload or scientific
promotion was performed. The optional bundle-verification command could not run
because this checkout lacks `artifacts/v6/manifest.json`. The historical private
replay evidence remains historical and unchanged. Synthetic fixtures use their
own disposable accounting ledgers; the research ledger was not reset or changed.

The [machine-readable validation record](validation.json) binds these checks and
the source-package fingerprint to the implementation commit. It records local
checks; subsequent GitHub CI status belongs to its actual tested commit.

See [supported replay lifecycle and invocation](../../ML_WORKFLOW.md#schema-v3-joint-calibration).
