# Reclaim redundant archived competition-data copies

The archive-only hardlink check excludes the independent active bundle. A zero
result from that check therefore does not mean there are no data copies shared
in content between `local/archive/` and the active `data/` folder.

This separate cleanup removes only regular archived files under a data directory
whose basename, SHA256 and size match one of these verified active assets:

- `data/raw/train.zip`, `evaluation.zip`, `example_submission.csv`
- `data/cache/train_events.npz`, `evaluation_events.npz`,
  `evaluation_rowids.npy`, `preference_training.npz`

No model checkpoint, combiner, prediction array, submission CSV, experiment
report, source, unknown cache or symbolic link is removed. All unique V5 assets
remain archived. Every live model/data asset is verified through the successful
V6 relocation receipt before/after cleanup. Existing shared inodes are excluded
unless all their names are proposed archive copies, so estimated savings do not
count data still occupying storage through another filename.

```bash
python tools/clean_archived_data_copies.py self-test
python tools/clean_archived_data_copies.py plan
```

Review the exact remove/keep pairs and allocated-space estimate. `plan` changes
no archived files. Then apply:

```bash
python tools/clean_archived_data_copies.py apply
python tools/publish_private.py
df -h .
```

The self-test covers actual file removal/restoration on tiny synthetic fixtures;
the cleanup is gated on its current passing receipt. Apply is WSL/Linux-only.
Every file in the archive is hashed before and after: only the exact listed data
copies may disappear, and every remaining file and symbolic link must match.
The active keepers are checked immediately before each archived copy is removed.
Interrupted apply can resume. Full plans and progress stay under ignored
`local/archived-data-cleanup/`; the compact result is published under
`docs/results/archived_data_cleanup.json`.

This is removal of redundant data copies, not removal of the data itself. It
does mean the archive is no longer an independently complete copy of those data
files. Current portable inference uses the verified active data and is unaffected.
Before restoring/running old archived studies, recreate removed data copies:

```bash
python tools/clean_archived_data_copies.py restore
```

Restore requires the current verified live data and sufficient free space, and
refuses different occupied paths. It copies exact bytes back and restores the
recorded permissions/timestamps. Run it BEFORE `consolidate.py restore --apply`
if you need to recover the old folder layout. Do not apply the earlier archive
hardlink plan after this cleanup; its full-archive inventory will be stale.

Linux free space and Windows virtual-disk size are different. This tool does not
compact the WSL virtual disk or touch Windows Downloads.
