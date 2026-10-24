# Archive disk cleanup

The first cleanup shares storage for byte-identical archived binaries. It keeps
every filename, every unique model, every dataset, every experiment report and
all symbolic-link targets. It does not delete whole experiment folders.

Only regular `.npz`, `.npy`, `.pt`, `.pth`, `.cbm`, `.csv` and `.zip` files of at
least 1 MiB are candidates. Files must have identical SHA256, size, permissions
and ownership, and be on the same filesystem. Existing hardlinks are counted
once. Any inode linked outside the archive is excluded. Active `data/` and
`artifacts/` copies are not linked to the archive.

This deliberately retains unique V5 assets in the archive: the portable bundle
collects all V6 members, but does not independently collect every V5 event model.
No rejected study's unique checkpoint or prediction array is discarded.

In WSL, from the team repository:

```bash
python tools/deduplicate_archive.py self-test
python tools/deduplicate_archive.py plan
```

`plan` writes an ignored full content inventory and proposed changes to
`local/archive_dedup_plan.json`, with a separate saved run record under
`local/archive-dedup/`. It changes no archived files. The GiB estimate comes from
the allocated blocks of duplicate inodes expected to lose their final link;
underlying filesystem sharing/compression can affect physical savings.

The self-test must pass in WSL with no skipped cases before planning/applying.
It exercises real hardlinks and symbolic links using synthetic files in
`local/test-runs/`, never competition files. Windows sandbox checks cannot
exercise all Linux filesystem operations, so the WSL check is mandatory.

To apply the displayed plan:

```bash
python tools/deduplicate_archive.py apply
python tools/publish_private.py
df -h .
```

`apply` requires the completed archive journal and the successful V6 replay
receipt. It rehashes all archived regular files before and after cleanup, checks
all archived symbolic links, and revalidates the live bundle. Each duplicate is
atomically replaced by a hardlink to an archived keeper, retaining the path and
bytes. It refuses active processes in the archive, symlink operation paths,
nested mounts and Git checkouts. An interrupted run can resume with `apply`.
Do not edit archive files while planning or applying cleanup.

A hardlink is another filename for the same stored bytes. Treat the archive as
read-only: **copy files to a new working directory before editing or retraining**.
Changing one hardlinked archived file in place would change its aliases. Copies
in the active data/model bundle remain independent. Duplicate modification
timestamps may become the keeper's timestamp; original per-file timestamps are
retained in the cleanup plan.

Cleanup retains archive restoration of the original directory names, but restore
does not undo storage sharing. To recover independent files for future edits,
copy the needed study instead of running old training directly in the archive.

The compact completion report is `docs/results/archive_deduplication.json` and
is published with the code. Full inventories/progress remain local. WSL Linux
free space may increase without the Windows virtual-disk file shrinking; this
tool does not compact that virtual disk or touch Windows Downloads.
