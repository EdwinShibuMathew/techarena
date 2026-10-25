# How I consolidated the project and recover old files

## What I preserved

I created a separate source repository, collected the verified V6 assets, and
moved 34 old experiment folders into `local/archive/`. The migration preserved
my original sources, studies, V5/V6 checkpoints and submission CSVs.

I later removed seven archived data copies that matched the active bundle,
reclaiming about 0.947 GiB. Models and experiment results stayed unchanged;
[the data-copy cleanup guide](ARCHIVED_DATA_COPIES.md) explains how I can recreate
those data copies before restoring the old folder layout.

`bundle` copies V6 event/preference checkpoints, combiner/prediction assets, shared caches,
available original archives, example submission and the V5/V6 submitted CSVs.
Every copied file is SHA256 checked. Existing different destination bytes cause
an error, not an overwrite. Available compact study reports are retained.

The new manifest uses repository-relative paths. It records original locations
for provenance without changing the old frozen reports. Saved symbolic-link
targets are dereferenced only when copying specific required assets.

## The migration order I used

1. `inventory`: record the actual WSL `~/projects/techarena*` folders and sizes.
2. `bundle`: collect the known benchmark assets without moving original folders.
3. `verify`: check collected bytes (also done before portable prediction).
4. `predict_v6.py`: replay all five members and require the submitted CSV hash.
5. `archive`: inspect the exact proposed source/destination list.
6. `archive --apply`: move old plain experiment folders inside `local/archive`.
7. `publish_private.py`: publish code and reports privately, excluding local assets.

I used directory renames on the same filesystem to archive the studies. That
made the folder moves reversible, but did not create a backup on another device
or reclaim disk space. My later data-copy cleanup was a separate verified step.

Folders containing a Git checkout are excluded from automatic relocation. The
tool blocks active user processes whose working directory is under a candidate
folder. Run it from the new repository and move other terminals out of old
folders first. Nested symlink/mount/Git checks are also performed before moves.

Archived historical symlinks and frozen manifests may still mention their old
absolute paths. Their bytes are preserved as evidence. They are not the new
active pipeline; restore the original layout before using those old runners.
The current V6 pipeline uses the independent collected asset paths instead.

## How I recover the original layout

The migration journals each folder move so I can resume an interrupted archive.
After the data-copy cleanup, I first recreate the removed archive data files with
`python tools/clean_archived_data_copies.py restore`. I can then preview and
reverse the folder moves with:

```bash
python tools/consolidate.py restore
python tools/consolidate.py restore --apply
```

Restore refuses to overwrite occupied original paths. Do not manually edit the
journal or frozen reports to bypass a failed check.

## Outside this migration

Other Windows chat/project folders and Downloads CSVs are left unchanged. They
may contain unrelated work, and some are attached to chats. The inventory and
archive commands here target only the reviewed WSL TechArena project folders.
No GitHub repository is deleted, and no collaborators are invited automatically.
