# Consolidation and rollback

## What is protected

The repository is a new source copy. Original Windows sources, WSL studies,
V5/V6 checkpoints and submission CSVs are not edited during preparation.

`bundle` copies V6 event/preference checkpoints, combiner/prediction assets, shared caches,
available original archives, example submission and the V5/V6 submitted CSVs.
Every copied file is SHA256 checked. Existing different destination bytes cause
an error, not an overwrite. Available compact study reports are retained.

The new manifest uses repository-relative paths. It records original locations
for provenance without changing the old frozen reports. Saved symbolic-link
targets are dereferenced only when copying specific required assets.

## Required order

1. `inventory`: record the actual WSL `~/projects/techarena*` folders and sizes.
2. `bundle`: collect the known benchmark assets without moving original folders.
3. `verify`: check collected bytes (also done before portable prediction).
4. `predict_v6.py`: replay all five members and require the submitted CSV hash.
5. `archive`: inspect the exact proposed source/destination list.
6. `archive --apply`: move old plain experiment folders inside `local/archive`.
7. `publish_private.py`: publish code and reports privately, excluding local assets.

Archive is a reversible directory rename on the same filesystem. It is not a
deletion or a backup on another device. Disk reclamation/deduplication is not
performed automatically. Keep separate backups for protection against disk loss.

Folders containing a Git checkout are excluded from automatic relocation. The
tool blocks active user processes whose working directory is under a candidate
folder. Run it from the new repository and move other terminals out of old
folders first. Nested symlink/mount/Git checks are also performed before moves.

Archived historical symlinks and frozen manifests may still mention their old
absolute paths. Their bytes are preserved as evidence. They are not the new
active pipeline; restore the original layout before using those old runners.
The current V6 pipeline uses the independent collected asset paths instead.

## Recovery

Archive moves are journaled before the first rename and after each completed
rename. An interrupted archive can resume with the same command. Restore with:

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
