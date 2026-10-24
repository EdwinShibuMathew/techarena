"""Remove ONLY archived data copies that byte-match the verified active data bundle.

Models, predictions, reports, code, unknown data, and symlinks are never deleted.
Use self-test, plan, apply; restore recreates removed archive copies from live data.
"""

import argparse
import collections
import json
import os
import re
import shutil
import stat
import sys
import unittest
import uuid
from pathlib import Path

import consolidate as c
import deduplicate_archive as archive

DATA = {"data/raw/train.zip", "data/raw/evaluation.zip", "data/raw/example_submission.csv",
        "data/cache/train_events.npz", "data/cache/evaluation_events.npz",
        "data/cache/evaluation_rowids.npy", "data/cache/preference_training.npz"}


def linux_only():
    if os.name != "posix":
        raise RuntimeError("Run apply/restore in WSL/Linux.")


def self_test():
    suite = unittest.defaultTestLoader.discover(str(c.ROOT / "tests"), pattern="test_archived_data_copies.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("All archived-data cleanup tests must pass without skips.")
    c.write_json(c.ROOT / "local/archived_data_cleanup_self_test.json", {
        "passed": True, "tests": result.testsRun,
        "code_sha256": c.digest(c.ROOT / "tools/clean_archived_data_copies.py"),
        "tests_sha256": c.digest(c.ROOT / "tests/test_archived_data_copies.py")})


def checked(root):
    proof = c.read_json(root / "local/archived_data_cleanup_self_test.json")
    if (proof.get("passed") is not True
            or proof["code_sha256"] != c.digest(root / "tools/clean_archived_data_copies.py")
            or proof["tests_sha256"] != c.digest(root / "tests/test_archived_data_copies.py")):
        raise RuntimeError("Run the current archived-data cleanup self-test first.")
    c.require_receipt(root)
    manifest = c.read_json(root / "artifacts/v6/manifest.json")
    for relative in DATA & set(manifest["files"]):
        current = root
        for part in Path(relative).parts:
            current = current / part
            if current.is_symlink():
                raise RuntimeError(f"Canonical data must be an independent regular copy: {relative}")
        if not stat.S_ISREG(current.lstat().st_mode):
            raise RuntimeError(f"Canonical data is not a regular file: {relative}")
    return manifest


def select(state, manifest):
    references = collections.defaultdict(list)
    for relative in sorted(DATA):
        if relative in manifest["files"]:
            entry = manifest["files"][relative]
            references[(Path(relative).name, entry["sha256"], entry["bytes"])].append(relative)
    proposed, inode_names = {}, collections.defaultdict(set)
    for name, entry in state["files"].items():
        inode_names[tuple(entry["inode"])].add(name)
        parts = Path(name).parts
        # Both archive prefix and a data/ ancestor are required. No work/, outputs/ evidence.
        if parts[:2] != ("local", "archive") or "data" not in parts[3:-1]:
            continue
        # In particular, never delete predictions/results put under a data/ folder.
        if "work" in parts[3:-1] or "outputs" in parts[3:-1]:
            continue
        candidates = references.get((Path(name).name, entry["sha256"], entry["bytes"]), [])
        if candidates:
            proposed[name] = {"archive": name, "canonical": candidates[0],
                              "sha256": entry["sha256"], "bytes": entry["bytes"]}
    chosen, releasable = [], {}
    for name, operation in sorted(proposed.items()):
        entry = state["files"][name]
        inode = tuple(entry["inode"])
        if entry["links"] != len(inode_names[inode]) or not inode_names[inode].issubset(proposed):
            continue  # Already shares storage outside proposed copies; do not claim savings.
        chosen.append(operation)
        releasable[inode] = entry["allocated_bytes"]
    return chosen, sum(releasable.values())


def plan(root=c.ROOT):
    root, _ = archive.location(root)
    manifest = checked(root)
    path = root / "local/archived_data_cleanup_plan.json"
    if path.exists():
        old = c.read_json(path)
        progress = root / "local/archived-data-cleanup" / old["id"] / "progress.json"
        if progress.exists() and not c.read_json(progress).get("restored"):
            raise RuntimeError("A cleanup already started. Use apply to resume or restore its data first.")
    state = archive.snapshot(root)
    operations, allocated = select(state, manifest)
    record = {"id": uuid.uuid4().hex, "created_at": c.utc(), "root": str(root),
        "code_sha256": c.digest(root / "tools/clean_archived_data_copies.py"),
        "manifest_sha256": c.digest(root / "artifacts/v6/manifest.json"),
        "journal_sha256": c.digest(root / "local/archive_journal.json"),
        "snapshot": state, "operations": operations, "estimated_bytes_reclaimable": allocated}
    c.write_json(root / "local/archived-data-cleanup" / record["id"] / "plan.json", record)
    c.write_json(path, record)
    print(json.dumps({"archive_copies_to_remove": len(operations),
        "estimated_reclaimable_GiB": round(allocated / 1024**3, 3),
        "models_predictions_reports_code_deleted": 0, "applied": False,
        "copies": [{"MiB": round(op["bytes"] / 1024**2, 2),
                    "remove_archive_copy": op["archive"], "keep_verified_copy": op["canonical"]}
                   for op in operations]}, indent=2))
    return record


def load_plan(root):
    manifest = checked(root)
    record = c.read_json(root / "local/archived_data_cleanup_plan.json")
    if record["root"] != str(root) or not re.fullmatch("[0-9a-f]{32}", record["id"]):
        raise RuntimeError("Cleanup plan location/id mismatch.")
    run = root / "local/archived-data-cleanup" / record["id"]
    if run.is_symlink() or c.read_json(run / "plan.json") != record:
        raise RuntimeError("Saved cleanup plan does not match its run record.")
    for relative, field in [("tools/clean_archived_data_copies.py", "code_sha256"),
                            ("artifacts/v6/manifest.json", "manifest_sha256"),
                            ("local/archive_journal.json", "journal_sha256")]:
        if c.digest(root / relative) != record[field]:
            raise RuntimeError(f"Cleanup evidence changed: {relative}")
    expected, size = select(record["snapshot"], manifest)
    if expected != record["operations"] or size != record["estimated_bytes_reclaimable"]:
        raise RuntimeError("Cleanup operations do not match the restrictive data-only rule.")
    return record, run


def consistent(root, record, allow_missing):
    current = archive.snapshot(root)
    expected = record["snapshot"]
    allowed = {op["archive"] for op in record["operations"]} if allow_missing else set()
    absent = set(expected["files"]) - set(current["files"])
    if absent - allowed:
        raise RuntimeError("An unplanned archived file is missing.")
    reduced = {"files": {k: v for k, v in expected["files"].items() if k not in absent},
               "symlinks": expected["symlinks"]}
    archive.same_contents(reduced, current)
    return current


def apply(root=c.ROOT):
    linux_only()
    root, archived = archive.location(root)
    record, run = load_plan(root)
    if c.running_under([archived]):
        raise RuntimeError("Archived files are in use. Close their jobs/editor terminals first.")
    progress_path = run / "progress.json"
    progress = c.read_json(progress_path) if progress_path.exists() else None
    if progress and progress.get("restored"):
        raise RuntimeError("This cleanup was restored. Make a new plan.")
    current = consistent(root, record, allow_missing=progress is not None)
    if progress and progress.get("passed"):
        if any(op["archive"] in current["files"] for op in record["operations"]):
            raise RuntimeError("Removed data copies reappeared. Review rather than deleting again.")
        print("Cleanup already complete; remaining archive contents and live bundle verified.")
        return progress
    # Verify every remaining source before the first removal; canonical assets were checked above.
    removed_links = collections.Counter(tuple(entry["inode"])
        for name, entry in record["snapshot"]["files"].items() if name not in current["files"])
    for op in record["operations"]:
        source = archive.archived_file(root, op["archive"])
        if op["archive"] in current["files"]:
            observed = current["files"][op["archive"]]
            original = record["snapshot"]["files"][op["archive"]]
            expected_links = original["links"] - removed_links[tuple(original["inode"])]
            if observed["inode"] != original["inode"] or observed["links"] != expected_links:
                raise RuntimeError(f"Archive copy links/identity changed: {source}")
    if progress is None:
        progress = {"id": record["id"], "started_at": c.utc(), "passed": False, "removed": []}
        c.write_json(progress_path, progress)
    for op in record["operations"]:
        source = archive.archived_file(root, op["archive"])
        canonical = c.asset(root, op["canonical"])
        if c.digest(canonical) != op["sha256"]:
            raise RuntimeError(f"Active keeper changed: {canonical}")
        if source.exists():
            observed = archive.fingerprint(source)
            original = record["snapshot"]["files"][op["archive"]]
            if observed["sha256"] != op["sha256"] or observed["inode"] != original["inode"]:
                raise RuntimeError(f"Archived data changed: {source}")
            source.unlink()  # Exact verified regular file; never a directory or symlink.
        if op["archive"] not in progress["removed"]:
            progress["removed"].append(op["archive"])
        c.write_json(progress_path, progress)
    after = consistent(root, record, allow_missing=True)
    if any(op["archive"] in after["files"] for op in record["operations"]):
        raise RuntimeError("Not all planned redundant copies were removed.")
    checked(root)
    progress.update({"passed": True, "finished_at": c.utc(),
                     "estimated_bytes_reclaimed": record["estimated_bytes_reclaimable"],
                     "models_predictions_reports_code_unchanged": True,
                     "active_data_and_v6_bundle_verified": True})
    c.write_json(progress_path, progress)
    c.write_json(root / "docs/results/archived_data_cleanup.json", {
        **progress, "note": "Only archived data copies removed; restore recreates them from verified active data. No model or experiment evidence removed."})
    print(json.dumps(progress, indent=2))
    return progress


def restore(root=c.ROOT):
    linux_only()
    root, archived = archive.location(root)
    record, run = load_plan(root)
    if c.running_under([archived]):
        raise RuntimeError("Archived folders are in use.")
    progress_path = run / "progress.json"
    if not progress_path.exists():
        raise RuntimeError("Cleanup has not started; no copies need restoration.")
    consistent(root, record, allow_missing=True)
    required = sum(op["bytes"] for op in record["operations"]
                   if not archive.archived_file(root, op["archive"]).exists())
    if shutil.disk_usage(root).free < required + 100 * 1024**2:
        raise RuntimeError("Insufficient free space to recreate archived copies.")
    for op in record["operations"]:
        source = c.asset(root, op["canonical"])
        if c.digest(source) != op["sha256"]:
            raise RuntimeError("Canonical data changed; cannot restore the recorded bytes.")
        target = archive.archived_file(root, op["archive"])
        c.copy_verified(source, target)
        entry = record["snapshot"]["files"][op["archive"]]
        if (getattr(target.stat(), "st_uid", 0), getattr(target.stat(), "st_gid", 0)) != (entry["uid"], entry["gid"]):
            raise RuntimeError("Restored ownership differs; review before claiming full restoration.")
        os.chmod(target, entry["mode"])
        os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
    consistent(root, record, allow_missing=False)
    checked(root)
    progress = c.read_json(progress_path)
    progress.update({"restored": True, "restored_at": c.utc()})
    c.write_json(progress_path, progress)
    print("Archived data copies restored and every archived file hash verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "plan", "apply", "restore"))
    args = parser.parse_args()
    {"self-test": self_test, "plan": plan, "apply": apply, "restore": restore}[args.action]()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        sys.exit(f"Data-copy cleanup stopped: {error}")
