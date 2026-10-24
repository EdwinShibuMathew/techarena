"""Reclaim identical archived binary copies; retain every path and unique file.

Run `plan`, inspect the report, then `apply`. Only archived files are relinked.
The live data/artifacts and V5/V6 source files are never used as link targets.
"""

import argparse
import collections
import json
import os
import re
import stat
import sys
import uuid
import unittest
from pathlib import Path

from consolidate import ROOT, digest, read_json, require_receipt, running_under, utc, write_json

SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".cbm", ".csv", ".zip"}
MINIMUM_BYTES = 1024 * 1024


def self_test():
    if os.name != "posix":
        raise RuntimeError("Run the real hardlink self-test in WSL/Linux.")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_archive_deduplication.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("All archive cleanup tests must pass with no skips in WSL.")
    write_json(ROOT / "local/archive_dedup_self_test.json", {
        "passed": True, "tests": result.testsRun, "finished_at": utc(),
        "script_sha256": digest(ROOT / "tools/deduplicate_archive.py"),
        "tests_sha256": digest(ROOT / "tests/test_archive_deduplication.py")})


def require_self_test(root):
    receipt = read_json(root / "local/archive_dedup_self_test.json")
    if (receipt.get("passed") is not True
            or receipt["script_sha256"] != digest(root / "tools/deduplicate_archive.py")
            or receipt["tests_sha256"] != digest(root / "tests/test_archive_deduplication.py")):
        raise RuntimeError("Run the current cleanup self-test successfully in WSL first.")


def within(path, parent):
    return path.is_relative_to(parent)


def location(root):
    root = Path(root).resolve(strict=True)
    archive = root / "local/archive"
    # Do not resolve away a symlink in either ancestor.
    for folder in (root / "local", archive):
        if folder.is_symlink() or not folder.is_dir():
            raise RuntimeError(f"Expected a plain directory: {folder}")
    return root, archive


def archived_file(root, relative):
    root, archive = location(root)
    relative = Path(relative)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or not within(path, archive):
        raise RuntimeError(f"Path outside archive: {relative}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"Symlink in file operation path: {current}")
        if current.exists() and os.path.ismount(current):
            raise RuntimeError(f"Mount in file operation path: {current}")
    return path


def topology(root):
    root, archive = location(root)
    journal = read_json(root / "local/archive_journal.json")
    if (journal.get("restored") or journal["repository"] != str(root)
            or journal["projects"] != str(root.parent) or not journal["moves"]):
        raise RuntimeError("A completed archive in this repository is required.")
    expected = set()
    for move in journal["moves"]:
        target = Path(move["destination"])
        if (not move["done"] or target.parent != archive or target.is_symlink()
                or not target.is_dir() or Path(move["source"]).exists()):
            raise RuntimeError("Archive journal is incomplete or the old layout reappeared.")
        st = target.stat()
        if [st.st_dev, st.st_ino] != move["identity"]:
            raise RuntimeError(f"Archived directory identity changed: {target}")
        expected.add(target.name)
    if set(p.name for p in archive.iterdir()) != expected:
        raise RuntimeError("Archive contains entries not accounted for in its journal.")
    return archive


def fingerprint(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Expected regular file: {path}")
    sha = digest(path)
    after = path.lstat()
    attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in attributes):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"sha256": sha, "bytes": after.st_size,
            "inode": [after.st_dev, after.st_ino], "links": after.st_nlink,
            "allocated_bytes": getattr(after, "st_blocks", 0) * 512,
            "mode": stat.S_IMODE(after.st_mode), "uid": getattr(after, "st_uid", 0),
            "gid": getattr(after, "st_gid", 0), "mtime_ns": after.st_mtime_ns}


def snapshot(root):
    archive = topology(root)
    files, links = {}, {}

    def fail(error):
        raise error

    for directory, directories, names in os.walk(archive, followlinks=False, onerror=fail):
        here = Path(directory)
        if os.path.ismount(here) or (here / ".git").exists():
            raise RuntimeError(f"Nested mount or Git checkout: {here}")
        for name in directories + names:
            path = here / name
            if path.is_symlink():
                links[path.relative_to(root).as_posix()] = os.readlink(path)
        directories[:] = [name for name in directories if not (here / name).is_symlink()]
        for name in names:
            path = here / name
            if path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            files[relative] = fingerprint(archived_file(root, relative))
    return {"files": files, "symlinks": links}


def operations(state, run_id):
    # Refuse to share any inode with files outside the archive, including live assets.
    inode_paths = collections.defaultdict(list)
    for name, entry in state["files"].items():
        inode_paths[tuple(entry["inode"])].append(name)
    groups = collections.defaultdict(list)
    for name, entry in state["files"].items():
        if Path(name).suffix.lower() not in SUFFIXES or entry["bytes"] < MINIMUM_BYTES:
            continue
        inode = tuple(entry["inode"])
        if entry["links"] != len(inode_paths[inode]):
            continue
        key = (entry["sha256"], entry["bytes"], entry["mode"], entry["uid"], entry["gid"], inode[0])
        groups[key].append(name)
    result, freed_inodes = [], {}
    for group in groups.values():
        # Prefer an inode that already has several archived names.
        keeper = min(group, key=lambda p: (-len(inode_paths[tuple(state["files"][p]["inode"])]), p))
        keeper_inode = state["files"][keeper]["inode"]
        for target in sorted(group):
            original = state["files"][target]
            if original["inode"] == keeper_inode:
                continue
            temporary = str(Path(target).parent / f".techarena-dedup-{run_id}-{len(result)}.tmp").replace("\\", "/")
            result.append({"keeper": keeper, "target": target, "temporary": temporary,
                           "sha256": original["sha256"]})
            freed_inodes[tuple(original["inode"])] = original
    # An inode only disappears if EVERY archived name for it will be replaced.
    replaced = {entry["target"] for entry in result}
    estimated = sum(entry["allocated_bytes"] for inode, entry in freed_inodes.items()
                    if set(inode_paths[inode]).issubset(replaced))
    return result, estimated


def make_plan(root=ROOT):
    root, _ = location(root)
    require_self_test(root)
    require_receipt(root)
    plan_path = root / "local/archive_dedup_plan.json"
    if plan_path.exists():
        previous = read_json(plan_path)
        progress = root / "local/archive-dedup" / previous["id"] / "progress.json"
        if progress.exists() and not read_json(progress).get("passed"):
            raise RuntimeError("An unfinished cleanup exists. Run apply to resume it before making another plan.")
    print("Hashing archived files; unique files and all report contents will be retained.", flush=True)
    state = snapshot(root)
    run_id = uuid.uuid4().hex
    changes, estimated = operations(state, run_id)
    plan = {"version": 1, "id": run_id, "created_at": utc(), "root": str(root),
            "archive_journal_sha256": digest(root / "local/archive_journal.json"),
            "bundle_manifest_sha256": digest(root / "artifacts/v6/manifest.json"),
            "cleanup_script_sha256": digest(root / "tools/deduplicate_archive.py"),
            "snapshot": state, "operations": changes,
            "estimated_allocated_bytes_reclaimable": estimated}
    run_directory = root / "local/archive-dedup" / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)
    write_json(run_directory / "plan.json", plan)
    write_json(plan_path, plan)
    print(json.dumps({"archive_files": len(state["files"]), "symlinks_preserved": len(state["symlinks"]),
        "duplicate_paths_to_relink": len(changes), "unique_files_deleted": 0,
        "estimated_reclaimable_GiB": round(estimated / 1024**3, 3),
        "plan": str(plan_path), "applied": False}, indent=2))
    return plan


def same_contents(expected, actual):
    if expected["symlinks"] != actual["symlinks"] or set(expected["files"]) != set(actual["files"]):
        raise RuntimeError("Archived filenames or symbolic-link targets changed. Nothing further will be relinked.")
    for name, entry in expected["files"].items():
        other = actual["files"][name]
        if any(entry[key] != other[key] for key in ("sha256", "bytes", "mode", "uid", "gid")):
            raise RuntimeError(f"Archived file content/permissions changed: {name}")


def check_operations(plan):
    original = plan["snapshot"]["files"]
    names, temporaries = set(), set()
    for operation in plan["operations"]:
        keeper, target, temporary = (operation[k] for k in ("keeper", "target", "temporary"))
        if keeper == target or target in names or temporary in temporaries or temporary in original:
            raise RuntimeError("Ambiguous duplicate operation.")
        if keeper not in original or target not in original or Path(target).suffix.lower() not in SUFFIXES:
            raise RuntimeError("Cleanup operation is not an approved archived binary duplicate.")
        left, right = original[keeper], original[target]
        if right["bytes"] < MINIMUM_BYTES or operation["sha256"] != left["sha256"]:
            raise RuntimeError("Invalid duplicate fingerprint.")
        if any(left[k] != right[k] for k in ("sha256", "bytes", "mode", "uid", "gid")):
            raise RuntimeError("Duplicate bytes or permissions do not match.")
        if Path(temporary).parent != Path(target).parent or not Path(temporary).name.startswith(f'.techarena-dedup-{plan["id"]}-'):
            raise RuntimeError("Unsafe temporary link path.")
        names.add(target)
        temporaries.add(temporary)
    if any(operation["keeper"] in names for operation in plan["operations"]):
        raise RuntimeError("A keeper may not itself be replaced.")


def apply_plan(root=ROOT):
    if os.name != "posix":
        raise RuntimeError("Run cleanup in WSL/Linux, not Windows.")
    root, archive = location(root)
    require_self_test(root)
    require_receipt(root)
    topology(root)
    blockers = running_under([archive])
    if blockers:
        raise RuntimeError("Archived folders are in use: " + json.dumps(blockers))
    plan = read_json(root / "local/archive_dedup_plan.json")
    if (plan["root"] != str(root) or plan["version"] != 1
            or not re.fullmatch(r"[0-9a-f]{32}", plan["id"])):
        raise RuntimeError("Cleanup plan belongs to a different location/version.")
    for relative, field in [("local/archive_journal.json", "archive_journal_sha256"),
                            ("artifacts/v6/manifest.json", "bundle_manifest_sha256"),
                            ("tools/deduplicate_archive.py", "cleanup_script_sha256")]:
        if digest(root / relative) != plan[field]:
            raise RuntimeError(f"Plan evidence changed: {relative}. Make a new plan.")
    run_directory = root / "local/archive-dedup" / plan["id"]
    if (not run_directory.is_dir() or run_directory.is_symlink()
            or read_json(run_directory / "plan.json") != plan):
        raise RuntimeError("Saved plan and run record do not match.")
    check_operations(plan)
    # Recover only known temporary hardlinks from an interrupted atomic replacement.
    for operation in plan["operations"]:
        temporary = archived_file(root, operation["temporary"])
        if temporary.exists():
            keeper = archived_file(root, operation["keeper"])
            if not temporary.is_file() or not os.path.samefile(keeper, temporary) or digest(temporary) != operation["sha256"]:
                raise RuntimeError(f"Unexpected temporary file: {temporary}")
            temporary.unlink()
    current = snapshot(root)
    same_contents(plan["snapshot"], current)
    # Every inode must still be its original inode or its planned keeper's inode.
    keepers = {entry["target"]: entry["keeper"] for entry in plan["operations"]}
    inode_names = collections.Counter(tuple(entry["inode"]) for entry in current["files"].values())
    for name, entry in current["files"].items():
        original = plan["snapshot"]["files"][name]
        allowed = [original["inode"]]
        if name in keepers:
            allowed.append(plan["snapshot"]["files"][keepers[name]]["inode"])
        if entry["inode"] not in allowed:
            raise RuntimeError(f"Unplanned file replacement: {name}")
    for operation in plan["operations"]:
        for field in ("keeper", "target"):
            entry = current["files"][operation[field]]
            if entry["links"] != inode_names[tuple(entry["inode"])]:
                raise RuntimeError("A planned inode is linked outside the archive. No relinking performed.")
    progress_path = run_directory / "progress.json"
    if progress_path.exists():
        previous = read_json(progress_path)
        if previous.get("passed") and "summary" in previous:
            print("Cleanup already completed; all archived bytes and live assets verified.")
            print(json.dumps(previous["summary"], indent=2))
            return previous["summary"]
    progress = {"id": plan["id"], "started_at": utc(), "passed": False, "completed_operations": 0}
    write_json(progress_path, progress)
    for index, operation in enumerate(plan["operations"]):
        keeper = archived_file(root, operation["keeper"])
        target = archived_file(root, operation["target"])
        temporary = archived_file(root, operation["temporary"])
        if not os.path.samefile(keeper, target):
            if digest(keeper) != operation["sha256"] or digest(target) != operation["sha256"]:
                raise RuntimeError("Duplicate changed immediately before relinking; stopped.")
            # No unlink-first operation: an atomic replace retains the target path even on interruption.
            os.link(keeper, temporary, follow_symlinks=False)
            if digest(temporary) != operation["sha256"] or digest(target) != operation["sha256"]:
                raise RuntimeError("Duplicate changed during relinking; stopped with recoverable temporary link.")
            os.replace(temporary, target)
        progress["completed_operations"] = index + 1
        write_json(progress_path, progress)
    after = snapshot(root)
    same_contents(plan["snapshot"], after)
    require_receipt(root)
    original_inodes = {tuple(v["inode"]): v for v in current["files"].values()}
    final_inodes = {tuple(v["inode"]): v for v in after["files"].values()}
    released = sum(v["allocated_bytes"] for inode, v in original_inodes.items() if inode not in final_inodes)
    progress.update({"passed": True, "finished_at": utc(), "all_archive_file_hashes_unchanged": True,
                     "all_archive_symlinks_unchanged": True, "live_bundle_verified": True,
                     "allocated_bytes_of_inodes_released_this_run": released})
    summary = {"cleanup": "archive-only exact binary deduplication", "id": plan["id"],
        "finished_at": progress["finished_at"], "passed": True,
        "archive_files_retained": len(after["files"]), "unique_files_deleted": 0,
        "duplicate_paths_sharing_storage": len(plan["operations"]),
        "estimated_allocated_bytes_reclaimable_in_plan": plan["estimated_allocated_bytes_reclaimable"],
        "allocated_bytes_of_inodes_released_this_run": released,
        "all_archive_file_hashes_and_symlinks_unchanged": True, "live_bundle_verified": True,
        "note": "Archive duplicates now share storage. Copy archived files before editing. WSL free space differs from Windows VHD size."}
    write_json(root / "docs/results/archive_deduplication.json", summary)
    progress["summary"] = summary
    write_json(progress_path, progress)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "plan", "apply"))
    args = parser.parse_args()
    if args.action == "self-test":
        self_test()
    elif args.action == "plan":
        make_plan()
    else:
        apply_plan()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        sys.exit(f"Cleanup stopped: {error}")
