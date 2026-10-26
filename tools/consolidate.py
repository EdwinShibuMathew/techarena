"""Inventory, verified asset copying and reversible WSL experiment relocation.

No command deletes an experiment. Archive/restore print a plan unless --apply.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = [
    ("dim8_seed42", "techarena-row-auc-final/seed42/work", "event_best.pt", 8),
    ("dim8_seed43", "techarena-row-auc-final/seed43/work", "event_best.pt", 8),
    ("dim8_seed44", "techarena-seed44-trial/final/work/seed44", "event_final.pt", 8),
    ("dim16_seed42", "techarena-dim16-trial/final/work/seed42", "event_final.pt", 16),
    ("dim16_seed43", "techarena-dim16-trial/final/work/seed43", "event_final.pt", 16),
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                prefix=".receipt-", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_manifest_binding(root, manifest):
    """Accept exact bytes or a uniform LF/CRLF-only representation change."""
    content = (Path(root) / "configs/source_manifest.json").read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    declared = manifest["source_manifest_sha256"]
    method = "exact"
    if actual != declared:
        lf = content.replace(b"\r\n", b"\n")
        crlf = lf.replace(b"\n", b"\r\n")
        if (b"\r" in lf or content not in (lf, crlf)
                or declared not in {hashlib.sha256(v).hexdigest() for v in (lf, crlf)}):
            raise RuntimeError("Historical source manifest changed.")
        method = "lf-crlf-only"
    return {"declared_sha256": declared, "checkout_sha256": actual, "method": method}


def inside(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def posix():
    return os.name == "posix"


def asset(root, relative):
    path = root / relative
    if not inside(path, root) or Path(relative).is_absolute():
        raise ValueError(f"Asset path escapes repository: {relative}")
    return path


def candidates(projects, root=ROOT):
    projects = Path(projects).resolve(strict=True)
    return sorted([p for p in projects.iterdir() if (p.name == "techarena" or p.name.startswith("techarena-"))
                   and p != root.resolve() and (p.is_dir() or p.is_symlink())], key=lambda p: p.name)


def inventory(projects, root=ROOT):
    rows = []
    for folder in candidates(projects, root):
        entry = {"path": str(folder), "symlink_root": folder.is_symlink(),
                 "git_checkout": (folder / ".git").exists(), "regular_files": 0,
                 "logical_bytes": 0, "allocated_bytes_unique_in_folder": 0, "links": [], "large_files": []}
        seen = set()
        if not folder.is_symlink():
            for directory, names, files in os.walk(folder, followlinks=False):
                for name in names + files:
                    path = Path(directory) / name
                    if path.is_symlink():
                        entry["links"].append({"path": str(path.relative_to(folder)), "target": os.readlink(path), "exists": path.exists()})
                for name in files:
                    path = Path(directory) / name
                    if path.is_symlink() or not path.is_file():
                        continue
                    st = path.stat()
                    entry["regular_files"] += 1
                    entry["logical_bytes"] += st.st_size
                    key = (st.st_dev, st.st_ino)
                    if key not in seen:
                        entry["allocated_bytes_unique_in_folder"] += getattr(st, "st_blocks", (st.st_size + 511) // 512) * 512
                        seen.add(key)
                    if st.st_size >= 25 * 1024 * 1024:
                        entry["large_files"].append({"path": str(path.relative_to(folder)), "bytes": st.st_size})
        rows.append(entry)
    report = {"created_at": utc(), "projects": str(Path(projects).resolve()), "folders": rows,
              "note": "No links followed; per-folder allocation does not deduplicate hardlinks shared across folders."}
    write_json(root / "local/inventory.json", report)
    for row in rows:
        flags = " [Git checkout: excluded from automatic archive]" if row["git_checkout"] else " [symlink]" if row["symlink_root"] else ""
        print(f'{row["logical_bytes"] / 1024**3:8.2f} GiB  {Path(row["path"]).name}{flags}')
    print(f"Inventory saved: {root / 'local/inventory.json'}")
    return report


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Required source missing: {source}")
    expected = digest(source)
    if destination.exists():
        if not destination.is_file() or digest(destination) != expected:
            raise FileExistsError(f"Different destination already exists: {destination}")
        return expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".copying")
    shutil.copyfile(source, temporary)
    if digest(temporary) != expected or digest(source) != expected:
        raise RuntimeError(f"Copy verification failed or source changed: {source}")
    temporary.replace(destination)
    return expected


def bundle(projects, root=ROOT):
    projects = Path(projects).resolve(strict=True)
    manifest_path = root / "artifacts/v6/manifest.json"
    if manifest_path.exists():
        verify_bundle(root)
        print("Existing bundle verified. No original folders were moved.")
        return
    old = projects / "techarena-final"
    final = projects / "techarena-row-auc-final"
    checks = read_json(final / "outputs/candidate_submission_checks.json")
    if not checks["submission_ready"] or not checks["v5_files_unchanged"] or not checks["validation_pipeline_check_passed"]:
        raise RuntimeError("V6 original checks are not passing.")
    config = read_json(root / "configs/v6.json")
    if checks["submission_sha256"] != config["submission_sha256"]:
        raise RuntimeError("V6 submission identity differs from the known benchmark.")
    v5checks = read_json(old / "outputs/five_model_submission_checks.json")
    planned, members = [], []
    for name in ["train_events.npz", "evaluation_events.npz", "evaluation_rowids.npy", "preference_training.npz"]:
        planned.append((old / "data/cache" / name, "data/cache/" + name))
    raw = projects / "techarena/data/raw"
    planned.append((raw / "example_submission.csv", "data/raw/example_submission.csv"))
    for name in ["train.zip", "evaluation.zip"]:
        if (raw / name).is_file():
            planned.append((raw / name, "data/raw/" + name))
    planned += [(old / "outputs/techarena_submission_v5.csv", "artifacts/submissions/v5.csv"),
                (final / "outputs/techarena_submission_v6_candidate.csv", "artifacts/submissions/v6.csv")]
    for key, original, checkpoint, dimension in MEMBERS:
        folder = projects / original
        prefix = "artifacts/v6/" + key + "/"
        record = {"key": key, "dimension": dimension}
        preference_name = "preference_best.pt" if checkpoint == "event_best.pt" else "preference_final.pt"
        planned.append((folder / preference_name, prefix + "preference.pt"))
        record["preference"] = prefix + "preference.pt"
        for source_name, target_name, field in [(checkpoint, "event.pt", "checkpoint"),
                ("combiner_used.cbm", "combiner.cbm", "combiner"),
                ("evaluation_predictions.npz", "reference_predictions.npz", "reference")]:
            planned.append((folder / source_name, prefix + target_name))
            record[field] = prefix + target_name
        members.append(record)
    # Validate every required source before making any expensive copies.
    for source, relative in planned:
        if not source.is_file():
            raise FileNotFoundError(source)
        asset(root, relative)
    required_bytes = sum(p.stat().st_size for p, relative in planned if not asset(root, relative).exists())
    if shutil.disk_usage(root).free < required_bytes + 100 * 1024**2:
        raise RuntimeError(f"Not enough free space for verified copies ({required_bytes / 1024**3:.2f} GiB plus margin).")
    files = {}
    for source, relative in planned:
        print(f"Collecting {relative}", flush=True)
        expected = copy_verified(source, asset(root, relative))
        files[relative] = {"sha256": expected, "bytes": source.stat().st_size, "original": str(source)}
    if files["artifacts/submissions/v6.csv"]["sha256"] != config["submission_sha256"]:
        raise RuntimeError("Original V6 CSV differs from the accepted fingerprint.")
    if files["artifacts/submissions/v5.csv"]["sha256"] != v5checks["submission_sha256"]:
        raise RuntimeError("Original V5 CSV differs from its recorded fingerprint.")
    # Keep small original JSON evidence with the team source, not tensor arrays.
    report_names = {"ensemble_trial_results.json", "paired_results.json", "baseline_results.json", "baseline_summary.json",
                    "candidate_submission_checks.json", "five_model_submission_checks.json", "results.json"}
    for folder in candidates(projects, root):
        if folder.is_symlink():
            continue
        for name in report_names:
            source = folder / "outputs" / name
            if source.is_file() and source.stat().st_size < 2 * 1024**2:
                copy_verified(source, root / "docs/results" / folder.name / name)
    environment = {"python": sys.version, "packages": {}}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "torch", "triton"]:
        try:
            environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment["packages"][name] = None
    write_json(root / "local/environment.json", environment)
    copy_verified(root / "local/environment.json", root / "docs/runtime_versions.json")
    manifest = {"version": "portable_v6_assets_v1", "created_at": utc(), "source_projects": str(projects),
        "config_sha256": digest(root / "configs/v6.json"), "source_manifest_sha256": digest(root / "configs/source_manifest.json"),
        "members": members, "files": files, "expected_submission_sha256": config["submission_sha256"]}
    write_json(manifest_path, manifest)
    verify_bundle(root)
    print("Bundle verified. Originals are intact. Next: python tools/predict_v6.py")


def verify_bundle(root=ROOT):
    manifest = read_json(root / "artifacts/v6/manifest.json")
    config = read_json(root / "configs/v6.json")
    if manifest["config_sha256"] != digest(root / "configs/v6.json") or manifest["expected_submission_sha256"] != config["submission_sha256"]:
        raise RuntimeError("Bundle configuration changed.")
    source_manifest_binding(root, manifest)
    source_manifest = read_json(root / "configs/source_manifest.json")
    for relative, expected in source_manifest["files"].items():
        if digest(asset(root, relative)) != expected:
            raise RuntimeError(f"Historical source changed: {relative}")
    if [m["key"] for m in manifest["members"]] != config["members"]:
        raise RuntimeError("Member order changed.")
    for relative, info in manifest["files"].items():
        path = asset(root, relative)
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise RuntimeError(f"Collected asset changed: {relative}")
    for member in manifest["members"]:
        for field in ["checkpoint", "preference", "combiner", "reference"]:
            if member[field] not in manifest["files"]:
                raise RuntimeError(f"Unverified member asset: {member[field]}")
    print(f"Verified {len(manifest['files'])} collected assets and original source fingerprints.", flush=True)
    return manifest


def require_receipt(root):
    verify_bundle(root)
    receipt = read_json(root / "local/v6_relocation_check.json")
    if "source_manifest_binding" in receipt:
        if receipt["source_manifest_binding"] != source_manifest_binding(root, read_json(root / "artifacts/v6/manifest.json")):
            raise RuntimeError("Replay source manifest binding changed.")
    if receipt["passed"] is not True or receipt["bundle_manifest_sha256"] != digest(root / "artifacts/v6/manifest.json"):
        raise RuntimeError("A current successful portable V6 replay is required before archive.")
    config = read_json(root / "configs/v6.json")
    if receipt["submission_sha256"] != config["submission_sha256"]:
        raise RuntimeError("Portable CSV did not reproduce V6.")
    if digest(asset(root, receipt["output_relative"])) != config["submission_sha256"]:
        raise RuntimeError("Portable CSV changed after verification.")
    differences = receipt["member_max_abs_differences"]
    if set(differences) != set(config["members"]) or any(not math.isfinite(v) or v < 0 or v > 1e-15 for v in differences.values()):
        raise RuntimeError("All five exact member replays must be recorded.")
    required_code = {"tools/predict_v6.py", "tools/consolidate.py", "configs/v6.json", "configs/source_manifest.json"}
    required_code.update(read_json(root / "configs/source_manifest.json")["files"])
    if set(receipt["code_sha256"]) != required_code:
        raise RuntimeError("Replay receipt must bind all portable prediction code.")
    for relative, expected in receipt["code_sha256"].items():
        if digest(asset(root, relative)) != expected:
            raise RuntimeError(f"Prediction code changed after replay: {relative}")


def running_under(folders):
    """Inspect only this user's process paths, never process arguments/secrets."""
    blockers = []
    proc = Path("/proc")
    if not proc.exists():
        raise RuntimeError("Archive requires WSL/Linux process inspection.")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            cwd = Path(os.readlink(entry / "cwd"))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(inside(cwd, folder) for folder in folders):
            blockers.append({"pid": int(entry.name), "working_directory": str(cwd)})
    return blockers


def safe_source(path, projects, root):
    path = Path(path)
    if path.is_symlink() or path.parent.resolve() != projects.resolve() or not (path.name == "techarena" or path.name.startswith("techarena-")):
        raise RuntimeError(f"Not an eligible plain TechArena project directory: {path}")
    if path.resolve() == root.resolve() or inside(root, path):
        raise RuntimeError("Cannot archive the current repository or its ancestor.")
    if not path.is_dir() or (path / ".git").exists():
        raise RuntimeError(f"Missing directory or Git checkout needs separate review: {path}")
    # Root rename preserves symlink bytes. Nested mounts/Git checkouts need separate review.
    for directory, names, _ in os.walk(path, followlinks=False):
        here = Path(directory)
        if os.path.ismount(here) or (here / ".git").exists():
            raise RuntimeError(f"Nested mount/Git checkout cannot be relocated automatically: {here}")
        names[:] = [n for n in names if not (here / n).is_symlink()]


def archive(projects, apply=False, root=ROOT):
    projects = Path(projects).resolve(strict=True)
    if not posix() or root.resolve().parent != projects:
        raise RuntimeError("Run archive in the WSL repository directly inside the selected projects directory.")
    require_receipt(root)
    journal_path = root / "local/archive_journal.json"
    archive_root = root / "local/archive"
    if archive_root.is_symlink() or not inside(archive_root, root):
        raise RuntimeError("Archive location must be inside this repository, without a symlink root.")
    if journal_path.exists():
        journal = read_json(journal_path)
        if journal["projects"] != str(projects) or journal["repository"] != str(root.resolve()):
            raise RuntimeError("Archive journal belongs to a different location.")
        if journal.get("restored"):
            raise RuntimeError("This archive was restored. Review the journal before planning another migration.")
    else:
        chosen, excluded = [], []
        for path in candidates(projects, root):
            if path.is_symlink() or (path / ".git").exists():
                excluded.append(str(path))
                continue
            safe_source(path, projects, root)
            target = archive_root / path.name
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            st = path.stat()
            chosen.append({"source": str(path), "destination": str(target), "done": False, "identity": [st.st_dev, st.st_ino]})
        journal = {"created_at": utc(), "projects": str(projects), "repository": str(root.resolve()),
                   "moves": chosen, "excluded": excluded, "restored": False}
    for item in journal["moves"]:
        source, target = Path(item["source"]), Path(item["destination"])
        if not inside(target, archive_root) or target.parent.resolve() != archive_root.resolve() or source.parent.resolve() != projects:
            raise RuntimeError("Unsafe archive journal path.")
        if source.name != target.name or source == root.resolve() or not (source.name == "techarena" or source.name.startswith("techarena-")):
            raise RuntimeError("Unsafe archive journal mapping.")
        print(f"{source} -> {target}" + (" [completed]" if item["done"] else ""))
    if journal["excluded"]:
        print("Excluded Git checkouts/symlink roots:", json.dumps(journal["excluded"], indent=2))
    if not apply:
        print("Plan only. No folders moved. Review it, then use archive --apply.")
        return journal
    pending = []
    for item in journal["moves"]:
        source, target = Path(item["source"]), Path(item["destination"])
        if source.exists() and not target.exists():
            safe_source(source, projects, root)
            if [source.stat().st_dev, source.stat().st_ino] != item["identity"]:
                raise RuntimeError(f"Source directory changed identity: {source}")
            if source.stat().st_dev != root.stat().st_dev:
                raise RuntimeError("Archive must remain on the same filesystem.")
            pending.append(source)
        elif not source.exists() and target.is_dir() and not target.is_symlink():
            if [target.stat().st_dev, target.stat().st_ino] != item["identity"]:
                raise RuntimeError(f"Archive directory changed identity: {target}")
            item["done"] = True  # interrupted after rename, before journal update
        else:
            raise RuntimeError(f"Ambiguous archive state: {source} / {target}")
    blockers = running_under(pending)
    if blockers:
        raise RuntimeError("Move terminals out of old folders and stop their running jobs first: " + json.dumps(blockers))
    archive_root.mkdir(parents=True, exist_ok=True)
    write_json(journal_path, journal)
    for item in journal["moves"]:
        if item["done"]:
            continue
        source, target = Path(item["source"]), Path(item["destination"])
        source.rename(target)
        item["done"] = True
        write_json(journal_path, journal)
    print("Archive complete. Nothing deleted; old experiments are inside local/archive/.")
    return journal


def restore(apply=False, root=ROOT):
    path = root / "local/archive_journal.json"
    journal = read_json(path)
    projects = Path(journal["projects"]).resolve()
    if not posix() or journal["repository"] != str(root.resolve()) or root.resolve().parent != projects:
        raise RuntimeError("Restore must run in the original WSL repository.")
    if journal.get("restored"):
        print("Archive already restored.")
        return
    pending = []
    for item in journal["moves"]:
        source, target = Path(item["source"]), Path(item["destination"])
        if source.parent.resolve() != projects or target.parent.resolve() != (root / "local/archive").resolve() or source.name != target.name or source == root.resolve() or not (source.name == "techarena" or source.name.startswith("techarena-")) or not inside(target, root):
            raise RuntimeError("Unsafe restore journal path.")
        if target.is_dir() and not target.is_symlink() and not source.exists() and not source.is_symlink():
            if [target.stat().st_dev, target.stat().st_ino] != item["identity"]:
                raise RuntimeError(f"Archive identity changed: {target}")
            print(f"{target} -> {source}")
            pending.append((source, target, item))
        elif source.is_dir() and not source.is_symlink() and not target.exists():
            if [source.stat().st_dev, source.stat().st_ino] != item["identity"]:
                raise RuntimeError(f"Original location occupied by a different directory: {source}")
            item["done"] = False  # interrupted restore after rename
        else:
            raise RuntimeError(f"Original path occupied or archive missing: {source}")
    if not apply:
        print("Plan only. Use restore --apply to reverse these moves.")
        return
    blockers = running_under([target for _, target, _ in pending])
    if blockers:
        raise RuntimeError("Archived folders are in use: " + json.dumps(blockers))
    for source, target, item in reversed(pending):
        target.rename(source)
        item["done"] = False
        write_json(path, journal)
    journal["restored"] = True
    write_json(path, journal)
    print("Original directory layout restored.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "bundle", "verify", "archive", "restore"])
    parser.add_argument("--projects", type=Path, default=Path.home() / "projects")
    parser.add_argument("--apply", action="store_true", help="Apply archive/restore after reviewing the default plan.")
    args = parser.parse_args()
    if args.apply and args.action not in ["archive", "restore"]:
        parser.error("--apply is only for archive/restore")
    if args.action == "inventory":
        inventory(args.projects)
    elif args.action == "bundle":
        bundle(args.projects)
    elif args.action == "verify":
        verify_bundle()
    elif args.action == "archive":
        archive(args.projects, args.apply)
    else:
        restore(args.apply)


if __name__ == "__main__":
    main()
