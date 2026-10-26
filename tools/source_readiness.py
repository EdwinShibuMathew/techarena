"""Validate and package committed source without accessing private model assets."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 2 * 1024**2
PRIVATE_DIRECTORIES = {"data", "artifacts", "local", "archive"}
FORBIDDEN_SUFFIXES = {
    ".csv", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".npz", ".npy",
    ".pt", ".pth", ".cbm", ".log", ".pkl", ".pickle", ".joblib", ".onnx",
    ".safetensors", ".parquet", ".arrow", ".pyc", ".pyo",
}
FROZEN_PATHS = ("legacy", "configs/source_manifest.json", "configs/v6.json")
V6_SHA256 = "c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def validate_files(root, names):
    """Inspect only explicitly tracked files, never walk local asset folders."""
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Unsafe tracked path: {name}")
        if relative.parts[0] in PRIVATE_DIRECTORIES:
            raise ValueError(f"Private asset directory is tracked: {name}")
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Prohibited asset format is tracked: {name}")
        if relative.name == ".env" or relative.name.startswith(".env.") and relative.name != ".env.example":
            raise ValueError(f"Private environment file is tracked: {name}")
        path = root / relative
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Symlinks are not source-release files: {name}")
        if not path.is_file():
            raise ValueError(f"Tracked entry is not a regular file: {name}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Tracked file exceeds 2 MiB: {name}")
        if b"\0" in path.read_bytes():
            raise ValueError(f"Binary content is tracked: {name}")


def tracked_files(root):
    entries = git(root, "ls-files", "--stage", "-z").decode().split("\0")
    names = []
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        mode, _, stage = metadata.split()
        if mode not in {"100644", "100755"} or stage != "0":
            raise ValueError(f"Unsupported or unmerged tracked entry: {name}")
        names.append(name)
    return names


def verify(root=ROOT, base_ref=None):
    root = Path(root).resolve()
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    # Tests and the archive must describe the same committed source tree.
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked source differs from HEAD; commit it before checking readiness")
    names = tracked_files(root)
    validate_files(root, names)
    if base_ref:
        base = git(root, "rev-parse", "--verify", base_ref + "^{commit}").decode().strip()
        changed = git(root, "diff", "--name-only", base, "HEAD", "--", *FROZEN_PATHS).decode().splitlines()
        if changed:
            raise ValueError("Frozen baseline paths changed: " + ", ".join(changed))
    manifest_path = root / "configs/source_manifest.json"
    manifest = json.loads(manifest_path.read_text())["files"]
    legacy = {name for name in names if name.startswith("legacy/") and name.endswith(".py")}
    if not legacy or set(manifest) != legacy:
        raise ValueError("Frozen source manifest does not exactly match tracked legacy Python files")
    for name, expected in manifest.items():
        if sha256((root / name).read_bytes()) != expected:
            raise ValueError(f"Frozen source hash mismatch: {name}")
    config = json.loads((root / "configs/v6.json").read_text())
    if (config.get("submission_sha256") != V6_SHA256
            or config.get("evaluation_rows") != 529390
            or config.get("evaluation_events") != 52939):
        raise ValueError("V6 benchmark fingerprint or evaluation dimensions changed")
    return {
        "git_commit": commit,
        "tracked_files": len(names),
        "frozen_source_files_verified": len(manifest),
        "source_manifest_sha256": sha256(manifest_path.read_bytes()),
        "v6_config_sha256": sha256((root / "configs/v6.json").read_bytes()),
        "v6_submission_sha256": V6_SHA256,
        "evaluation_rows": config["evaluation_rows"],
    }


def validate_archive(root, archive_path, names):
    expected = set(names)
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "techarena-source":
                raise ValueError(f"Unsafe archive entry: {member.name}")
            if member.isdir():
                continue
            name = PurePosixPath(*path.parts[1:]).as_posix()
            if not member.isfile() or name not in expected or name in seen:
                raise ValueError(f"Unexpected archive entry: {member.name}")
            with archive.extractfile(member) as stream:
                if sha256(stream.read()) != sha256((root / name).read_bytes()):
                    raise ValueError(f"Archive content differs from tested source: {name}")
            seen.add(name)
    if seen != expected:
        raise ValueError("Source archive has missing tracked files")


def require_prerequisites(cpu_test_result, source_integrity_result, model_test_result):
    if any(result != "success" for result in (cpu_test_result, source_integrity_result, model_test_result)):
        raise ValueError(f"Readiness blocked: tests={cpu_test_result}, integrity={source_integrity_result}, models={model_test_result}")


def package(root, output_dir, cpu_test_result, source_integrity_result, base_ref=None, *, model_test_result=None):
    require_prerequisites(cpu_test_result, source_integrity_result, model_test_result)
    root = Path(root).resolve()
    report = verify(root, base_ref)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "techarena-source.tar.gz"
    receipt = output_dir / "source-readiness.json"
    sums = output_dir / "SHA256SUMS"
    if any(path.exists() for path in (archive, receipt, sums)):
        raise FileExistsError("Use a new output directory; existing release files are preserved")
    git(root, "archive", "--format=tar.gz", "--prefix=techarena-source/",
        "--output=" + str(archive), report["git_commit"])
    validate_archive(root, archive, tracked_files(root))
    # Detect source changes during packaging before issuing a readiness receipt.
    if verify(root, base_ref) != report:
        raise ValueError("Source changed during packaging")
    report.update({
        "schema_version": 1,
        "kind": "source_readiness",
        "checks": {"cpu_tests": cpu_test_result, "source_integrity": source_integrity_result, "model_contracts": model_test_result, "source_archive": "success"},
        "archive": {"filename": archive.name, "sha256": sha256(archive.read_bytes())},
        "model_inference_performed": False,
        "private_assets_verified": False,
        "competition_submission_created": False,
    })
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with sums.open("x", encoding="utf-8") as stream:
        for path in (archive, receipt):
            stream.write(f"{sha256(path.read_bytes())}  {path.name}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("verify", "package"):
        command = actions.add_parser(action)
        command.add_argument("--base-ref", help="PR base or previous main commit; baseline paths cannot change")
        if action == "package":
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--cpu-test-result", required=True,
                                 choices=["success", "failure", "cancelled", "skipped"])
            command.add_argument("--model-test-result", required=True, choices=["success", "failure", "cancelled", "skipped"])
            command.add_argument("--source-integrity-result", required=True,
                                 choices=["success", "failure", "cancelled", "skipped"])
    args = parser.parse_args()
    try:
        report = (verify(ROOT, args.base_ref) if args.action == "verify" else
                  package(ROOT, args.output_dir, args.cpu_test_result, args.source_integrity_result, args.base_ref, model_test_result=args.model_test_result))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Source readiness failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
