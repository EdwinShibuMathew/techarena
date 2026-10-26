"""Publish an existing private repository's feature branch for pull-request review."""

import json
import subprocess
from pathlib import Path
from urllib.parse import quote

from consolidate import ROOT, digest, read_json
from source_readiness import validate_files

OWNER, NAME = "EdwinShibuMathew", "techarena"


def run(args, capture=True):
    result = subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def main():
    # Refuse protected/detached branches before staging, committing or networking.
    top = Path(run(["git", "rev-parse", "--show-toplevel"])).resolve()
    if top != ROOT.resolve():
        raise SystemExit("Not an independent repository root; refusing to stage parent files.")
    branch = run(["git", "branch", "--show-current"])
    if not branch or branch == "main":
        raise SystemExit("Create or switch to a feature branch first. Publication uses a pull request into main.")
    for relative, expected in read_json(ROOT / "configs/source_manifest.json")["files"].items():
        if digest(ROOT / relative) != expected:
            raise SystemExit(f"Historical source changed before publication: {relative}")
    try:
        profile = json.loads(run(["gh", "api", "user"]))
    except FileNotFoundError:
        raise SystemExit("GitHub CLI (gh) is unavailable in this terminal. Use the WSL terminal where you completed sign-in.")
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        raise SystemExit("GitHub account check failed. Run gh auth status in this same WSL terminal. Details: " + detail)
    if profile["login"].lower() != OWNER.lower():
        raise SystemExit(f"Signed into {profile['login']}; expected {OWNER}. No repository was created.")
    full_name = OWNER + "/" + NAME
    repo = json.loads(run(["gh", "repo", "view", full_name, "--json", "nameWithOwner,isPrivate,url"]))
    if not repo["isPrivate"] or repo["nameWithOwner"].lower() != full_name.lower():
        raise SystemExit("Repository privacy/identity check failed. No push performed.")
    remotes = run(["git", "remote"]).splitlines()
    url = repo["url"] + ".git"
    if "origin" not in remotes:
        raise SystemExit("Configure origin for the existing private repository before publication.")
    old = run(["git", "remote", "get-url", "origin"])
    if old not in [url, repo["url"], f"git@github.com:{full_name}.git"]:
        raise SystemExit(f"Origin points elsewhere ({old}). No remote was changed.")
    paths = ["README.md", ".gitignore", ".gitattributes", ".github", "requirements.txt", "requirements-ci.txt",
             "docs", "configs", "tools", "tests", "legacy"]
    if (ROOT / "experiments").exists():
        paths.append("experiments")
    # Include future explicit environment locks without staging arbitrary files.
    paths.extend(p.name for p in sorted(ROOT.glob("requirements-*.txt")) if p.name not in paths)
    run(["git", "add", *paths])
    try:
        validate_files(ROOT, list(filter(None, run(["git", "ls-files", "-z"]).split("\0"))))
    except ValueError as error:
        raise SystemExit(f"{error}. No push performed.") from error
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if staged.returncode not in [0, 1]:
        raise SystemExit("Cannot inspect staged changes. No push performed.")
    if staged.returncode == 1:
        email = f"{profile['id']}+{profile['login']}@users.noreply.github.com"
        run(["git", "-c", f"user.name={profile['login']}", "-c", f"user.email={email}",
             "commit", "-m", "Update TechArena source and documentation"], capture=False)
    run(["git", "push", "--set-upstream", "origin", branch], capture=False)
    print("Private team repository:", repo["url"])
    print("Open a pull request:", repo["url"] + "/compare/main..." + quote(branch, safe=""))
    print("Add teammates in repository Settings > Collaborators; no invitations were sent by this tool.")


if __name__ == "__main__":
    main()
