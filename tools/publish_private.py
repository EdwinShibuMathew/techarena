"""Publish this code-only repository privately using the existing WSL gh login."""

import json
import subprocess
from pathlib import Path

from consolidate import ROOT, digest, read_json

OWNER, NAME = "EdwinShibuMathew", "techarena"


def run(args, capture=True):
    result = subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def main():
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
    if not (ROOT / ".git").exists():
        run(["git", "init", "-b", "main"])
    top = Path(run(["git", "rev-parse", "--show-toplevel"])).resolve()
    if top != ROOT.resolve():
        raise SystemExit("Not an independent repository root; refusing to stage parent files.")
    run(["git", "add", "README.md", ".gitignore", ".gitattributes", "requirements.txt", "docs", "configs", "tools", "tests", "legacy"])
    tracked = run(["git", "ls-files", "-z"]).split("\0")
    forbidden = {".csv", ".zip", ".npz", ".npy", ".pt", ".pth", ".cbm", ".log"}
    for name in filter(None, tracked):
        path = ROOT / name
        if name.split("/")[0] in {"data", "artifacts", "local", "archive"} or path.suffix.lower() in forbidden or path.stat().st_size > 2 * 1024**2:
            raise SystemExit(f"Local asset or oversized file is staged: {name}. No push performed.")
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if staged.returncode not in [0, 1]:
        raise SystemExit("Cannot inspect staged changes. No push performed.")
    if staged.returncode == 1:
        email = f"{profile['id']}+{profile['login']}@users.noreply.github.com"
        run(["git", "-c", f"user.name={profile['login']}", "-c", f"user.email={email}",
             "commit", "-m", "Consolidate TechArena source, V6 evidence and team workflow"], capture=False)
    full_name = OWNER + "/" + NAME
    view = subprocess.run(["gh", "repo", "view", full_name, "--json", "nameWithOwner,isPrivate,url"], cwd=ROOT, text=True, capture_output=True)
    if view.returncode:
        # If name exists but cannot be accessed, creation fails; never fall back to public.
        run(["gh", "repo", "create", full_name, "--private", "--description", "ChaiScript TechArena competition code and reproducible V6 workflow"], capture=False)
    repo = json.loads(run(["gh", "repo", "view", full_name, "--json", "nameWithOwner,isPrivate,url"]))
    if not repo["isPrivate"] or repo["nameWithOwner"].lower() != full_name.lower():
        raise SystemExit("Repository privacy/identity check failed. No push performed.")
    remotes = run(["git", "remote"]).splitlines()
    url = repo["url"] + ".git"
    if "origin" in remotes:
        old = run(["git", "remote", "get-url", "origin"])
        if old not in [url, repo["url"], f"git@github.com:{full_name}.git"]:
            raise SystemExit(f"Origin points elsewhere ({old}). No remote was changed.")
    else:
        run(["git", "remote", "add", "origin", url])
    run(["gh", "auth", "setup-git"])
    run(["git", "push", "--set-upstream", "origin", "main"], capture=False)
    print("Private team repository:", repo["url"])
    print("Add teammates in repository Settings > Collaborators; no invitations were sent by this tool.")


if __name__ == "__main__":
    main()
