# Edwin: add teammates and prepare the V6 downloads

Repository: [EdwinShibuMathew/techarena](https://github.com/EdwinShibuMathew/techarena).

This guide is for Edwin. Teammates should use [Reproduce V6 in WSL](REPRODUCE_V6_WSL.md).
The guides describe the transfer process; the environment/data download archives
must still be created and shared. Adding repository access does not transfer them.

## 1. Invite each teammate

1. Ask for their GitHub username.
2. Open the repository, then **Settings → Collaborators** under **Access**.
3. Select **Add people**, find the correct account, and send the invitation.
4. Ask them to accept it and confirm they can open the private repository.

These are the steps in [GitHub's invitation guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository).
Personal-repository collaborators have write access; ask the team to use branches
and pull requests for changes. No invitations are sent by these Markdown files.

## 2. What teammates need

| Download | Purpose | How they receive it |
|---|---|---|
| Private Git repository | Code, instructions, model recipe and compact results | Git clone after accepting the invitation |
| `techarena-v6-assets.tar.gz` | Active `data/` and `artifacts/`, including all 29 verified assets and their manifest | Private team file transfer |
| `techarena-v6-env-linux-x86_64.tar.gz` | A relocatable copy of your working Python environment | Private team file transfer |
| `SHA256SUMS`, `CODE_COMMIT.txt`, `runtime_versions.json` | Download integrity and matching code/environment records | Same private transfer |

Share the asset/environment files through a team-restricted folder or another
private transfer method. Keep competition data and model weights out of Git.
The environment is Linux x86_64: it is intended for compatible Ubuntu WSL2
machines, not native Windows, macOS or ARM.

## 3. Prepare the files in your existing WSL repository

Use Ubuntu/WSL, where your working `techarena` environment is installed. Start
with the active repository; do not package `local/archive/`. Pause work that
might change the active assets or install packages while packaging.

```bash
conda activate techarena
cd ~/projects/techarena-team
git pull --ff-only
python tools/consolidate.py verify
```

The verification must succeed before continuing. Check that the code used for
the export is committed with `git status --short`; commit intended source changes
before sharing so teammates can check out the same revision.

Create a separate helper environment once. This installs the packaging tool
without changing the working model environment. If it already exists, reuse it.

```bash
conda create -n techarena-transfer -c conda-forge python=3.12 conda-pack -y
```

Create the transfer directory, pack the working environment, then pack the active
assets. Keep the output filenames unused; the environment packer refuses to
overwrite an existing archive. If repeating an export, use a fresh export folder
and keep its files together.

```bash
mkdir -p local/team-export &&
test ! -e local/team-export/techarena-v6-assets.tar.gz &&
test ! -e local/team-export/techarena-v6-env-linux-x86_64.tar.gz &&
git rev-parse HEAD > local/team-export/CODE_COMMIT.txt &&
cp docs/runtime_versions.json local/team-export/runtime_versions.json &&
conda run -n techarena-transfer conda-pack -n techarena \
  -o "$PWD/local/team-export/techarena-v6-env-linux-x86_64.tar.gz" &&
tar -czf local/team-export/techarena-v6-assets.tar.gz data artifacts &&
python tools/consolidate.py verify &&
(
  cd local/team-export &&
  sha256sum techarena-v6-assets.tar.gz \
    techarena-v6-env-linux-x86_64.tar.gz \
    CODE_COMMIT.txt runtime_versions.json > SHA256SUMS &&
  sha256sum -c SHA256SUMS &&
  du -h techarena-v6-assets.tar.gz techarena-v6-env-linux-x86_64.tar.gz
)
```

`conda-pack` prepares the environment for relocation, rather than simply copying
its directory. See [its official usage and caveats](https://conda.github.io/conda-pack/).
If it reports missing or modified package files, stop and share the error;
do not bypass it with ignore flags or reinstall packages in `techarena` blindly.

Send all five files in `local/team-export/` to each teammate privately. Keep the
archives together with their own checksum and commit records. Downloads and
extracted environments consume additional space; this export does not remove
or change V5/V6 or the archived studies. Nothing here uploads the transfer files.

## 4. Ask each teammate to report these checks

- They accepted the invitation and cloned the repository.
- All four entries in `SHA256SUMS` report `OK`.
- `python tools/consolidate.py verify` verifies all 29 collected assets.
- CUDA is available in the transferred environment.
- `python tools/predict_v6.py` prints that the portable replay passed.
- Their generated CSV has the SHA256 shown in the teammate guide.

Access and asset verification alone do not establish exact reproduction. The
packed environment preserves installed packages, but CPU/GPU hardware, drivers
and system libraries can still affect numerical results. Keep the strict replay
checks. Edwin's RTX 4070 Laptop GPU run is the reference that has already passed.

## 5. Normal collaboration

Teammates should make a branch, commit code/docs, push it, and open a pull request.
Do not commit datasets, checkpoints, submissions, environment archives or tokens.
`publish_private.py` is an Edwin-only helper and checks his GitHub identity;
teammates should use ordinary Git commands instead. Existing `legacy/` files and
V6 configuration fingerprints are part of the reproduction contract, so put
new experiments in new files instead of changing the frozen baseline.
