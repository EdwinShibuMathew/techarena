# How I manage team access and share V6

Repository: [EdwinShibuMathew/techarena](https://github.com/EdwinShibuMathew/techarena).

This is my checklist for giving teammates access and preparing our V6 downloads.
If you're joining the team, use [Reproduce V6 in WSL](REPRODUCE_V6_WSL.md).

I have prepared the five transfer files and checked their hashes. The asset
archive is about 1015 MiB and the environment archive about 3.2 GiB. I share these
files privately; adding repository access alone does not transfer the data or models.
The commands below document how I prepared the export and how I can rebuild it.

## 1. How I invite teammates

1. I ask for each teammate's GitHub username.
2. I open the repository, then **Settings → Collaborators** under **Access**.
3. I select **Add people**, confirm the account, and send the invitation.
4. I ask them to accept it and check that they can open the private repository.

These are the steps in [GitHub's invitation guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository).
Personal-repository collaborators have write access. I ask the team to use
branches and pull requests so we can review changes before merging.
`main` requires an up-to-date PR, passing CPU tests, source integrity and release
readiness checks, and resolved conversations, including for admins. The initial
solo-friendly policy requires zero external approvals. See [the CI guide](CI_RELEASE.md).

## 2. What teammates need

| Download | Purpose | How they receive it |
|---|---|---|
| Private Git repository | Code, instructions, model recipe and compact results | Git clone after accepting the invitation |
| `techarena-v6-assets.tar.gz` | Active `data/` and `artifacts/`, including all 29 verified assets and their manifest | Private team file transfer |
| `techarena-v6-env-linux-x86_64.tar.gz` | A relocatable copy of my working Python environment | Private team file transfer |
| `SHA256SUMS`, `CODE_COMMIT.txt`, `runtime_versions.json` | Download integrity and matching code/environment records | Same private transfer |

I share the asset/environment files through a team-restricted folder or another
private transfer method. I keep competition data and model weights out of Git.
The environment is Linux x86_64: it is intended for compatible Ubuntu WSL2
machines, not native Windows, macOS or ARM.

## 3. How I prepare the files in WSL

I package from Ubuntu/WSL using my working `techarena` environment and the active
repository. I exclude `local/archive/` and pause training or package installation
that could change the assets or environment during the export.

```bash
conda activate techarena
cd ~/projects/techarena-team
git pull --ff-only
python tools/consolidate.py verify
```

The verification must succeed before continuing. Check that the code used for
the export is committed with `git status --short`; commit intended source changes
before sharing so teammates can check out the same revision.

I keep the packaging tool in a separate helper environment so I don't change
the working model environment. I create it once and reuse it for later exports.

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

I share all five files in `local/team-export/` privately, keeping the archives
together with their checksum and commit records. The export creates extra files;
it does not remove or change V5/V6 or the archived studies. These commands build
the downloads locally; I handle the private transfer separately.

## 4. Checks I ask teammates to report

- They accepted the invitation and cloned the repository.
- All four entries in `SHA256SUMS` report `OK`.
- `python tools/consolidate.py verify` verifies all 29 collected assets.
- CUDA is available in the transferred environment.
- `python tools/predict_v6.py` prints that the portable replay passed.
- Their generated CSV has the SHA256 shown in the teammate guide.

Access and asset verification alone do not establish exact reproduction. The
packed environment preserves installed packages, but CPU/GPU hardware, drivers
and system libraries can still affect numerical results. Keep the strict replay
checks. My RTX 4070 Laptop GPU run is the reference that has already passed.

## 5. Normal collaboration

I ask teammates to make a branch, commit code/docs, push it, and open a pull request.
Do not commit datasets, checkpoints, submissions, environment archives or tokens.
`publish_private.py` is my publishing helper and checks my GitHub identity;
it publishes the current feature branch and refuses `main` or detached HEAD
before staging. Teammates should use ordinary Git commands instead. Existing `legacy/` files and
V6 configuration fingerprints are part of the reproduction contract, so put
new experiments in new files instead of changing the frozen baseline.
