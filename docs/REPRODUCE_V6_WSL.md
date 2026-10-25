# Teammate guide: reproduce V6 in WSL

Your goal is to run the five saved V6 models and recreate the submitted CSV.
You do not need to retrain them or download the old experiment archive.

There are three parts: get repository access, install the supplied environment,
and run the saved models against the supplied data. A Git clone alone is not
enough because large assets are deliberately excluded from the repository.

## 1. Get these from Edwin first

Accept the GitHub invitation and confirm you can open
[EdwinShibuMathew/techarena](https://github.com/EdwinShibuMathew/techarena).
Ask Edwin for these six files from one matching export:

```text
techarena-v6-assets.tar.gz
techarena-v6-env-linux-x86_64.tar.gz
SHA256SUMS
CODE_COMMIT.txt
runtime_versions.json
```

The assets contain the competition archives, example CSV, processed caches,
model checkpoints, combiners, reference predictions, V5/V6 submission backups,
and the asset manifest. Download them only from the private team transfer Edwin
supplies. No public asset-download URL is configured in this repository.

The packed environment includes Python and the model libraries; you do not need
to install Miniconda, PyTorch or the unpinned `requirements.txt` separately for
this route. It does not include a Windows NVIDIA display driver.

## 2. Check your computer

The reference setup is Windows with Ubuntu WSL2 on x86_64, Python 3.12.15,
approximately 15 GiB available WSL memory, and an NVIDIA RTX 4070 Laptop GPU
with 8 GiB VRAM. Those are measured reference specifications, not tested minimums.
Allow space for both compressed downloads and extracted assets/environment;
25 GiB free is a practical initial budget, not a measured requirement.

The supplied environment was built on Ubuntu 26.04.1 LTS. Prefer a matching
Ubuntu release for the closest setup; other Linux releases/system libraries and
GPU models require verification rather than an assumption of compatibility.
ARM computers and non-NVIDIA GPUs are outside this verified CUDA setup.

If WSL is already working, keep it and inspect it first. In **Windows PowerShell**:

```powershell
wsl --list --verbose
```

Your Ubuntu distribution should show version `2`. If WSL is not installed, open
PowerShell as Administrator, run the following, restart if requested, and launch
Ubuntu to create your Linux username/password:

```powershell
wsl --install
wsl --update
```

See [Microsoft's installation guide](https://learn.microsoft.com/en-us/windows/wsl/install)
if your installation differs. All remaining shell commands are in **Ubuntu/WSL**,
not PowerShell.

Install/update a compatible NVIDIA **Windows** driver using
[NVIDIA's WSL instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Do not install a Linux display driver inside WSL. The packed PyTorch environment
supplies its CUDA runtime; its driver compatibility still matters.

In Ubuntu, check:

```bash
uname -m
cat /etc/os-release
free -h
nvidia-smi
```

Expect `x86_64`, your Ubuntu details, and an NVIDIA GPU visible to WSL. If the
GPU check fails, fix the Windows-driver/WSL setup before attempting CUDA replay.
The CUDA version displayed by `nvidia-smi` describes driver support, not the
Python environment's installed runtime.

## 3. Install tools and clone the private repository

In Ubuntu:

```bash
sudo apt update &&
sudo apt install -y git gh build-essential ca-certificates
```

Sign into **your own GitHub account inside WSL**, using the browser flow:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

Then clone into the Linux home filesystem. This avoids placing the working
project under `/mnt/c/`. The destination should not already contain a project:

```bash
mkdir -p ~/projects &&
git clone https://github.com/EdwinShibuMathew/techarena.git ~/projects/techarena-team
```

If the repository appears missing, confirm that you accepted the invitation
using the same account shown by `gh auth status`. See the
[GitHub CLI sign-in reference](https://cli.github.com/manual/gh_auth_login).

## 4. Copy and verify the six downloads

Put them together in `~/techarena-downloads`. You can copy them from Windows
Downloads with Ubuntu's file browser or `cp`. For example, replace `Alice` with
your own Windows username:

```bash
mkdir -p ~/techarena-downloads
cp /mnt/c/Users/Alice/Downloads/techarena-v6-assets.tar.gz ~/techarena-downloads/
cp /mnt/c/Users/Alice/Downloads/techarena-v6-env-linux-x86_64.tar.gz ~/techarena-downloads/
cp /mnt/c/Users/Alice/Downloads/SHA256SUMS ~/techarena-downloads/
cp /mnt/c/Users/Alice/Downloads/CODE_COMMIT.txt ~/techarena-downloads/
cp /mnt/c/Users/Alice/Downloads/runtime_versions.json ~/techarena-downloads/
```

Verify the downloads before extracting them:

```bash
cd ~/techarena-downloads && sha256sum -c SHA256SUMS
```

All four listed checks must report `OK`. The checksum file is supplied by Edwin;
obtain it through the same trusted private team transfer. A mismatch means stop
and obtain the correct file, rather than editing the recorded checksum.

Use the code revision recorded with this export:

```bash
cd ~/projects/techarena-team &&
git fetch origin &&
git checkout --detach "$(cat "$HOME/techarena-downloads/CODE_COMMIT.txt")"
```

Detached mode pins the replay to that commit. For later development, return to
`main` and create a branch. Keep the pinned revision for your first reproduction.

## 5. Install the supplied Python environment

Choose its permanent directory now; do not move it after `conda-unpack`. In Ubuntu:

```bash
mkdir -p ~/envs &&
test ! -e "$HOME/envs/techarena-v6" &&
mkdir "$HOME/envs/techarena-v6" &&
tar -xzf "$HOME/techarena-downloads/techarena-v6-env-linux-x86_64.tar.gz" \
  -C "$HOME/envs/techarena-v6" &&
source "$HOME/envs/techarena-v6/bin/activate" &&
conda-unpack
```

If that directory already exists, do not extract over it. Reuse a correctly
installed environment, or choose a new unused permanent directory and adjust
the activation command. These steps use the
[official conda-pack relocation procedure](https://conda.github.io/conda-pack/).

In future Ubuntu terminals, activate with:

```bash
source "$HOME/envs/techarena-v6/bin/activate"
```

Check the interpreter, package versions and GPU:

```bash
which python
python --version
python - <<'PY'
import importlib.metadata as metadata
import torch
for package in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "torch", "triton"]:
    print(f"{package}: {metadata.version(package)}")
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

The recorded environment, fetched from the repository on 2026-10-25, is:

| Component | Recorded version |
|---|---|
| Python | 3.12.15 |
| NumPy | 2.5.4 |
| pandas | 3.1.0 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| CatBoost | 1.2.10 |
| PyTorch | 2.14.1+cu130 |
| Triton | 3.8.0 |

The export's `runtime_versions.json` and repository `docs/runtime_versions.json`
are the records to compare against. These versions describe Edwin's installation;
they are not a claim that all matching packages are currently available from
public installers. This is why the packed working environment is the recommended
route. Do not replace its packages with the latest versions before the first run.

## 6. Extract the V6 assets into the clone

Use a fresh clone with no existing `data/` or `artifacts/` directory. From its root:

```bash
cd ~/projects/techarena-team &&
test ! -e data && test ! -e artifacts &&
tar -xzf "$HOME/techarena-downloads/techarena-v6-assets.tar.gz" -C .
```

The layout must contain these paths directly under the repository:

```text
data/raw/                         # Original archives and example CSV
data/cache/                       # Four processed caches
artifacts/v6/manifest.json
artifacts/v6/dim8_seed42/
artifacts/v6/dim8_seed43/
artifacts/v6/dim8_seed44/
artifacts/v6/dim16_seed42/
artifacts/v6/dim16_seed43/
artifacts/submissions/v5.csv
artifacts/submissions/v6.csv
```

The manifest retains original locations as provenance; those old absolute
locations do not have to exist on your machine. The portable replay uses the
relative paths in this layout. Keep all 29 manifest assets, including the
preference checkpoints and V5 backup required by the verifier.

The export may include Edwin's previous `artifacts/reproduced_v6.csv`. To ensure
your run creates a fresh CSV, the command below uses a separate output name.

## 7. Verify and reproduce

With the supplied environment active and your terminal in the repository:

```bash
python tools/consolidate.py verify &&
python tools/predict_v6.py --output artifacts/teammate_reproduced_v6.csv &&
sha256sum artifacts/teammate_reproduced_v6.csv
```

Successful output includes:

```text
Verified 29 collected assets and original source fingerprints.
V6 portable replay passed; CSV is byte-identical to the submitted candidate.
```

The generated CSV must have 529,390 prediction rows, with header
`rowid,predicted_probability`, and this exact SHA256:

```text
c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c
```

The success receipt is `local/v6_relocation_check.json`. Send Edwin the final
success message, CSV hash, recorded commit and GPU name. No competition upload
is performed by these commands.

## 8. If a check fails

| Symptom | Next step |
|---|---|
| `gh` cannot see the repository | Accept the invitation; check the WSL GitHub account |
| Download checksum mismatch | Obtain the matching export file again |
| Missing manifest or asset | Check extraction location and that the whole asset archive was supplied |
| Asset/source fingerprint mismatch | Stop; check the pinned commit and matching export |
| Python imports fail | Check `which python` points into `~/envs/techarena-v6/` and `conda-unpack` completed |
| CUDA unavailable or driver too old | Check Windows NVIDIA driver, WSL2 and the environment's CUDA runtime |
| A member's predictions differ | Keep the error, package versions, Ubuntu version and GPU details; ask Edwin to investigate |
| Existing output differs | Use another new output filename inside `artifacts/`; do not overwrite baseline files |

Exact replay uses a strict per-member tolerance and exact final CSV hash. A
packed environment reduces software differences; it does not guarantee identical
floating-point results across GPU/CPU models, drivers or system libraries.
Do not loosen the checks to claim reproduction. `--device cpu` exists for
diagnostics, but CPU replay has not been established as byte-identical.

You do not need Edwin's old sibling projects or `local/archive/`. On a new clone,
do not run inventory/bundle/archive/cleanup commands intended for his migration.
Full-data training is not yet a single portable command; this guide reproduces
the saved V6 inference pipeline. Put new experiments on a branch and keep the
baseline code/configuration and model assets intact.
