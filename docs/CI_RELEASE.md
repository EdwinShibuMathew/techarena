# CI, branch protection and source releases

The repository distributes source and privately transferred model assets. A
green source-release check means the tested code can be packaged for sharing.
It does not establish numerical model reproduction or promote a new model.

## Merge policy

Make changes on a feature branch and open a PR into `main`. All conversations
must be resolved and the branch must be up to date with `main`. These required
GitHub Actions checks apply to administrators too:

| Check | What it verifies |
|---|---|
| `cpu-tests` | `pytest tests/` on Ubuntu 24.04 and Python 3.12 with the CPU-only dependencies in `requirements-ci.txt` |
| `source-integrity` | Frozen source hashes and unchanged V6 baseline paths; no tracked private assets, binary content, symlinks or files larger than 2 MiB |
| `release-readiness` | Both prerequisite jobs succeeded and the exact tested commit produces a complete, byte-verified source archive |

The initial solo-friendly policy requires a PR but zero external approvals.
Force pushes and deletion of `main` are disabled. Required checks are bound to
the GitHub Actions app, not arbitrary commit-status producers. When another
reviewer joins, change `required_approving_review_count` to 1 in the protection
configuration through a PR, then apply and read back the live settings.

Every PR runs the workflow, including documentation-only changes. Checks use
read-only repository permissions, do not use repository secrets, and do not
download competition data or model weights. PRs are tested at GitHub's merge
commit. The subsequent push to `main` tests and packages the actual merged SHA.

`legacy/`, `configs/source_manifest.json`, and `configs/v6.json` are frozen
relative to the PR base or previous `main` commit. Legacy Python files must
also exactly match the source manifest. Put new experiments under `experiments/`
and new model configurations in separate files.

## Source packages

Successful pushes to `main` upload a private workflow artifact named
`source-release-<commit>-<attempt>` containing:

- `techarena-source.tar.gz`: tracked source from that tested commit.
- `source-readiness.json`: commit, prerequisite results, archive hash, source
  manifest/config hashes and the V6 benchmark fingerprint.
- `SHA256SUMS`: hashes of the archive and receipt.

Packages and JUnit test reports are retained for 14 days. PRs validate packaging
without uploading a source package. Test reports are uploaded even when tests
fail, if pytest produced a report. Missing, failed, cancelled or skipped
prerequisites make release readiness fail rather than silently pass.

To regenerate a package, open **Actions → Source CI → Run workflow** and select
`main`. Manual runs on other branches fail the integrity check. This creates a
workflow artifact, not a GitHub Release or competition upload. Failed runs
publish no ready source package. Inspect the failed job and its test report,
fix it through a PR, and rerun the checks; do not disable a required check.

For a local source-only check on a clean committed checkout:

```bash
python tools/source_readiness.py verify --base-ref origin/main
pytest tests/
```

Before sharing model assets or submitting competition predictions, independently
run the existing private-asset verification and exact inference replay in the
working model environment:

```bash
python tools/consolidate.py verify
python tools/predict_v6.py
```

The source receipt explicitly records `model_inference_performed: false`,
`private_assets_verified: false`, and `competition_submission_created: false`.
The CI dependency pins are not a lock for the inference environment.

## Applying repository protection

The desired policy lives in `configs/github_branch_protection.json`. After the
workflow has produced all three successful checks, an authenticated repository
admin can apply it and verify the result:

```bash
gh api --method PUT repos/EdwinShibuMathew/techarena/branches/main/protection \
  --input configs/github_branch_protection.json
gh api repos/EdwinShibuMathew/techarena/branches/main/protection
```

Confirm that all three check runs came from `github-actions` (app ID 15368)
before applying the configuration. Read back `strict: true`, admin enforcement,
the three app-bound checks, zero required approvals, conversation resolution,
and disabled force pushes/deletion. The bootstrap workflow PR is merged while
`main` is unprotected, then protection is enabled after the first green main run.

Private-repository branch protection requires a supported GitHub plan. If the
API rejects the policy for plan or permission reasons, retain the workflows,
report the blocker, and leave the repository private. Do not claim protection
is enabled until the live API confirms it. GitHub documents [branch protection
availability](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
