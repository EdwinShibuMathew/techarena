---
name: techarena-maintenance
description: >-
  Resume TechArena ChaiScript repository work, assess PR or issue feedback, and
  plan gated V8 development using preserved project context. Use for TechArena
  maintenance and handoffs, not unrelated repositories or generic ML advice.
---

# TechArena maintenance

Use this skill to recover the project's engineering decisions without treating
an old chat as current repository state. It supplements repository instructions
and the user's task; it grants no additional execution or external permissions.

## Establish current truth

Locate the intended checkout with `git rev-parse --show-toplevel`, then inspect
`git status --short`, its revision and `AGENTS.md`. Preserve unrelated work;
use an isolated checkout when needed rather than resetting the user's tree.
Use `git ls-files` and bounded source reads instead of walking asset directories.

- For project orientation, ML constraints or integrity review, read
  [project context](references/project-context.md).
- When continuing this conversation's PRs or research decisions, read the dated
  [session handoff](references/session-handoff.md). Verify referenced PR states,
  branch tips, review comments, CI and budget evidence before relying on them.
- Treat current code, verified manifests and current repository instructions as
  the sources for execution details. Resolve paths in references relative to
  the repository root, not a remembered machine-specific checkout.

## Match the work to the request

For PR feedback, collect the general discussion, review submissions and inline
threads: actionable findings in this project have appeared as issue comments
with no inline threads. Compare findings with the current head and reproduce
material defects using disposable fixtures where useful. Make focused fixes,
add regressions for the observed failure, and report evidence and remaining
limits. Do not infer that a historical finding still applies to changed code.

For an outdated PR, compare it with current `main` and the supported workflow.
Separate valuable scientific evidence from obsolete execution code. Repair,
extract, archive or retire according to the user's goal and current evidence;
PR #4's archival decision is a precedent, not a rule for every old branch.

For new V8 work, use the merged shared workflow and a separately declared
hypothesis. A request to review, document or create a skill does not authorize
new research training. Failed recipes remain closed; source CI or a positive
point estimate does not establish promotion.

## Reuse the task-specific guidance

Read only the guidance needed for the task:

- [Run experiments](../run-experiment/SKILL.md) for scientific trials, together
  with the current shared workflow and actual dimension-specific initialization
  contracts. Do not generalize an epoch-260 example to dim16 epoch-196 assets.
- [Reproduce V6](../reproduce-v6/SKILL.md) when exact baseline inference is needed.
- [Verify the pipeline](../verify-pipeline/SKILL.md) for source and regression
  checks; use the current CPU model-contract checker for required model behavior.
- [Token-efficient inspection](../token-saver/SKILL.md) when navigating the
  codebase; prefer tracked-source inventories and explicit asset manifests.

Distinguish source packaging, synthetic model validation, exact private inference
and scientific promotion in every result. Select checks appropriate to the
change; do not rerun expensive CUDA inference merely to refresh a historical
claim. Publish or change PR/issue state only within the user's authorized scope.
Keep unrelated edits, frozen baseline files, old receipts and prior outcomes
intact. If new evidence changes a durable decision, update its authoritative
document and the relevant reference instead of accumulating transcript history.
