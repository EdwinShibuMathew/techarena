"""Freeze a new V7-relative upload trial in the authoritative research workspace."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ml_workflow.contracts import ExperimentSpec, digest, require
from ml_workflow.policy import V7_SHA, validate_anchor
from ml_workflow.storage import read, write_new
from adapter import CANDIDATES


def prepare(root=ROOT):
    ledger = read(root / 'artifacts/experiments/compute-ledger.json')
    prior = root / 'docs/results/v7.2-joint-calibration-v1/decision.json'
    decision = read(prior)
    spec = deepcopy(read(root / 'experiments/v7.2-joint-calibration-v1/experiment.json'))
    require(ledger['used_seconds'] == decision['total_compute_seconds'], 'Continue the latest authoritative decision')
    require(ledger['carry_identity'] == spec['inputs']['carry']['sha256'], 'Carry identity changed')
    require(all(a['status'] == 'complete' for a in ledger['attempts']), 'Unfinished research reservation')
    spec.update(release_name='v7_3_trial', release_policy='v7_upload_trial', upload_comparator='V7',
                anchor_sha256=V7_SHA, anchor_gamma=1.12,
                hypothesis='A small event-wide residual learned on fixed V7 improves global row ranking beyond V7 sharpening alone',
                adapter='experiments/v7.3-v7-anchor-v1/adapter.py',
                candidates=deepcopy(CANDIDATES), candidate_order=list(CANDIDATES),
                selection='maximin_nonnegative_v7_origins')
    spec['inputs']['continuation'] = {'path': prior.relative_to(root).as_posix(), 'sha256': digest(prior)}
    for role, relative in [('v7_config', 'configs/v7.json'), ('v7_validation', 'docs/results/v7-validation/validation.json')]:
        spec['inputs'][role] = {'path': relative, 'sha256': digest(root / relative)}
    ExperimentSpec.parse(spec)
    validate_anchor(root, spec)
    destination = root / 'experiments/v7.3-v7-anchor-v1/experiment.json'
    write_new(destination, spec)
    print(destination)


if __name__ == '__main__':
    prepare()
