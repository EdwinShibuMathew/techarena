"""Declare the new campaign in the existing, authoritative research workspace."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ml_workflow.contracts import ExperimentSpec, digest, require
from ml_workflow.storage import read, write_new
from adapter import CANDIDATES


def prepare(root=ROOT):
    ledger = read(root / 'artifacts/experiments/compute-ledger.json')
    prior = root / 'docs/results/v7.1-entropy-sharpening-v1/decision.json'
    decision = read(prior)
    spec = deepcopy(read(root / 'experiments/v7.1-entropy-sharpening-v1/experiment.json'))
    require(ledger['used_seconds'] == decision['total_compute_seconds'], 'Continue the latest authoritative decision')
    require(ledger['carry_identity'] == spec['inputs']['carry']['sha256'], 'Carry identity changed')
    require(all(a['status'] == 'complete' for a in ledger['attempts']), 'Unfinished research reservation')
    spec.update(schema_version=3, release_name='v7_2',
                hypothesis='Joint bounded event-wide confidence and conditional-score calibration transfers across model ages better than mass-preserving sharpening',
                adapter='experiments/v7.2-joint-calibration-v1/adapter.py',
                candidates=deepcopy(CANDIDATES), candidate_order=list(CANDIDATES),
                selection='maximin_nonnegative_origins_and_late',
                postprocessing_contract={'event_mass': 'learned', 'within_event_order': 'preserve', 'state_schema_version': 2})
    spec['inputs']['continuation'] = {'path': prior.relative_to(root).as_posix(), 'sha256': digest(prior)}
    ExperimentSpec.parse(spec)
    destination = root / 'experiments/v7.2-joint-calibration-v1/experiment.json'
    write_new(destination, spec)
    print(destination)


if __name__ == '__main__':
    prepare()
