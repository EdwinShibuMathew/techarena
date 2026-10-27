"""Public synthetic history-graph workflow; no private inference or promotion."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ml_workflow.contracts import DAY, JAN20, MEMBERS, compose, digest, first_indices, identifier
from ml_workflow.policy import V7_SHA
from ml_workflow.storage import read, save_arrays, write_new


def create(root, fixture_id='history-smoke', informative=True):
    identifier(fixture_id)
    folder = root / 'artifacts/validation-inputs' / fixture_id
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2026)
    def events(start, hours, labeled):
        t = start + np.arange(hours) * 3600
        i = np.stack([rng.permutation(10) for _ in t])
        h = np.full((len(t), 20), -1, dtype=np.int64)
        ht = np.zeros_like(h)
        if informative:
            h[:, :3] = [0, 1, 2]
            ht[:, :3] = t[:, None] - [1200, 600, 60]
        data = {'u': np.arange(len(t)) + (0 if labeled else 1000), 'i': i,
                't': t, 'tb': ((t - JAN20) // (6 * 3600)) % 4,
                'dt': (((t - JAN20) // DAY) >= 5).astype(int), 'h': h, 'ht': ht,
                'rowids': np.array([[('train' if labeled else 'eval') + '-' + str(n) + '-' + str(j)
                                    for j in range(10)] for n in range(len(t))])}
        if labeled: data['y'] = ((i == 9) & (data['tb'][:, None] > 0)).astype(np.int8)
        return data
    train = events(JAN20, 168, True)
    evaluation = events(JAN20 + 7 * DAY, 48, False)
    spec = deepcopy(read(root / 'experiments/v7.4-history-graph-v1/experiment.template.json'))
    spec.update(purpose='synthetic_validation', device='cpu', budget={'phase_seconds': 120, 'total_seconds': 600})
    inputs = {}
    def declare(name, path): inputs[name] = {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}
    def arrays(name, **values):
        path = folder / (name + '.npz'); save_arrays(path, **values); declare(name, path)
    arrays('train', **train); arrays('evaluation', **evaluation)
    path = folder / 'inventory.json'; write_new(path, {}); declare('inventory', path)
    path = folder / 'anchor-validation.json'; write_new(path, {'synthetic': True}); declare('v7_validation', path)
    path = folder / 'v7.json'
    write_new(path, {'submission_sha256': V7_SHA, 'sharpening_gamma': 1.12,
                     'validation_report_sha256': inputs['v7_validation']['sha256']}); declare('v7_config', path)
    path = folder / 'example.csv'
    path.write_text('rowid,predicted_probability\n' + ''.join(row + ',0\n' for row in evaluation['rowids'].ravel()))
    declare('example', path)
    def baseline(name, rowids):
        members = {member: np.full(rowids.shape, .05) for member in MEMBERS}
        arrays(name, version=np.asarray('synthetic'), member_names=np.asarray(list(MEMBERS)),
               rowids=rowids, members=np.stack(list(members.values())), scores=compose(members))
    for case in spec['development'] + list(spec['promotion'].values()):
        target = first_indices(train['u'], train['t'], case['target_cutoff'], case['target_end'])
        fitting = first_indices(train['u'], train['t'], case['meta_cutoff'], case['meta_cutoff'] + DAY)
        baseline(case['baseline'], train['rowids'][target])
        baseline('baseline_fit_' + case['name'], train['rowids'][fitting])
    baseline('baseline_evaluation', evaluation['rowids'])
    spec['inputs'] = inputs
    path = folder / 'experiment.json'; write_new(path, spec)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-id', default='history-smoke')
    args = parser.parse_args(); print(create(ROOT, args.fixture_id))
