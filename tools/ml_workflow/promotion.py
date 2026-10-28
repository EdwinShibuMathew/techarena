"""Fail-closed, artifact-backed promotion decisions for future experiments.

This verifier publishes nothing. Historical schemas and trial policies are not
reinterpreted. A numerical gate summary alone never constitutes release proof.
"""
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path

import numpy as np

from .contracts import canonical, digest, first_indices, inside, require, JAN20, DAY
from .storage import read, verify_snapshot
from . import policy
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc

COHORTS = {'primary': (JAN20 + 5 * DAY, JAN20 + 7 * DAY, 43964),
           'secondary': (JAN20 + 6 * DAY, JAN20 + 7 * DAY, 35138),
           'five_day': (JAN20 + 2 * DAY, JAN20 + 7 * DAY, 51488)}
BOOTSTRAP = {'resamples': 1000, 'seed': 2026, 'minimum_valid': 990}
EXPECTED_ROWS = 529390


def load_official_arrays(metadata_path, metadata):
    """Pinned full corpus, kept separate from evidence-chain fixture tests."""
    arrays = {}
    for key in ('u', 't', 'y', 'rowids'):
        name = 'train-' + key + '.npy'; path = inside(metadata_path.parent, name)
        require(digest(path) == metadata['arrays'][name]['sha256'], 'Event data changed: ' + key)
        arrays[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    require(len(arrays['u']) == 686824 and int(arrays['y'].sum()) == 432023, 'Official training corpus differs')
    return arrays


def late_mask(times, begin):
    age = (times - begin) / 3600; late = (age >= 48) & (age < 120)
    require(int(late.sum()) == 7596, 'Official late cohort differs')
    return late


def metric_gates(comparisons):
    """Diagnostic numerical checks; deliberately returns no ready/eligible flag."""
    gates = {anchor: {name: False for name in ('primary', 'secondary', 'five_day', 'late')}
             for anchor in ('V6', 'V7')}
    blockers = []
    for anchor in gates:
        for name in gates[anchor]:
            try:
                row = comparisons[name][anchor]
                policy.validate_metric(row, confidence=name in ('primary', 'five_day'))
                gates[anchor][name] = bool(row['ci_lower'] > 0 if name in ('primary', 'five_day') else row['auc_diff'] >= 0)
                if not gates[anchor][name]: blockers.append(f'{anchor}/{name}: improvement requirement failed')
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                blockers.append(f'{anchor}/{name}: missing or invalid comparison ({error})')
    return {'gates': gates, 'blockers': blockers}


def verify_promotion(root, evidence):
    """Recompute eight checks and verify independent, accounted release evidence.

    See experiments/v7.5-audit/PROMOTION_CONTRACT.md for the descriptor. Invalid
    descriptors return blockers rather than permitting partial publication.
    """
    root = Path(root).resolve()
    result = {'eligible': False, 'gates': {a: {g: False for g in ('primary', 'secondary', 'five_day', 'late')}
                                         for a in ('V6', 'V7')}, 'blockers': [], 'comparisons': {},
              'evidence_sha256': None, 'model_role': 'unverified', 'diagnostic_only': True}
    try:
        result['evidence_sha256'] = canonical(evidence)
        require(isinstance(evidence, dict) and type(evidence.get('schema_version')) is int
                and evidence['schema_version'] in (1, 2), 'Unsupported promotion evidence schema')
        extended = evidence['schema_version'] == 2
        if extended:
            from .historical import PROFILE
            require(evidence.get('release_profile') == PROFILE, 'Unknown extended release profile')
        role = evidence.get('model_role'); result['model_role'] = role
        require(role == 'candidate', 'Fallback/diagnostic evidence cannot promote a candidate')
        require(evidence.get('purpose') == 'research', 'Only real research evidence can promote')
        require(evidence.get('baseline_lineage') == 'rebuilt_v6_v7_recipe', 'Require explicit rebuilt comparator lineage')
        require(evidence.get('bootstrap') == BOOTSTRAP, 'Bootstrap contract changed')
        inventory = evidence['artifacts']
        require(isinstance(inventory, dict) and inventory, 'Missing artifact inventory')
        def artifact(name):
            require(isinstance(name, str) and name in inventory, 'Artifact is not inventoried: ' + str(name))
            path = inside(root, name)
            require(digest(path) == inventory[name], 'Artifact changed: ' + name)
            return path
        for name in inventory: artifact(name)
        manifest = read(artifact(evidence['run_manifest']))
        require(manifest['candidate'] == evidence['candidate'] == evidence['development']['winner'], 'Frozen winner changed')
        require(evidence['development']['passed'] is True, 'Development eligibility failed or is incomplete')
        # The development selection must itself be hash-bound, not a caller flag.
        development = read(artifact(evidence['development']['report']))
        require(development['winner'] == evidence['candidate'] and development['passed'] is True,
                'Development report does not support the selected candidate')
        require(manifest['protocol']['release_policy'] == 'v6_v7_four_gate'
                and policy.dual_winner(manifest['protocol'], development['results']) == evidence['candidate'],
                'Development selection does not reproduce')
        require(manifest['baseline_lineage'] == evidence['baseline_lineage']
                and manifest['comparators'] == evidence['comparators']
                and set(manifest['comparators']) == {'V6', 'V7'}
                and all(isinstance(v, str) and len(v) == 64 for v in manifest['comparators'].values()), 'Comparator identities differ')
        for name, expected in manifest['models'].items():
            require(name in inventory and inventory[name] == expected, 'Model not bound to the frozen manifest')
            artifact(name)
        require(manifest['models'], 'Empty inference model inventory')
        source_record = read(artifact(evidence['source_record']))
        source_folder = inside(root, evidence['source_directory'])
        verify_snapshot(source_folder, source_record)
        require(source_record['sha256'] == manifest['source_sha256'], 'Scientific source changed')
        if extended:
            from .historical import verify_floors, floor_gates
            require(manifest['protocol']['release_profile'] == evidence['release_profile'], 'Manifest profile differs')
            floors = verify_floors(read(artifact(evidence['historical_floors'])), artifact, read, source_record)
            baseline = manifest['baseline_inputs']
            require(set(baseline) == {'data_sha256', 'recipe', 'models'}
                    and baseline['data_sha256'] == manifest['data_sha256'] and baseline['models']
                    and all(name.endswith(('.pt', '.cbm', '.npz')) for name in baseline['models']),
                    'Baseline identity contains candidate coefficients or invalid inputs')
            for name, value in baseline['models'].items():
                require(inventory.get(name) == value, 'Unbound comparator model'); artifact(name)
            require(manifest['comparators'] == {a: canonical({'anchor': a, 'baseline': baseline}) for a in ('V6', 'V7')},
                    'Comparator identity does not reproduce independently of the candidate')
        engineering = read(artifact(evidence['engineering_receipt']))
        require(engineering['kind'] == 'source_readiness' and engineering['git_commit'] == source_record['commit']
                and all(engineering['checks'].get(k) == 'success' for k in
                        ('cpu_tests', 'model_contracts', 'source_integrity', 'source_archive')), 'Engineering checks are incomplete')
        budget = read(artifact(evidence['accounting_record']))
        genesis_path = artifact(evidence['budget_genesis']); genesis = read(genesis_path)
        require(genesis['total_seconds'] == budget['total_seconds'] == 43200
                and budget['carry_identity'] == digest(genesis_path)
                and 0 <= budget['used_seconds'] <= 43200
                and budget['attempts'] and all(a['status'] in ('complete', 'failed', 'abandoned_reserved') for a in budget['attempts'])
                and all(np.isfinite(a['seconds']) and a['seconds'] >= 0 for a in budget['attempts'])
                and abs(sum(a['seconds'] for a in budget['attempts']) - budget['used_seconds']) < 1e-6,
                'Accounting is unfinished, reset, over budget or unreconciled')
        # Require actual first-user cohorts from the pinned dataset, including
        # labels and row IDs, rather than accepting self-described cohort sizes.
        metadata_path = artifact(evidence['data_metadata']); metadata = read(metadata_path)
        require(digest(metadata_path) == manifest['data_sha256'], 'Dataset identity differs')
        arrays = load_official_arrays(metadata_path, metadata)
        comparisons = {}
        evaluator = PairedBootstrapEvaluator()
        for name, (begin, end, count) in COHORTS.items():
            with np.load(artifact(evidence['predictions'][name]), allow_pickle=False) as saved:
                ix = first_indices(arrays['u'], arrays['t'], begin, end)
                require(len(ix) == count and np.array_equal(saved['indices'], ix)
                        and np.array_equal(saved['rowids'], arrays['rowids'][ix])
                        and np.array_equal(saved['labels'], arrays['y'][ix]), 'Official cohort/labels/order differs: ' + name)
                y = saved['labels']; candidate = saved['candidate']
                require(candidate.shape == y.shape and np.isfinite(candidate).all()
                        and ((candidate >= 0) & (candidate <= 1)).all(), 'Invalid candidate probabilities')
                comparisons[name] = {}
                with np.load(artifact(manifest['references'][name]), allow_pickle=False) as reference_saved:
                    require(np.array_equal(reference_saved['indices'], ix) and np.array_equal(reference_saved['rowids'], saved['rowids'])
                            and all(np.allclose(reference_saved[k], saved[k], rtol=0, atol=1e-15) for k in ('baseline', 'v7')),
                            'Frozen comparator vectors differ')
                for anchor, key in (('V6', 'baseline'), ('V7', 'v7')):
                    reference = saved[key]
                    require(reference.shape == y.shape and np.isfinite(reference).all()
                            and ((reference >= 0) & (reference <= 1)).all(), 'Invalid reference probabilities')
                    identity = manifest['comparators'][anchor]
                    require(str(saved[anchor + '_identity'].item()) == identity, 'Prediction comparator identity differs')
                    before, after = compute_roc_auc(y, reference), compute_roc_auc(y, candidate)
                    row = {'baseline_auc': before, 'candidate_auc': after, 'auc_diff': after - before}
                    if name in ('primary', 'five_day'):
                        row.update(asdict(evaluator.evaluate(candidate, reference, y, arrays['u'][ix])))
                    comparisons[name][anchor] = row
                    if name == 'five_day':
                        late = late_mask(arrays['t'][ix], begin)
                        a, b = compute_roc_auc(y[late], reference[late]), compute_roc_auc(y[late], candidate[late])
                        comparisons.setdefault('late', {})[anchor] = {'baseline_auc': a, 'candidate_auc': b, 'auc_diff': b - a}
        computed = metric_gates(comparisons)
        result.update(computed); result['comparisons'] = comparisons
        if extended:
            result['historical_floors'] = floor_gates(comparisons, floors)
            result['blockers'].extend('historical/' + name + ': point threshold failed'
                                      for name, passed in result['historical_floors'].items() if not passed)
        # CSV validity does not imply scientific eligibility. Both are required.
        validator_path = Path(__file__).resolve().parents[2] / 'legacy/submission_export.py'
        require(source_record['files'].get('legacy/submission_export.py') == digest(validator_path), 'CSV validator source differs')
        loader = importlib.util.spec_from_file_location('promotion_csv_validator', validator_path)
        validator = importlib.util.module_from_spec(loader); loader.loader.exec_module(validator)
        example = artifact(evidence['example_submission'])
        exports = evidence['exports']; require(len(exports) >= 2, 'Require independent inference exports')
        if extended: require(len(exports) == 3, 'Require two exports and a third standalone replay')
        csv_paths = []; expected_hash = None
        owners = set(); first_vectors = None
        for number, export in enumerate(exports, 1):
            csv_path = artifact(export['csv']); receipt = read(artifact(export['receipt']))
            freshness = ('fresh_neural_features', 'fresh_context_features', 'fresh_slate_features') if extended else ('fresh_neural_features', 'fresh_graph')
            require(receipt['candidate'] == evidence['candidate'] and all(receipt.get(k) is True for k in freshness)
                    and receipt['source_sha256'] == source_record['sha256']
                    and receipt['manifest_sha256'] == digest(artifact(evidence['run_manifest']))
                    and receipt['model_sha256'] == canonical(manifest['models'])
                    and receipt['csv_sha256'] == digest(csv_path), 'Inference receipt does not bind the selected model/source')
            owner = receipt['compute_owner']
            require(owner.startswith(manifest['owner_prefix']) and owner not in owners
                    and any(a['owner'] == owner and a['status'] == 'complete' for a in budget['attempts']),
                    'Independent inference lacks its own completed budget reservation')
            owners.add(owner)
            require(all(not os.path.samefile(csv_path, prior) for prior in csv_paths), 'Exports alias the same physical file')
            validator.validate_csv(csv_path, example, EXPECTED_ROWS)
            if extended:
                require(receipt['number'] == number and receipt['purpose'] == 'research'
                        and receipt['vectors_sha256'] == digest(artifact(export['vectors'])), 'Replay vector identity differs')
                from row_auc_final_utils import verify_csv_probabilities
                with np.load(artifact(export['vectors']), allow_pickle=False) as saved:
                    vectors = {k: saved[k] for k in ('candidate', 'members', 'rowids')}
                verify_csv_probabilities(csv_path, vectors['rowids'].ravel(), vectors['candidate'])
                require(vectors['candidate'].shape == (EXPECTED_ROWS // 10, 10)
                        and vectors['members'].shape == (5, EXPECTED_ROWS // 10, 10)
                        and np.isfinite(vectors['members']).all(), 'Replay member schema differs')
                if first_vectors is None: first_vectors = vectors
                else:
                    require(np.array_equal(vectors['rowids'], first_vectors['rowids'])
                            and all(np.allclose(vectors[k], first_vectors[k], rtol=0, atol=1e-15) for k in ('candidate', 'members')),
                            'Independent member predictions differ')
            if expected_hash is None: expected_hash = digest(csv_path)
            else: require(digest(csv_path) == expected_hash, 'Independent CSV hashes differ')
            csv_paths.append(csv_path)
        result['eligible'] = not result['blockers']
        result['diagnostic_only'] = False
    except (ValueError, KeyError, TypeError, AttributeError, OSError, IndexError, OverflowError) as error:
        result['blockers'].append(str(error))
    return result
