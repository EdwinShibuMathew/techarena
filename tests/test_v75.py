"""V7.5 chronological replicas, feature schema, leases and delivery contracts."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments/v7.5-history-age-v1'), str(ROOT / 'tools'),
                str(ROOT / 'legacy'), str(ROOT / 'experiments/v7.4-rebuild-v1')]
import protocol as v75
import campaign
import pipeline
from ml_workflow.storage import read, write_new, save_arrays
from ml_workflow.contracts import digest
from ml_workflow.contracts import canonical, first_indices


def test_fitting_prefixes_are_causal_and_cover_staleness():
    for hours in (24, 48, 96, 120):
        meta = v75.JAN20 + hours * 3600
        prefixes = v75.fitting_prefixes(meta, True)
        assert prefixes and all(0 <= meta - c <= 96 * 3600 for c in prefixes)
        assert v75.FULL not in prefixes and max(prefixes) == meta
        assert v75.fitting_prefixes(meta, False) == [meta]
    assert len(v75.ADDED_NAMES) == len(set(v75.ADDED_NAMES)) == 10
    assert list(v75.CANDIDATES) == ['A', 'B', 'C']


def test_v75_cache_rejects_changed_identity_and_bytes(tmp_path):
    path = tmp_path / 'features.npz'
    pipeline.cache(tmp_path, path, {'source': 'a'}, lambda: {'x': np.zeros(2)})
    with pytest.raises(ValueError, match='Cache'):
        pipeline.cache(tmp_path, path, {'source': 'b'}, lambda: {})
    with path.open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='Cache'):
        pipeline.cache(tmp_path, path, {'source': 'a'}, lambda: {})


def test_internal_worker_requires_controller(tmp_path):
    with patch.object(sys, 'argv', ['campaign.py', 'train', '--root', str(tmp_path)]):
        with pytest.raises(SystemExit) as failure: campaign.main()
    assert failure.value.code == 1
    assert not (tmp_path / 'artifacts/rebuild/compute-ledger.json').exists()


def test_new_model_loader_enforces_feature_width_and_receipt(tmp_path):
    pytest.importorskip('catboost')
    n = 100; rng = np.random.default_rng(2026)
    X = rng.normal(size=(n, 47)).astype(np.float32); labels = (X[:, 0] > 0).astype(np.int8)
    names = ['f' + str(i) for i in range(47)]; path = tmp_path / 'model.cbm'
    pipeline.fit_model(tmp_path, path, X, names, labels, {'test': 'schema'})
    p = pipeline.predict_model(path, X, names, 47)
    assert p.shape == (10, 10) and np.isfinite(p).all()
    with pytest.raises(ValueError, match='schema'): pipeline.predict_model(path, X, names, 37)


def test_baseline_publication_requires_two_accounted_replays(tmp_path):
    work = tmp_path / 'work'; work.mkdir()
    with patch.object(campaign, 'completed', return_value=False):
        with pytest.raises(ValueError, match='Two accounted'): pipeline.publish_baseline(tmp_path, work)
    assert not (tmp_path / 'artifacts/submissions/rebuilt_v6_submission.csv').exists()


def sample(n=168, begin=None):
    begin = v75.JAN20 if begin is None else begin
    times = begin + np.arange(n) * 3600
    history = np.full((n, 20), -1, dtype=np.int64); history[:, :3] = [0, 1, 0]
    ht = np.zeros_like(history); ht[:, :3] = times[:, None] - [90, 60, 30]
    result = dict(u=np.arange(n), t=times, i=np.tile(np.arange(10), (n, 1)), h=history, ht=ht,
                  tb=times // 21600 % 4, dt=((times // 86400 + 3) % 7 >= 5).astype(np.int8),
                  rowids=np.array([[f'{begin}-{e}-{j}' for j in range(10)] for e in range(n)]))
    if begin == v75.JAN20:
        result['y'] = np.zeros((n, 10), dtype=np.int8); result['y'][:, 9] = 1; result['y'][::4] = 0
    return result


def test_graph_features_match_explicit_edges_and_empty_history():
    pytest.importorskip('scipy')
    module = pipeline.graph_module(); raw = sample(24); cutoff = v75.JAN20 + v75.DAY
    values = module.build_graph(raw, cutoff)
    target = sample(2, cutoff); target['h'][0] = -1; target['ht'][0] = 0
    X = pipeline.extra_features(values, target, 24, 24).reshape(2, 10, 10)
    # Candidate support is a global prefix statistic; history-dependent columns are neutral.
    np.testing.assert_array_equal(X[0][:, [0, 1, 2, 3, 5, 6]], 0)
    edges = dict(zip(values['keys'].tolist(), values['weights'].tolist()))
    for half, column in [(24, 0), (72, 1)]:
        h = target['h'][1, :3]; ages = (target['t'][1] - target['ht'][1, :3]) / 3600
        weights = np.exp2(-ages / half)
        for item in range(10):
            expected = sum(edges.get(item * 50000 + int(hh), 0) * w for hh, w in zip(h, weights)) / sum(weights)
            np.testing.assert_allclose(X[1, item, column], expected, atol=1e-7)
    np.testing.assert_allclose(X[1, :, 2].sum(), 0, atol=1e-7)
    assert X[1, 9, 5] == 1 and X[1, 9, 6] == 1
    np.testing.assert_allclose(X[:, :, 8], np.log1p(24), atol=1e-6)


def test_replica_weight_preserves_each_original_row():
    names = np.array(['f'] * 47); ix = np.arange(3)
    matrices = [dict(X=np.ones((30, 47), dtype=np.float32) * i, feature_names=names, indices=ix) for i in range(3)]
    labels = np.zeros((3, 10), dtype=np.int8); labels[:, 2] = 1
    X, y, weights = pipeline.replica_training(matrices, labels)
    np.testing.assert_array_equal(y, np.tile(labels.ravel(), 3))
    np.testing.assert_allclose(weights.reshape(3, 30).sum(0), 1)
    np.testing.assert_array_equal(X[:30], 0)
    matrices[-1]['indices'] = ix[::-1]
    with pytest.raises(ValueError, match='Replica alignment'): pipeline.replica_training(matrices, labels)


def test_full_prefix_checkpoint_cannot_score_validation_dates(tmp_path):
    pytest.importorskip('torch')
    work = tmp_path / 'work'; folder = work / 'models' / str(v75.FULL) / 'dim8_seed42'; folder.mkdir(parents=True)
    write_new(work / 'manifest.json', {})
    raw = sample()
    with patch.object(campaign, 'completed', return_value=True):
        with pytest.raises(ValueError, match='precedes checkpoint'):
            pipeline.features37(tmp_path, work, raw, v75.FULL, v75.JAN20 + 5 * v75.DAY,
                                v75.JAN20 + 6 * v75.DAY, 'dim8_seed42')


def test_research_reserve_refuses_before_claiming(tmp_path):
    work = tmp_path / 'work'; work.mkdir()
    write_new(tmp_path / 'artifacts/rebuild/compute-ledger.json', {'used_seconds': 40000})
    with patch.object(campaign, 'initialize', return_value=(work, {})):
        with pytest.raises(ValueError, match='reserve'): campaign.launch(tmp_path, 'score', 'test', research=True)
    assert not (work / 'attempts').exists()


def test_each_dual_gate_failure_is_terminal(tmp_path):
    def metric():
        return dict(baseline_auc=.8, candidate_auc=.801, auc_diff=.001,
                    ci_lower=.0001, ci_upper=.002, resamples=1000,
                    late=dict(events=2, baseline_auc=.8, candidate_auc=.801, auc_diff=.001))
    develop = dict(winner='A', passed=True, results={})
    for c in v75.CANDIDATES:
        for case in ('jan22_long', 'jan24_forward'):
            develop['results'][c + '/' + case] = {**metric(), 'status': 'valid', 'v7_comparison': metric()}
    screen = dict(winner='A', passed=True, results={k: {**metric(), 'status': 'valid', 'v7_comparison': metric()} for k in ('primary', 'secondary')})
    stress = dict(winner='A', passed=True, results={'five_day': {**metric(), 'status': 'valid', 'v7_comparison': metric()}})
    import copy
    for comparison in ('baseline', 'v7'):
        for stage, case, key in [('screen', 'primary', 'ci_lower'), ('screen', 'secondary', 'auc_diff'),
                                 ('stress', 'five_day', 'ci_lower'), ('stress', 'five_day', 'late')]:
            reports = copy.deepcopy(dict(develop=develop, screen=screen, stress=stress))
            row = reports[stage]['results'][case]
            if comparison == 'v7': row = row['v7_comparison']
            if key == 'late': row['late']['auc_diff'] = -.001
            else: row[key] = -.001
            with patch.object(pipeline, 'checked', side_effect=lambda work, name: reports[name]):
                with pytest.raises(ValueError, match='eight'): pipeline.verify_gates(tmp_path)
    develop['results']['A/jan22_long']['status'] = 'invalid'
    assert pipeline.select_winner(develop['results']) == 'B'


def test_failed_accounting_prevents_task_success(tmp_path):
    from ml_workflow import budget
    work = tmp_path / 'work'; work.mkdir(); source_folder = work / 'snapshot'; source_folder.mkdir()
    parent = tmp_path / 'artifacts/rebuild'; write_new(parent / 'genesis.json', {'generation': 'test'})
    write_new(parent / 'compute-ledger.json', {'used_seconds': 0., 'total_seconds': 43200,
                                             'carry_identity': digest(parent / 'genesis.json'), 'attempts': []})
    write_new(work / 'manifest.json', {'test': True}); source = {'sha256': 'a' * 64, 'files': {}}
    real_write = budget.write_json; calls = [0]
    def fail_final(path, value):
        calls[0] += 1
        if calls[0] == 2: raise OSError('final accounting failed')
        real_write(path, value)
    def fake_worker(self, command, cwd):
        claim_path = next((work / 'attempts').glob('*.claim.json')); claim = read(claim_path)
        write_new(work / 'attempts' / (claim['token'] + '.validated.json'),
                  {'manifest_sha256': digest(work / 'manifest.json'), 'source_sha256': source['sha256'], 'files': {}})
        return 0
    with patch.object(campaign, 'initialize', return_value=(work, {})), \
         patch.object(campaign.rebuild, 'freeze', return_value=(source_folder, source)), \
         patch.object(budget.Budget, 'run', fake_worker), patch.object(budget, 'write_json', fail_final):
        with pytest.raises(OSError, match='accounting failed'): campaign.launch(tmp_path, 'train', 'train-test', 20)
    assert not (work / 'tasks/train-test.json').exists()
    assert read(parent / 'compute-ledger.json')['used_seconds'] == 20


def test_synthetic_end_to_end_graph_combiner_gates_and_csv(tmp_path):
    """Real graphs, CatBoost, bootstraps and CSVs; neural features/row counts are fixtures."""
    pytest.importorskip('catboost'); pytest.importorskip('scipy')
    import csv
    import submission_export
    root = tmp_path; work = root / 'work'; work.mkdir()
    parent = root / 'artifacts/rebuild'; write_new(parent / 'genesis.json', {'test': 'synthetic'})
    write_new(parent / 'compute-ledger.json', {'total_seconds': 43200, 'used_seconds': 0.,
                 'carry_identity': digest(parent / 'genesis.json'), 'attempts': []})
    write_new(parent / 'data/metadata.json', {'synthetic': True})
    write_new(work / 'manifest.json', {'synthetic_fixture': True})
    write_new(work / 'scientific-source.json', {'fixture': True})
    write_new(work / 'baseline-delivery.json', {'pending': 'fixture baseline export follows numerical stages'})
    raw = sample(); evaluation = sample(24, v75.FULL); evaluation_data = evaluation
    for key, value in evaluation.items(): np.save(parent / 'data' / ('evaluation-' + key + '.npy'), value)
    example = root / 'data/raw/example_submission.csv'; example.parent.mkdir(parents=True)
    with example.open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['rowid', 'predicted_probability'])
        writer.writerows((key, .1) for key in evaluation['rowids'].ravel())
    def task_marker(task, outputs):
        token = task + '-fixture'
        path = work / 'attempts' / (token + '.validated.json'); write_new(path, {'files': outputs})
        write_new(work / 'tasks' / (task + '.json'), {'token': token, 'manifest_sha256': digest(work / 'manifest.json'), 'validation_sha256': digest(path)})
    for c in v75.EXTRA_CUTOFFS:
        for member in v75.MEMBERS: task_marker(f'train-{c}-{member}', {})
    for member in v75.MEMBERS: task_marker(f'train-{v75.FULL}-{member}', {})
    def neural_fixture(root, work, raw, cutoff, begin, end, member, evaluation=False, fresh=False):
        target = evaluation_data if evaluation else raw
        ix = np.arange(24) if evaluation else first_indices(target['u'], target['t'], begin, end)
        X = np.zeros((len(ix) * 10, 37), dtype=np.float32)
        X[:, 0] = np.repeat(np.sin(target['t'][ix] / 3600), 10)
        values = dict(X=X, feature_names=np.array(['f' + str(i) for i in range(37)]), indices=ix,
                      probs=np.full((len(ix), 10), .06), raw_probs=np.full((len(ix), 10), .06))
        if fresh: return values
        return pipeline.cache(work, pipeline.feature_path(work, cutoff, begin, end, member),
                              {'fixture_cutoff': cutoff, 'begin': begin, 'end': end}, lambda: values)
    def launch(root, action, task, limit=1200, arguments=None, research=False):
        if campaign.completed(work, task): return
        args = arguments or {}
        if action == 'baseline-combiner': outputs = pipeline.baseline_combiner(work, raw, args['member'], args['meta'])
        else: outputs = pipeline.research_worker(root, work, raw, {'action': action, 'arguments': args})
        task_marker(task, outputs)
    original_export, original_validate = submission_export.export_submission, submission_export.validate_csv
    def small_export(ids, p, example, output, expected): return original_export(ids, p, example, output, len(ids))
    def small_validate(path, example, expected): return original_validate(path, example, len(evaluation['u']) * 10)
    with patch.object(pipeline, 'get37', neural_fixture), patch.object(pipeline, 'features37', neural_fixture), \
         patch.object(pipeline, 'check_cohort_size', return_value=None), \
         patch.object(submission_export, 'export_submission', small_export), patch.object(submission_export, 'validate_csv', small_validate):
        for stage in ('develop', 'screen', 'stress'): pipeline.controller(root, work, stage, launch)
        assert all(all(group.values()) for group in pipeline.verify_gates(work)[1].values())
        (work / 'baseline-delivery.json').unlink()
        for number in (1, 2): task_marker(f'baseline-export-{number}', pipeline.baseline_export(root, work, raw, number))
        pipeline.publish_baseline(root, work)
        pipeline.controller(root, work, 'finalize', launch)
        result = pipeline.verify(root, work)
    assert result['v7_5_promoted'] is True
    assert digest(root / result['submission']) == digest(work / 'exports/3/v7_5.csv')
    with (work / 'exports/3/graph.npz').open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='Task output changed'): campaign.completed(work, 'export-3')
