"""Required V7.4 model, causality, promotion and synthetic workflow contracts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from ml_workflow import engine, history_postprocessing as history, policy
from ml_workflow.contracts import DAY, JAN20, ExperimentSpec, canonical, context, digest
from ml_workflow.storage import read, write_new
from ml_workflow.budget import Budget
from predict_v7 import calibrate_v7_probabilities
import postprocessor_replay as replay


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


ADAPTER = 'experiments/v7.4-history-graph-v1/adapter.py'
adapter = load('history_graph_adapter', ROOT / ADAPTER)
fixture = load('history_graph_fixture', ROOT / 'experiments/v7.4-history-graph-v1/prepare_fixture.py')
SCIPY = importlib.util.find_spec('scipy') is not None


def sample(n=40):
    rng = np.random.default_rng(2026)
    t = JAN20 + np.arange(n) * 1800
    h = np.full((n, 20), -1, dtype=np.int64)
    ht = np.zeros_like(h)
    h[:, :3] = [0, 1, 0]
    ht[:, :3] = t[:, None] - [1000, 500, 60]
    i = np.stack([rng.permutation(10) for _ in t])
    y = (i == 9).astype(np.int8)
    y[::4] = 0
    return {'u': np.arange(n) % 12, 'i': i, 't': t, 'tb': np.zeros(n, dtype=int),
            'dt': np.zeros(n, dtype=int), 'h': h, 'ht': ht, 'y': y,
            'rowids': np.array([[f'{e}-{j}' for j in range(10)] for e in range(n)])}


@unittest.skipUnless(SCIPY, 'SciPy required; enforced without skips by model-contracts CI')
class HistoryGraphModelTests(unittest.TestCase):
    def test_graph_matches_distinct_user_oracle_including_warmup(self):
        data = sample()
        graph = adapter.build_graph(data, JAN20 + DAY)
        users = {}
        for n, user in enumerate(data['u']):
            users.setdefault(int(user), set()).update(data['h'][n][data['h'][n] >= 0].tolist())
            users[int(user)].update(data['i'][n][data['y'][n] == 1].tolist())
        expected = {}
        items = sorted(set().union(*users.values()))
        for a in items:
            for b in items:
                if a == b: continue
                c = sum(a in s and b in s for s in users.values())
                if c >= 3:
                    na = sum(a in s for s in users.values()); nb = sum(b in s for s in users.values())
                    expected[a * 50000 + b] = c / np.sqrt(na * nb) * c / (c + 20)
        self.assertEqual(graph['keys'].tolist(), sorted(expected))
        np.testing.assert_allclose(graph['weights'], [expected[k] for k in sorted(expected)], rtol=0, atol=1e-16)
        repeated = {k: np.concatenate([v, v]) for k, v in data.items()}
        repeated['rowids'] = np.char.add(repeated['rowids'], np.repeat(['a', 'b'], len(data['u']))[:, None])
        duplicate_graph = adapter.build_graph(repeated, JAN20 + DAY)
        for key in graph: np.testing.assert_array_equal(graph[key], duplicate_graph[key])
        with self.assertRaisesRegex(ValueError, 'crosses cutoff'): adapter.build_graph(data, int(data['t'][10]))

    def test_sparse_neighbor_cap_and_deterministic_ties(self):
        data = sample(240)
        data['u'] = np.repeat(np.arange(3), 80)
        data['t'] = np.full(240, JAN20)
        data['h'][:] = -1; data['ht'][:] = 0
        data['i'][:, 0] = np.tile(np.arange(80), 3)
        # Keep candidates distinct within each event, unrelated opaque keys.
        data['i'][:, 1:] = np.arange(100, 109)
        data['y'][:] = 0; data['y'][:, 0] = 1
        graph = adapter.build_graph(data, JAN20 + DAY)
        self.assertEqual(len(graph['keys']), 80 * 64)
        np.testing.assert_array_equal(graph['keys'][graph['keys'] // 50000 == 0] % 50000, np.arange(1, 65))
        order = np.random.default_rng(7).permutation(240)
        alternate = adapter.build_graph({k: v[order] for k, v in data.items()}, JAN20 + DAY)
        for key in graph: np.testing.assert_array_equal(graph[key], alternate[key])

    def test_history_features_have_no_labels_and_use_causal_recency(self):
        prefix = sample(); graph = adapter.build_graph(prefix, JAN20 + DAY)
        target = context(prefix, np.arange(8), False)
        target['t'] += 3 * DAY
        target['ht'][:, :3] = target['t'][:, None] - [72 * 3600, 24 * 3600, 60]
        weights = dict(zip(graph['keys'].tolist(), graph['weights'].tolist()))
        for half in (24, 72):
            actual = adapter.affinities(graph, target, half)
            expected = np.zeros_like(actual)
            for e in range(len(expected)):
                age = (target['t'][e] - target['ht'][e, :3]) / 3600
                recency = 2 ** (-age / half)
                for j, item in enumerate(target['i'][e]):
                    expected[e, j] = sum(weights.get(int(item) * 50000 + int(h), 0.) * w
                                         for h, w in zip(target['h'][e, :3], recency)) / recency.sum()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-16)
        bad = deepcopy(target); bad['y'] = prefix['y'][:8]
        with self.assertRaisesRegex(ValueError, 'labels forbidden'): adapter.affinities(graph, bad, 24)
        target['h'][:] = -1; target['ht'][:] = 0
        np.testing.assert_array_equal(adapter.affinities(graph, target, 24), 0)
        target['h'][:, 0] = 49999; target['ht'][:, 0] = target['t'] - 1
        np.testing.assert_array_equal(adapter.affinities(graph, target, 24), 0)

    def test_encoded_identifiers_do_not_become_predictive_magnitudes(self):
        prefix = sample(); target = context(prefix, np.arange(5), False)
        target['t'] += DAY
        mapping = np.array([49999, 71, 401, 12, 235, 11, 812, 913, 8, 9999])
        def remap(data):
            changed = deepcopy(data); changed['i'] = mapping[data['i']].astype(np.int32)
            present = data['h'] >= 0; changed['h'][present] = mapping[data['h'][present]]
            return changed
        a = adapter.affinities(adapter.build_graph(prefix, JAN20 + DAY), target, 24)
        b = adapter.affinities(adapter.build_graph(remap(prefix), JAN20 + DAY), remap(target), 24)
        np.testing.assert_array_equal(a, b)

    def test_tilt_preserves_mass_endpoints_permutations_and_identity(self):
        rng = np.random.default_rng(12)
        p = rng.uniform(.001, .999, (30, 10))
        p[:5] = [[0] * 10, [1] * 10, [1, 0] * 5, [1e-300] * 10, [.05] * 10]
        z = rng.uniform(-5, 5, p.shape)
        z[8] = 0; z[9] = 2.2
        np.testing.assert_array_equal(adapter.tilt(p, z, 0), p)
        for alpha in (.01, .25):
            q = adapter.tilt(p, z, alpha)
            np.testing.assert_array_equal(q[8:10], p[8:10])
            np.testing.assert_allclose(q.sum(1), p.sum(1), rtol=0, atol=1e-14)
            np.testing.assert_array_equal(q[(p == 0) | (p == 1)], p[(p == 0) | (p == 1)])
            self.assertTrue(np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all())
            order = np.arange(9, -1, -1)
            np.testing.assert_allclose(adapter.tilt(p[:, order], z[:, order], alpha), q[:, order], rtol=0, atol=1e-15)
        q = adapter.tilt(np.full((1, 10), .05), np.arange(10)[None, :], .25)
        self.assertGreater(q[0, -1], q[0, 0])

    def test_implicit_and_loss_gradients_match_independent_finite_differences(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(.001, .15, (25, 10)); z = rng.normal(size=p.shape)
        y = np.zeros_like(p); y[1:, 0] = 1
        positive = np.flatnonzero(y.ravel() == 1); negative = np.flatnonzero(y.ravel() == 0)
        sampled = rng.choice(negative, (len(positive), 16))
        alpha, eps = .12, 1e-6
        _, jac = adapter.tilt(p, z, alpha, derivative=True)
        numeric = (adapter.tilt(p, z, alpha + eps) - adapter.tilt(p, z, alpha - eps)) / (2 * eps)
        np.testing.assert_allclose(jac, numeric, rtol=2e-6, atol=2e-9)
        value, gradient = adapter.objective(alpha, p, z, y, positive, sampled)
        numeric = (adapter.objective(alpha + eps, p, z, y, positive, sampled)[0] -
                   adapter.objective(alpha - eps, p, z, y, positive, sampled)[0]) / (2 * eps)
        self.assertAlmostEqual(gradient, numeric, places=7)
        self.assertTrue(np.isfinite(value))

    def test_actual_fit_reload_normalization_and_failure_refusal(self):
        rng = np.random.default_rng(2)
        p = np.full((80, 10), .05)
        features = rng.uniform(0, .1, p.shape); features[:, 9] += .3
        y = np.zeros_like(p); y[1:, 9] = 1
        params = adapter.CANDIDATES['history_24h']; metadata = {'meta_cutoff': JAN20}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'calibrator.json'
            saved = adapter.fit_calibrator(p, features, y, deepcopy(metadata), path, deepcopy(params))
            self.assertGreater(saved['alpha'], 0)
            self.assertLess(saved['objective'], saved['identity_objective'])
            self.assertEqual(saved, adapter.load_calibrator(path, metadata, params))
            centered = features - features.mean(1, keepdims=True)
            self.assertEqual(saved['normalization'], {'rms': float(np.sqrt(np.mean(centered ** 2)))})
            np.testing.assert_array_equal(adapter.predict_calibrator(p, features, saved),
                                          adapter.predict_calibrator(p, features, adapter.load_calibrator(path)))
            bad = deepcopy(saved); bad['alpha'] = .26
            with self.assertRaises(ValueError): adapter.predict_calibrator(p, features, bad)
            bad = deepcopy(saved); bad['normalization']['rms'] = float('inf')
            with self.assertRaises(ValueError): adapter.predict_calibrator(p, features, bad)
            with self.assertRaises(ValueError): adapter.load_calibrator(path, {'meta_cutoff': JAN20 + DAY}, params)
            failed = Path(tmp) / 'failed.json'
            with patch('scipy.optimize.minimize', return_value=SimpleNamespace(success=False, message='deliberate')):
                with self.assertRaisesRegex(ValueError, 'did not converge'):
                    adapter.fit_calibrator(p, features, y, metadata, failed, params)
            self.assertFalse(failed.exists())

    def test_graph_cache_identity_and_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = root / 'work'; work.mkdir()
            m = {'source': {'files': {ADAPTER: digest(ROOT / ADAPTER)}}, 'spec': {'adapter': ADAPTER}}
            data = sample()
            a, files = history.graph(root, work, m, adapter, data, JAN20 + DAY)
            b, _ = history.graph(root, work, m, adapter, data, JAN20 + DAY)
            for key in a: np.testing.assert_array_equal(a[key], b[key])
            other = root / 'other'; other.mkdir()
            with patch.object(adapter, 'build_graph', side_effect=AssertionError('cache miss')):
                history.graph(root, other, m, adapter, data, JAN20 + DAY)
            changed = deepcopy(data); changed['y'][0] = 0; changed['y'][1] = 0
            with self.assertRaisesRegex(ValueError, 'identity or bytes changed'):
                history.graph(root, work, m, adapter, changed, JAN20 + DAY)
            graph_file = next(work.glob('graphs/*.npz')); graph_file.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'identity or bytes changed'):
                history.graph(root, work, m, adapter, data, JAN20 + DAY)


def metric(gain=.01, low=.005):
    return {'baseline_auc': .8, 'candidate_auc': .8 + gain, 'auc_diff': gain,
            'ci_lower': low, 'ci_upper': .02, 'resamples': 1000}


class HistoryPromotionTests(unittest.TestCase):
    def test_authoritative_accounting_is_checked_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            carry = root / 'carry.json'; prior = root / 'prior.json'
            write_new(carry, {'total_compute_seconds': 5.})
            write_new(prior, {'total_compute_seconds': 6.})
            spec = {'inputs': {name: {'path': path.name, 'sha256': digest(path)}
                               for name, path in [('carry', carry), ('continuation', prior)]},
                    'budget': {'total_seconds': 14400}}
            ledger = root / 'artifacts/experiments/compute-ledger.json'
            state = {'total_seconds': 14400, 'used_seconds': 6., 'carry_identity': digest(carry),
                     'attempts': [{'owner': 'prior/phase', 'status': 'complete', 'seconds': 1.}]}
            write_new(ledger, state)
            before = ledger.read_bytes()
            self.assertEqual(history.preflight(root, spec)['used_seconds'], 6.)
            self.assertEqual(ledger.read_bytes(), before)
            for mutate in (lambda d: d.update(used_seconds=5.), lambda d: d.update(used_seconds=7.),
                           lambda d: d.update(carry_identity='changed'),
                           lambda d: d['attempts'][0].update(status='running')):
                bad = deepcopy(state); mutate(bad); ledger.write_text(json.dumps(bad)); before = ledger.read_bytes()
                with self.assertRaises(ValueError): history.preflight(root, spec)
                self.assertEqual(ledger.read_bytes(), before)

    def test_dual_publication_receipt_cannot_drop_a_comparator_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = root / 'artifacts/experiments/run'; work.mkdir(parents=True)
            (root / 'configs').mkdir()
            spec = read(ROOT / 'experiments/v7.4-history-graph-v1/experiment.template.json')
            m = {'run_id': 'run', 'spec': spec, 'source': {'commit': 'synthetic-control', 'sha256': 'f' * 64}}
            write_new(work / 'manifest.json', m)
            def row(case):
                return {**metric(), 'case': case, 'late': {'events': 3, **metric()},
                        'v7_comparison': {**metric(), 'anchor_sha256': policy.V7_SHA, 'late': metric()}}
            def report(stage, results, **extra):
                path = work / (stage + '.json')
                write_new(path, {'complete': True, 'passed': True, 'files': {}, 'results': results,
                                 'manifest_sha256': digest(work / 'manifest.json'), **extra})
                write_new(work / (stage + '.seal.json'), {'sha256': digest(path)})
                write_new(work / (stage + '.ledger.json'), {'exit_code': 0})
            development = {candidate + '/' + case['name']: row(case)
                           for candidate in spec['candidate_order'] for case in spec['development']}
            report('develop', development, winner='history_24h')
            report('screen', {name: row(spec['promotion'][name]) for name in ('primary', 'secondary')})
            report('stress', {'five_day': row(spec['promotion']['five_day'])})
            (work / 'candidate.csv').write_bytes(b'synthetic publication control; not numerical evidence')
            write_new(work / 'finalize.json', {'complete': True, 'independent_replay': True,
                      'manifest_sha256': digest(work / 'manifest.json'), 'rows': 20,
                      'csv_sha256': digest(work / 'candidate.csv'), 'files': {'candidate.csv': digest(work / 'candidate.csv')}})
            write_new(work / 'finalize.seal.json', {'sha256': digest(work / 'finalize.json')})
            write_new(work / 'finalize.ledger.json', {'exit_code': 0})
            with patch.object(engine, 'load', return_value=m):
                engine.publish(root, work, m)
                actual = engine.verify(root, 'run')
                self.assertTrue(actual['submission_ready']); self.assertTrue(actual['promotion_supported'])
                config = root / 'configs/v7_4.json'; receipt = read(config)
                self.assertTrue(all(v for gates in receipt['comparison_gate_results'].values() for v in gates.values()))
                receipt['comparison_gate_results']['V7']['late'] = False
                config.write_text(json.dumps(receipt)); (work / 'published.json').write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'Dual comparison receipt changed'): engine.verify(root, 'run')
            screen = read(work / 'screen.json')
            screen['results']['primary']['case'] = deepcopy(spec['promotion']['secondary'])
            (work / 'screen.json').write_text(json.dumps(screen))
            (work / 'screen.seal.json').write_text(json.dumps({'sha256': digest(work / 'screen.json')}))
            with self.assertRaisesRegex(ValueError, 'chronology changed'): engine.checked(work, 'screen')

    def test_dual_accounting_failure_prevents_publication(self):
        from ml_workflow import budget as budget_module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = root / 'artifacts/experiments/run'; work.mkdir(parents=True)
            spec = read(ROOT / 'experiments/v7.4-history-graph-v1/experiment.template.json')
            m = {'spec': spec, 'spec_sha256': 'f' * 64, 'source': {'commit': 'fixture'},
                 'carried_seconds': 0., 'carry_identity': None}
            original = budget_module.write_json
            def fail(path, state):
                if state['attempts'][-1]['status'] == 'complete': raise OSError('final accounting failure')
                original(path, state)
            with patch.object(engine, 'locate', return_value=work), patch.object(engine, 'load', return_value=m), \
                 patch.object(engine, 'checked', return_value={}), patch.object(Budget, 'run', return_value=0), \
                 patch.object(budget_module, 'write_json', side_effect=fail), patch.object(engine, 'publish') as publish:
                with self.assertRaisesRegex(OSError, 'accounting failure'): engine.run(root, 'run', 'finalize')
                publish.assert_not_called()
            self.assertFalse((root / 'configs/v7_4.json').exists())
            self.assertFalse((work / 'finalize.ledger.json').exists())
            state = read(root / 'artifacts/experiments/compute-ledger.json')
            self.assertEqual(state['attempts'][-1]['status'], 'running'); self.assertEqual(state['used_seconds'], 1200)

    def test_policy_requires_all_eight_gates_and_keeps_legacy_semantics(self):
        spec = read(ROOT / 'experiments/v7.4-history-graph-v1/experiment.template.json')
        ExperimentSpec.parse(spec)
        def reports():
            row = {**metric(), 'v7_comparison': {**metric(), 'anchor_sha256': policy.V7_SHA}}
            screen = {'primary': deepcopy(row), 'secondary': deepcopy(row)}
            five = deepcopy(row)
            five['late'] = metric(); five['v7_comparison']['late'] = metric()
            return screen, {'five_day': five}
        screen, stress = reports()
        self.assertTrue(policy.stage_passed(spec, 'screen', screen))
        self.assertTrue(policy.stage_passed(spec, 'stress', stress))
        for anchor in ('V6', 'V7'):
            for gate in ('primary', 'secondary', 'five_day', 'late'):
                screen, stress = reports()
                row = screen[gate] if gate in screen else stress['five_day']
                if anchor == 'V7': row = row['v7_comparison']
                if gate == 'late': row = row['late']
                if gate in ('primary', 'five_day'): row['ci_lower'] = 0
                else: row.update(metric(gain=-.001, low=-.002))
                self.assertFalse(policy.stage_passed(spec, 'screen' if gate in screen else 'stress', screen if gate in screen else stress))
        self.assertEqual(policy.release_flags({}), {'submission_ready': True})
        self.assertFalse(policy.release_flags({'release_policy': 'v7_upload_trial'})['promotion_supported'])
        self.assertTrue(policy.receipt_ready(spec, policy.release_flags(spec)))
        bad = policy.release_flags(spec); bad['promotion_supported'] = False
        self.assertFalse(policy.receipt_ready(spec, bad))
        for key, value in [('schema_version', 3), ('release_policy', 'v7_upload_trial'),
                           ('mode', 'ensemble_postprocessor'), ('selection', 'maximin_nonnegative_v7_origins')]:
            bad = deepcopy(spec); bad[key] = value
            with self.assertRaises(ValueError): ExperimentSpec.parse(bad)

    def test_selection_checks_both_anchors_and_aged_development(self):
        spec = read(ROOT / 'experiments/v7.4-history-graph-v1/experiment.template.json')
        rows = {candidate + '/' + case['name']:
                {**metric(), 'late': {'events': 4, **metric()},
                 'v7_comparison': {**metric(), 'late': metric()}}
                for candidate in spec['candidate_order'] for case in spec['development']}
        self.assertEqual(engine._winner(spec, rows), 'history_24h')
        rows['history_24h/jan22_long']['v7_comparison']['late'] = metric(-.001)
        self.assertEqual(engine._winner(spec, rows), 'history_72h')
        rows['history_72h/jan22_long']['late'].update(metric(-.001))
        self.assertIsNone(engine._winner(spec, rows))

    def test_missing_research_assets_never_claim_or_create_a_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = read(ROOT / 'experiments/v7.4-history-graph-v1/experiment.template.json')
            path = root / 'experiments/trial/experiment.json'; write_new(path, spec)
            with patch.object(Budget, '__enter__') as budget:
                with self.assertRaisesRegex(ValueError, 'Restore verified assets'):
                    engine.prepare(root, path, 'blocked-v74')
                budget.assert_not_called()
            self.assertFalse((root / 'artifacts').exists())

    def test_replay_validation_binds_fresh_graph_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = root / 'artifacts/experiments/run'
            write_new(work / 'manifest.json', {})
            stage = root / 'artifacts/stage'; stage.mkdir()
            graph_file = work / 'graphs/final-replay-token.npz'
            graph_file.parent.mkdir(); graph_file.write_bytes(b'graph fixture')
            write_new(graph_file.with_suffix('.json'), {'identity': 'fixture'})
            record = {'context_schema': 4, 'run_id': 'run', 'release_name': 'v7_4',
                      'manifest_sha256': 'a', 'config_sha256': 'b', 'source_snapshot_sha256': 'c',
                      'source_commit': 'fixture', 'rows': 20, 'csv_sha256': 'd'}
            before = replay._validation(record, 'replay-token', stage, root, 'bridge')
            self.assertEqual(before['graph_files'][graph_file.relative_to(root).as_posix()], digest(graph_file))
            graph_file.write_bytes(b'changed')
            self.assertNotEqual(before, replay._validation(record, 'replay-token', stage, root, 'bridge'))


@unittest.skipUnless(SCIPY, 'SciPy required; enforced without skips by model-contracts CI')
class HistoryWorkflowTests(unittest.TestCase):
    def repository(self, root):
        from test_workflow_integration import WorkflowIntegrationTests
        helper = WorkflowIntegrationTests(); helper.make_repository(root)
        return helper

    def test_actual_synthetic_campaign_and_independent_final_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); helper = self.repository(root)
            path = fixture.create(root, 'history-success')
            helper.cli(root, 'prepare', '--run-id', 'history-success', '--spec', str(path))
            for stage in ('develop', 'screen', 'stress', 'finalize'):
                helper.cli(root, stage, '--run-id', 'history-success')
            result = json.loads(helper.cli(root, 'verify', '--run-id', 'history-success'))
            self.assertFalse(result['submission_ready']); self.assertFalse(result['promotion_supported'])
            work = root / 'artifacts/validation/history-success'
            self.assertEqual((work / 'candidate.csv').read_bytes(), (work / 'independent-replay.csv').read_bytes())
            for number in (1, 2):
                record = read(work / f'graphs/final-{number}.json')
                self.assertFalse(record['cache_used']); self.assertEqual(record['identity']['cutoff'], JAN20 + 7 * DAY)
            self.assertFalse((root / 'configs/v7_4.json').exists())
            helper.cli(root, 'develop', '--run-id', 'history-success', code=1)
            # Re-sealing an altered report must still fail recomputed V7 gates.
            screen = read(work / 'screen.json')
            screen['results']['primary']['v7_comparison']['ci_lower'] = 0
            (work / 'screen.json').write_text(json.dumps(screen))
            (work / 'screen.seal.json').write_text(json.dumps({'sha256': digest(work / 'screen.json')}))
            helper.cli(root, 'verify', '--run-id', 'history-success', code=1)

    def test_uninformative_graph_closes_recipe_without_fallback_or_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); helper = self.repository(root)
            path = fixture.create(root, 'history-failure', informative=False)
            helper.cli(root, 'prepare', '--run-id', 'history-failure', '--spec', str(path))
            helper.cli(root, 'develop', '--run-id', 'history-failure', code=2)
            work = root / 'artifacts/validation/history-failure'
            for stage in ('screen', 'stress', 'finalize'):
                helper.cli(root, stage, '--run-id', 'history-failure', code=1)
                self.assertFalse((work / (stage + '.attempt.json')).exists())
            self.assertFalse((root / 'configs/v7_4.json').exists())

    def test_adapter_cannot_mutate_independent_lineage_or_normalization(self):
        from ml_workflow.contracts import CaseSpec
        for bad_field in ('metadata', 'normalization'):
            with self.subTest(field=bad_field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); helper = self.repository(root)
                path = fixture.create(root, 'mutations')
                helper.cli(root, 'prepare', '--run-id', 'mutations', '--spec', str(path))
                work = root / 'artifacts/validation/mutations'
                m = engine.load(root, work); module = engine.adapter(work, m); raw = engine.events(root, m, 'train')
                original = module.fit_calibrator
                def mutate(*args):
                    saved = original(*args)
                    if bad_field == 'metadata': saved['metadata']['meta_cutoff'] += DAY
                    else: saved['normalization']['rms'] *= 2
                    return saved
                with patch.object(module, 'fit_calibrator', side_effect=mutate):
                    with self.assertRaisesRegex(ValueError, 'expected calibrator|normalization differs'):
                        history.score(root, work, m, module, raw, 'history_24h', CaseSpec.parse(m['spec']['development'][0]))
                self.assertEqual(m['spec_sha256'], canonical(m['spec']))
                self.assertFalse((work / 'predictions').exists())
