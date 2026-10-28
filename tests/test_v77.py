"""Mandatory CPU contracts; synthetic evidence never promotes a real model."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments/v7.7-reliability-slate-v1'), str(ROOT / 'tools'), str(ROOT / 'legacy'),
                str(ROOT / 'experiments/v7.4-rebuild-v1')]
import v77_protocol as protocol
import v77_features as features
import v77_model as model
import v77_assets as assets
import v77_runner as runner
import v77_workflow as workflow
from ml_workflow import promotion, historical
from ml_workflow.contracts import canonical, digest, DAY, JAN20, first_indices
from ml_workflow.storage import read, write_new, save_arrays
from test_v75_audit import evidence_chain


def sample(n=80):
    rng = np.random.default_rng(2026)
    event = rng.normal(size=(n, 11)); row = rng.normal(size=(n, 10, 13)); row -= row.mean(1, keepdims=True)
    anchor = rng.uniform(.02, .3, (n, 10)); labels = np.zeros((n, 10), dtype=np.int8)
    labels[np.arange(n), np.argmax(row[:, :, 0], axis=1)] = 1; labels[event[:, 0] < 0] = 0
    return dict(event=event, row=row, anchor=anchor, indices=np.arange(n)), labels


def test_metadata_schedule_is_mature_and_causal():
    assert [len(protocol.prefixes(m)) for m in protocol.METAS] == [1, 2, 3, 3]
    for m in protocol.METAS:
        assert all(JAN20 + DAY <= p <= m and m - p <= 2 * DAY for p in protocol.prefixes(m))
    with pytest.raises(ValueError): protocol.prefixes(JAN20)
    assert not protocol.SPEC['retrain_neural']


def test_first_return_population_matches_query_since_each_prefix():
    raw = dict(u=np.array([1, 1, 2, 2, 3, 4]), t=np.array([1, 11, 12, 13, 20, 10]))
    np.testing.assert_array_equal(features.cohort(raw, 0, 10), [2, 4, 5])
    np.testing.assert_array_equal(features.cohort(raw, 10, 10), [1, 2, 4, 5])
    assert 1 not in features.cohort(raw, 0, 10)
    with pytest.raises(ValueError): features.cohort(raw, 11, 10)


def test_stable_folds_do_not_depend_on_view_order_or_future_users():
    u = np.arange(100); f = protocol.folds(u)
    assert set(f) == set(range(5))
    np.testing.assert_array_equal(protocol.folds(u[::-1]), f[::-1])
    np.testing.assert_array_equal(protocol.folds(np.r_[u, 9999])[:100], f)
    np.testing.assert_array_equal(protocol.folds(u[::3]), f[::3])


def test_strata_are_balanced_and_duplicate_event_influence_is_capped():
    ix = np.array([1, 2, 3, 1, 3]); strata = np.array([[0, 0], [0, 0], [0, 0], [1, 2], [1, 2]])
    w = features.balanced_weights(ix, strata)
    assert w[:3].sum() == pytest.approx(w[3:].sum())
    assert np.bincount(ix, weights=w).max() <= 1
    np.testing.assert_allclose(features.balanced_weights(ix[::-1], strata[::-1]), w[::-1])


def test_context_uses_only_prefix_labels_and_query_available_history():
    raw = dict(u=np.array([0, 0, 1]), t=np.array([0, 2, 5]), y=np.zeros((3, 10)),
               h=np.full((3, 20), -1), ht=np.zeros((3, 20), dtype=int))
    raw['y'][0, 0] = 1
    out = features.user_context(raw, raw, np.array([1, 2]), 1)
    changed = deepcopy(raw); changed['y'][1:, 0] = 1
    np.testing.assert_array_equal(features.user_context(changed, raw, np.array([1, 2]), 1), out)
    assert out[0, 0] == np.log(2) and out[0, 1] == pytest.approx((1 + 50 * .1) / 60)
    assert out[1, 6] == 1 and out[1, 3] == np.log1p(720)
    assert out[:, 5].tolist() == [np.log1p(720)] * 2


def test_slate_features_are_equivariant_tie_symmetric_and_nonconstant():
    rng = np.random.default_rng(14); vectors = [rng.normal(size=(20, 10, d)) for d in (8, 8, 8, 16, 16)]
    cosine = features.geometry(vectors); members = rng.uniform(.01, .2, (5, 20, 10)); anchor = members.mean(0)
    context = rng.normal(size=(20, 7)); event, row = features.master(members, anchor, context, cosine)
    permutation = rng.permutation(10)
    e, r = features.master(members[:, :, permutation], anchor[:, permutation], context, cosine[:, permutation][:, :, permutation])
    np.testing.assert_allclose(e, event, atol=1e-14)
    np.testing.assert_allclose(r, row[:, permutation], atol=1e-14)
    np.testing.assert_allclose(row.mean(1), 0, atol=1e-14)
    assert (row[:, :, -3:].std((0, 1)) > .01).all()
    np.testing.assert_allclose(features.rank_credit(np.ones((2, 10))), .5)
    e, r = features.master(np.zeros_like(members), np.zeros_like(anchor), context, cosine)
    assert np.isfinite(e).all() and np.isfinite(r).all()


@pytest.mark.parametrize('candidate,width', [('A', 5), ('B', 12), ('C', 18), ('D', 25)])
def test_real_head_fit_is_bounded_deterministic_and_improves_fixed_objective(tmp_path, candidate, width):
    data, y = sample(); w = np.ones(len(y))
    state = model.fit(data, y, w, candidate, {'purpose': 'synthetic_validation'}, tmp_path / 'head.json')
    assert len(state['coefficients']) == width and state['objective'] < state['identity_objective']
    np.testing.assert_array_equal(model.predict(data, state), model.predict(data, model.load(tmp_path / 'head.json')))
    other = model.fit(data, y, w, candidate, {'purpose': 'synthetic_validation'}, tmp_path / 'again.json')
    assert other == state
    X = model.standardized(data, state['normalization'])
    delta = model.correction(torch.tensor(state['coefficients'], dtype=torch.float64), torch.tensor(X['event']), torch.tensor(X['row']), candidate).numpy()
    assert np.abs(delta).max() <= (.5 if width in (18, 25) else .25)


def test_zero_identity_endpoints_cold_users_and_initial_gradients(tmp_path):
    data, y = sample(30); data['anchor'][0, :2] = [0, 1]
    state = model.fit(data, y, np.ones(30), 'D', {'purpose': 'synthetic_validation'}, tmp_path / 'head.json')
    p = model.predict(data, state); assert p[0, :2].tolist() == [0, 1]
    identity = deepcopy(state); identity['coefficients'] = [0.] * 25
    np.testing.assert_array_equal(model.predict(data, identity), data['anchor'])
    X = model.standardized(data, state['normalization']); theta = torch.zeros(25, dtype=torch.float64, requires_grad=True)
    correction = model.correction(theta, torch.tensor(X['event']), torch.tensor(X['row']), 'D')
    assert torch.count_nonzero(correction) == 0
    (correction * torch.tensor(y - .1)).sum().backward()
    assert torch.abs(theta.grad[:12]).sum() > 0 and torch.abs(theta.grad[12:]).sum() > 0
    # No history/coverage activation mask exists: every event can change.
    assert (p != data['anchor']).any(1).all()


def test_normalization_is_weighted_and_cluster_ess_counts_original_events(tmp_path):
    data, y = sample(4); data['indices'] = np.array([1, 2, 2, 2]); weights = np.array([1., 1/3, 1/3, 1/3])
    norm = model.normalization(data, weights)
    np.testing.assert_allclose(norm['event']['mean'], np.average(data['event'], axis=0, weights=weights))
    s = model.fit(data, y, weights, 'A', {'purpose': 'synthetic_validation'}, tmp_path / 'head.json')
    assert s['effective_events'] == pytest.approx(2) and s['unique_events'] == 2
    assert np.abs(model.standardized({'event': data['event'] * 1e6, 'row': data['row'] * 1e6}, norm)['row']).max() == 5


def test_nonconvergence_and_invalid_model_provenance_fail_closed(tmp_path, monkeypatch):
    data, y = sample()
    from types import SimpleNamespace
    monkeypatch.setattr(model, 'minimize', lambda *a, **k: SimpleNamespace(success=False, fun=1., message='fixture limit'))
    with pytest.raises(ValueError, match='converge'): model.fit(data, y, np.ones(len(y)), 'A', {'purpose': 'research'}, tmp_path / 'head.json')
    assert not (tmp_path / 'head.json').exists()


def test_objective_matches_independent_numpy_loss_and_finite_difference(tmp_path, monkeypatch):
    data, y = sample(12); weights = np.linspace(.2, 1, 12)
    X = model.standardized(data, model.normalization(data, weights))
    theta = np.linspace(-.2, .2, 18)
    def loss(t):
        shift = .25 * np.tanh(t[0] + X['event'][:, :4] @ t[1:5])
        relative = .125 * np.tanh(X['row'] @ t[5:]); relative -= relative.mean(1, keepdims=True)
        z = np.log(data['anchor']) - np.log1p(-data['anchor']) + shift[:, None] + relative
        row_w = np.repeat(weights, 10); positive = np.flatnonzero(y.ravel()); negative = np.flatnonzero(~y.ravel().astype(bool))
        selected = np.random.default_rng(5026).choice(negative, (len(positive), 16), replace=True,
                                                    p=row_w[negative] / row_w[negative].sum())
        bce = ((np.logaddexp(0, z) - y * z) * weights[:, None]).sum() / (10 * weights.sum())
        pair = np.average(np.logaddexp(0, z.ravel()[selected] - z.ravel()[positive, None]).mean(1), weights=row_w[positive])
        return .5 * (bce + pair) + .01 * np.square(t).sum()
    def inspect(fun, initial, **kwargs):
        from types import SimpleNamespace
        actual, gradient = fun(theta)
        assert actual == pytest.approx(loss(theta), abs=1e-14)
        finite = np.array([(loss(theta + np.eye(18)[i] * 1e-6) - loss(theta - np.eye(18)[i] * 1e-6)) / 2e-6 for i in range(18)])
        np.testing.assert_allclose(gradient, finite, atol=1e-9, rtol=0)
        return SimpleNamespace(success=True, fun=fun(initial)[0], x=initial, nit=0, message='synthetic gradient inspection')
    monkeypatch.setattr(model, 'minimize', inspect)
    model.fit(data, y, weights, 'C', {'purpose': 'synthetic_validation'}, tmp_path / 'head.json')


def eligible_rows():
    rows = {}
    for candidate in 'ABCD':
        for case in ('jan22_long', 'jan24_forward'):
            r = dict(baseline_auc=.8, candidate_auc=.801, auc_diff=.001)
            r['late'] = dict(r, events=20); r['v7_comparison'] = deepcopy(r)
            rows[candidate + '/' + case] = r
    return rows


def test_selection_maximin_aged_gates_and_frozen_order():
    rows = eligible_rows(); assert workflow.choose(rows) == 'A'
    rows['A/jan22_long']['v7_comparison']['late'].update(candidate_auc=.799, auc_diff=-.001)
    assert workflow.choose(rows) == 'B'
    for r in rows.values():
        r.update(candidate_auc=.799, auc_diff=-.001)
    assert workflow.choose(rows) is None


@pytest.mark.parametrize('gate', ['primary', 'secondary', 'five_day', 'late'])
def test_each_historical_floor_requires_strictly_greater_point(gate):
    rows = {g: {'V6': {'candidate_auc': value[3] + .00001}} for g, value in historical.FLOORS.items()}
    assert all(historical.floor_gates(rows, {g: v[3] for g, v in historical.FLOORS.items()}).values())
    rows[gate]['V6']['candidate_auc'] = historical.FLOORS[gate][3]
    assert not historical.floor_gates(rows, {g: v[3] for g, v in historical.FLOORS.items()})[gate]


@pytest.mark.parametrize('anchor,gate', [(a, g) for a in ('V6', 'V7') for g in ('primary', 'secondary', 'five_day', 'late')])
def test_each_original_gate_independently_blocks_release(anchor, gate):
    r = dict(baseline_auc=.8, candidate_auc=.801, auc_diff=.001, resamples=1000, ci_lower=.0001, ci_upper=.002)
    rows = {g: {a: deepcopy(r) for a in ('V6', 'V7')} for g in ('primary', 'secondary', 'five_day', 'late')}
    if gate in ('primary', 'five_day'): rows[gate][anchor]['ci_lower'] = 0
    else: rows[gate][anchor].update(candidate_auc=.799, auc_diff=-.001)
    checked = promotion.metric_gates(rows)
    assert not checked['gates'][anchor][gate] and len(checked['blockers']) == 1


@pytest.fixture
def extended_chain(evidence_chain):
    root, e, refresh, rewrite = evidence_chain
    from submission_export import export_submission
    source = read(root / 'source.json'); floors = {'schema_version': 1, 'profile': historical.PROFILE, 'floors': {}}
    documents = {}
    for name, (filename, keys, count, value) in historical.FLOORS.items():
        data = documents.setdefault(filename, {})
        for k in keys: data = data.setdefault(k, {})
        data['candidate_auc'] = value
    for filename, data in documents.items():
        write_new(root / filename, data); write_new(root / 'snapshot/source' / filename, data)
        source['files'][filename] = digest(root / filename)
    for name, (filename, keys, count, value) in historical.FLOORS.items():
        floors['floors'][name] = dict(path=filename, keys=keys, events=count, auc=value, sha256=digest(root / filename))
    (root / 'source.json').write_text(json.dumps(source)); write_new(root / 'floors.json', floors)
    e.update(schema_version=2, release_profile=historical.PROFILE, historical_floors='floors.json')
    baseline = {'data_sha256': digest(root / 'metadata.json'), 'recipe': {'fixture': True}, 'models': {'model.pt': digest(root / 'model.pt')}}
    identities = {a: canonical({'anchor': a, 'baseline': baseline}) for a in ('V6', 'V7')}
    e['comparators'] = identities
    rewrite('manifest.json', lambda m: m.update(baseline_inputs=baseline, comparators=identities,
        protocol={**m['protocol'], 'release_profile': historical.PROFILE}))
    for filename in e['predictions'].values():
        data = workflow.arrays(root / filename)
        data.update(**{a + '_identity': np.asarray(v) for a, v in identities.items()})
        with (root / filename).open('wb') as stream: np.savez(stream, **data)
    data = workflow.arrays(root / 'primary.npz')
    for number in (1, 2, 3):
        csv = root / f'new-export{number}.csv'
        export_submission(data['rowids'].ravel(), data['candidate'], root / 'example.csv', csv, len(data['rowids']) * 10)
        vectors = root / f'vectors{number}.npz'
        save_arrays(vectors, candidate=data['candidate'], rowids=data['rowids'], members=np.stack([data['baseline']] * 5))
        receipt = dict(candidate='A', number=number, purpose='research', fresh_neural_features=True,
                       fresh_context_features=True, fresh_slate_features=True, source_sha256=source['sha256'],
                       manifest_sha256=digest(root / 'manifest.json'), model_sha256=canonical(read(root / 'manifest.json')['models']),
                       csv_sha256=digest(csv), vectors_sha256=digest(vectors), compute_owner=f'generation/export{number}')
        write_new(root / f'new-export{number}.json', receipt)
    rewrite('ledger.json', lambda ledger: (ledger['attempts'].append(dict(owner='generation/export3', status='complete', seconds=1.)), ledger.update(used_seconds=3.)))
    e['exports'] = [dict(csv=f'new-export{n}.csv', receipt=f'new-export{n}.json', vectors=f'vectors{n}.npz') for n in (1, 2, 3)]
    refresh()
    return root, e, refresh, rewrite


def test_schema2_recomputes_all_twelve_checks_and_honest_freshness(extended_chain):
    root, e, _, _ = extended_chain
    decision = promotion.verify_promotion(root, e)
    assert decision['eligible'], decision['blockers']
    assert all(decision['historical_floors'].values())
    assert 'fresh_graph' not in read(root / 'new-export1.json')


@pytest.mark.parametrize('failure', ['context', 'slate', 'floor_source', 'two_exports', 'members', 'profile', 'head_in_baseline'])
def test_schema2_rejects_corrupt_or_incomplete_proof(extended_chain, failure):
    root, e, refresh, rewrite = extended_chain
    if failure in ('context', 'slate'):
        rewrite('new-export1.json', lambda r: r.update(**{'fresh_' + failure + '_features': False}))
    elif failure == 'floor_source':
        filename = historical.FLOORS['late'][0]; (root / filename).write_text('{}'); refresh()
    elif failure == 'two_exports': e['exports'] = e['exports'][:2]
    elif failure == 'members':
        filename = root / 'vectors3.npz'; data = workflow.arrays(filename); data['members'][0, 0, 0] += .01
        with filename.open('wb') as stream: np.savez(stream, **data)
        rewrite('new-export3.json', lambda r: r.update(vectors_sha256=digest(filename)))
    elif failure == 'profile': e['release_profile'] = 'relaxed'
    else: rewrite('manifest.json', lambda m: m['baseline_inputs'].update(coefficients=[1.]))
    assert not promotion.verify_promotion(root, e)['eligible']


def test_small_actual_oof_head_and_csv_pipeline(tmp_path):
    """Real CatBoost OOF, slate/context, all four fitted heads and CSV replay."""
    from submission_export import export_submission, validate_csv
    n = 90; rng = np.random.default_rng(53); u = np.arange(n)
    labels = np.zeros((n, 10), np.int8); labels[np.arange(n), rng.integers(0, 10, n)] = 1; labels[::3] = 0
    raw = dict(u=u, y=labels, t=np.full(n, JAN20 + DAY), i=rng.integers(0, 200, (n, 10)),
               h=np.full((n, 20), -1), ht=np.zeros((n, 20), np.int64),
               rowids=np.array([[f'e{i}-c{j}' for j in range(10)] for i in range(n)]))
    X = rng.normal(size=(n * 10, 37)).astype(np.float32); X[:, 0] += labels.ravel()
    data = dict(X=X, feature_names=np.array([f'x{k}' for k in range(37)]), indices=np.arange(n))
    probabilities = []
    parameters = dict(protocol.COMBINER, iterations=3, depth=2)
    for member in protocol.MEMBERS:
        folder = tmp_path / member; probabilities.append(assets.oof(data, raw, folder, parameters))
        receipt = read(folder / 'oof.json')
        assert sum(row['held_events'] for row in receipt['folds']) == n
        assert all(row['training_events'] + row['held_events'] == n for row in receipt['folds'])
    members = np.stack(probabilities); anchor = members.mean(0)
    fitting = dict(raw, t=np.full(n, JAN20 - 1))
    event, row = features.master(members, anchor, features.user_context(fitting, raw, np.arange(n), JAN20),
                                 features.geometry([rng.normal(size=(n, 10, d)) for d in (8, 8, 8, 16, 16)]))
    values = dict(indices=np.arange(n), anchor=anchor, event=event, row=row)
    example = tmp_path / 'example.csv'
    example.write_text('rowid,predicted_probability\n' + ''.join(r + ',0\n' for r in raw['rowids'].ravel()))
    for candidate in 'ABCD':
        state = model.fit(values, labels, np.ones(n), candidate, {'purpose': 'synthetic_validation'}, tmp_path / (candidate + '.json'))
        p = model.predict(values, state)
        for run in range(3):
            csv = tmp_path / f'{candidate}-{run}.csv'; export_submission(raw['rowids'].ravel(), p, example, csv, n * 10)
            assert validate_csv(csv, example, n * 10)['rows'] == n * 10
        assert digest(tmp_path / f'{candidate}-0.csv') == digest(tmp_path / f'{candidate}-2.csv')
    assert not (tmp_path / 'configs/v7_7.json').exists()


@pytest.fixture
def launch_fixture(tmp_path, monkeypatch):
    root = tmp_path; work = root / runner.WORK; parent = root / 'artifacts/rebuild'
    write_new(parent / 'genesis.json', dict(generation='fixture'))
    write_new(parent / 'compute-ledger.json', dict(total_seconds=43200, used_seconds=0., carry_identity=digest(parent / 'genesis.json'), attempts=[]))
    write_new(work / 'manifest.json', dict(protocol=protocol.SPEC, genesis_sha256=digest(parent / 'genesis.json')))
    source = dict(sha256='a' * 64)
    write_new(work / 'scientific-source.json', dict(snapshot_relative='snapshot', source=source))
    monkeypatch.setattr(runner, 'verify_snapshot', lambda *a: None)
    monkeypatch.setattr(runner, 'scientific_open', lambda *a: None)
    actual_completed = runner.completed
    monkeypatch.setattr(runner, 'completed', lambda r, w, action, arguments=None, adopt=False:
                        {'passed': True} if action == 'preflight' else actual_completed(r, w, action, arguments, adopt))
    def fake_worker(budget, command, cwd):
        token = command[-1]; folder = work / 'jobs' / token
        write_new(folder / 'result.json', dict(csv=str(folder / 'export.csv')))
        (folder / 'export.csv').write_text('fixture attempt\n')
        files = {p.relative_to(work).as_posix(): digest(p) for p in folder.iterdir()}
        write_new(work / 'attempts' / (token + '.validated.json'), dict(source_sha256=source['sha256'],
            claim_sha256=digest(work / 'attempts' / (token + '.claim.json')), files=files,
            result=(folder / 'result.json').relative_to(work).as_posix()))
        return 0
    monkeypatch.setattr(runner.CampaignBudget, 'run', fake_worker)
    return root, work


def test_failed_export_accounting_retries_fresh_attempt_without_clobber(launch_fixture, monkeypatch):
    root, work = launch_fixture
    import ml_workflow.budget as budget_module
    original = budget_module.write_json; fail = [True]
    def interrupted(filename, state):
        if state['attempts'][-1]['status'] == 'complete' and fail[0]:
            fail[0] = False; raise OSError('accounting interrupted')
        return original(filename, state)
    monkeypatch.setattr(budget_module, 'write_json', interrupted)
    with pytest.raises(OSError, match='accounting interrupted'): runner.launch(root, work, 'export', dict(number=1))
    assert not (work / 'tasks' / (runner.key('export', dict(number=1)) + '.json')).exists()
    old = list((work / 'jobs').glob('*/export.csv'))[0]; old_hash = digest(old)
    runner.launch(root, work, 'export', dict(number=1))
    exports = list((work / 'jobs').glob('*/export.csv'))
    assert len(exports) == 2 and digest(old) == old_hash
    ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
    assert ledger['attempts'][0]['status'] == 'abandoned_reserved'
    assert ledger['attempts'][0]['seconds'] == 1200 and ledger['attempts'][1]['status'] == 'complete'


def test_completed_export_adoption_keeps_owner_and_does_not_charge_again(launch_fixture, monkeypatch):
    root, work = launch_fixture
    runner.launch(root, work, 'export', dict(number=1))
    marker = work / 'tasks' / (runner.key('export', dict(number=1)) + '.json'); original = read(marker)
    marker.unlink()
    runner.launch(root, work, 'export', dict(number=1))
    assert read(marker) == original
    assert len(read(root / 'artifacts/rebuild/compute-ledger.json')['attempts']) == 1


def test_paid_verification_resume_reuses_frozen_descriptor_and_seal(tmp_path):
    work = tmp_path / runner.WORK
    write_new(work / 'sealed.json', dict(attempts=[1])); expected = {'artifacts': {(work / 'sealed.json').relative_to(tmp_path).as_posix(): digest(work / 'sealed.json')}}
    write_new(work / 'promotion-evidence.json', expected)
    write_new(tmp_path / 'artifacts/rebuild/compute-ledger.json', dict(attempts=[1, 2], used_seconds=9.))
    assert read(workflow.evidence(tmp_path, work)) == expected
    (work / 'sealed.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'): workflow.evidence(tmp_path, work)


def test_publication_without_paid_proof_never_marks_ready(tmp_path):
    work = tmp_path / runner.WORK
    with pytest.raises((ValueError, FileNotFoundError)): workflow.publish(tmp_path, work)
    assert not (tmp_path / 'configs/v7_7.json').exists() and not (work / 'readiness.json').exists()


def test_immutable_record_preserves_different_existing_bytes(tmp_path):
    filename = tmp_path / 'receipt.json'; runner.same(filename, {'value': 1})
    runner.same(filename, {'value': 1})
    with pytest.raises(ValueError, match='differs'): runner.same(filename, {'value': 2})
    assert read(filename) == {'value': 1}


def test_closed_campaign_consumes_no_new_budget_or_claims(tmp_path):
    work = tmp_path / runner.WORK
    from unittest.mock import patch
    with patch.object(runner, 'completed', return_value={'passed': False}), patch.object(runner, 'CampaignBudget') as budget:
        with pytest.raises(ValueError, match='terminal'): runner.scientific_open(tmp_path, work)
        budget.assert_not_called()
    assert not work.exists()


def test_phase_ceiling_and_finalization_reserve_are_checked_under_lock(launch_fixture, monkeypatch):
    root, work = launch_fixture
    monkeypatch.setitem(protocol.SPEC['phase_limits'], 'prepare', 100.)
    with pytest.raises(ValueError, match='phase ceiling'): runner.launch(root, work, 'export', dict(number=1))
    assert not (work / 'attempts').exists()
    ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
    assert ledger['attempts'][0]['status'] == 'failed'


def test_controller_orders_all_stages_and_publishes_after_paid_verification(tmp_path, monkeypatch):
    """Orchestration fixture: real fitting/export exercised separately above."""
    log = []
    def launch(root, work, action, arguments=None, *args, **kwargs):
        log.append((action, arguments))
        return {'passed': True, 'winner': 'A'}
    monkeypatch.setattr(runner, 'scientific_open', lambda *a: None)
    monkeypatch.setattr(runner, 'launch', launch)
    for name in ('evidence', 'publish', 'report'):
        monkeypatch.setattr(workflow, name, lambda *a, _name=name: log.append((_name, None)))
    monkeypatch.setattr(workflow, 'verify', lambda *a: {})
    workflow.controller(tmp_path, tmp_path / runner.WORK, 'run')
    assert [args['number'] for name, args in log if name == 'export'] == [1, 2, 3]
    names = [name for name, args in log]
    assert names.index('selection') < names.index('freeze') < names.index('screen') < names.index('stress')
    assert names.index('evidence') < names.index('verify-release') < names.index('publish') < names.index('report')
    head_calls = [args for name, args in log if name == 'head']
    assert len(head_calls) == 10 and all(args['candidate'] == 'A' for args in head_calls[-2:])
