"""V7.6 causal forecasts, real CPU residual fitting and guarded delivery contracts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments/v7.6-anchored-residual-v2'), str(ROOT / 'tools'), str(ROOT / 'legacy')]
import runner as v76
import v76_protocol as protocol
import v76_assets as assets
import v76_workflow as workflow
import residual
from ml_workflow.contracts import canonical, digest, first_indices
from ml_workflow.storage import read, write_new, save_arrays


def sample(n=50):
    rng = np.random.default_rng(16)
    X = rng.normal(size=(n, 10, 13)); anchor = np.full((n, 10), .06)
    y = np.zeros_like(anchor); y[np.arange(n), np.argmax(X[:, :, 0], axis=1)] = 1; y[::3] = 0
    return dict(X=X, anchor=anchor, active=np.ones(n, bool), indices=np.arange(n)), y


def metadata(**kwargs): return {'purpose': 'synthetic_validation', **kwargs}


def test_forecast_pairs_are_causal_and_cover_stale_prefixes():
    views = [(b, c, d, e) for b in protocol.BLOCKS for c, d, e in protocol.pairs(b)]
    assert len(views) == 158
    for b, c, d, e in views:
        assert c < e <= min(c + protocol.DAY, d) <= b
        assert 6 * protocol.HOUR <= d - c <= 48 * protocol.HOUR
    assert max((b - d) / protocol.HOUR for b, c, d, e in views) > 120
    assert all(e <= protocol.JAN20 + 6 * protocol.DAY for b, c, d, e in views)
    with pytest.raises(ValueError): protocol.pairs(protocol.JAN20)


def test_inner_regimes_finish_before_first_outer_target():
    for b in protocol.INNER:
        regimes = protocol.validation_pairs(b)
        assert regimes['freshest'][1] == protocol.JAN20 + protocol.DAY
        assert regimes['oldest'][1] == protocol.JAN20 + 12 * protocol.HOUR
        assert b + 6 * protocol.HOUR <= protocol.JAN20 + 2 * protocol.DAY


def test_user_groups_are_stable_across_order_subsets_and_future_users():
    u = np.arange(100); full = protocol.user_folds(u)
    assert set(full) == set(range(5))
    np.testing.assert_array_equal(protocol.user_folds(u[::-1]), full[::-1])
    np.testing.assert_array_equal(protocol.user_folds(u[::3]), full[::3])
    np.testing.assert_array_equal(protocol.user_folds(np.r_[u, 999])[:100], full)


def test_fitting_window_excludes_future_and_validation_users():
    times = protocol.JAN20 + np.arange(12, 168) * protocol.HOUR
    raw = dict(u=np.arange(len(times)) % 31, t=times)
    b = protocol.BOUNDARIES[1]
    for fold in range(5):
        ix = protocol.fitting_indices(raw, b, fold)
        assert ((times[ix] >= b - 2 * protocol.DAY) & (times[ix] < b)).all()
        assert (protocol.user_folds(raw['u'][ix]) != fold).all()
        assert len(np.unique(raw['u'][ix])) == len(ix)
    with pytest.raises(ValueError): protocol.fitting_indices(raw, b, 5)


def test_weighted_normalization_counts_each_event_once():
    values, _ = sample(2); values['X'][0] = 0; values['X'][1] = 2
    duplicated = {k: np.concatenate([v[[0]], v[[1]], v[[1]], v[[1]]]) for k, v in values.items()}
    norm = residual.normalize_fit(duplicated)
    np.testing.assert_allclose(norm['mean'], np.ones(13), atol=1e-14, rtol=0)
    np.testing.assert_allclose(norm['scale'], np.ones(13), atol=1e-14, rtol=0)
    assert np.abs(residual.standardized(np.full((1, 10, 13), 100), norm)).max() == 5
    broken = deepcopy(norm); broken['scale'][0] = 0
    with pytest.raises(ValueError): residual.standardized(values['X'], broken)


def test_initial_identity_has_nonzero_trainable_output_gradient():
    values, labels = sample(20); torch.manual_seed(2026); model = residual.Residual()
    X = torch.tensor(values['X']); p = torch.tensor(values['anchor']); a = torch.tensor(values['active']); y = torch.tensor(labels)
    assert torch.count_nonzero(model(X)) == 0
    generator = torch.Generator().manual_seed(5026)
    residual.objective(model, X, p, a, y, 1., generator).backward()
    assert model.output.weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=.1); optimizer.step(); optimizer.zero_grad()
    residual.objective(model, X, p, a, y, 1., generator).backward()
    assert model.hidden.weight.grad.abs().sum() > 0


def test_residual_loss_matches_independent_oracle_and_finite_difference():
    import math
    values, y = sample(4); model = residual.Residual()
    with torch.no_grad(): model.output.bias.fill_(.2)
    X = torch.tensor(values['X']); p = torch.tensor(values['anchor']); active = torch.tensor(values['active']); labels = torch.tensor(y)
    pos = np.flatnonzero(y.ravel() == 1); neg = np.flatnonzero(y.ravel() == 0)
    sampled = np.resize(neg, (len(pos), 16)); base = np.log(values['anchor']) - np.log1p(-values['anchor'])
    delta = .25 * math.tanh(.2); z = base + delta
    bce = np.mean(np.logaddexp(0, z) - y * z)
    pair = np.logaddexp(0, z.ravel()[sampled] - z.ravel()[pos, None]).mean()
    kl = np.mean(np.logaddexp(0, z) - np.logaddexp(0, base) - values['anchor'] * delta)
    loss = residual.objective(model, X, p, active, labels, 10., sampled=torch.tensor(sampled))
    assert float(loss.detach()) == pytest.approx(.5 * bce + .5 * pair + 10 * kl, abs=1e-13)
    loss.backward(); analytic = float(model.output.bias.grad[0]); eps = 1e-5
    def objective_at(value):
        with torch.no_grad(): model.output.bias.fill_(value)
        return float(residual.objective(model, X, p, active, labels, 10., sampled=torch.tensor(sampled)).detach())
    numerical = (objective_at(.2 + eps) - objective_at(.2 - eps)) / (2 * eps)
    assert analytic == pytest.approx(numerical, rel=1e-6, abs=1e-9)


def test_no_click_batch_endpoints_and_neutral_events_are_finite():
    values, labels = sample(6); values['anchor'][0, :2] = [0, 1]; values['active'][1] = False
    model = residual.Residual(); norm = residual.normalize_fit(values)
    initial = residual.predict(values, model, {'normalization': norm})
    np.testing.assert_array_equal(initial, values['anchor'])
    with torch.no_grad(): model.output.bias.fill_(100)
    result = residual.predict(values, model, {'normalization': norm})
    np.testing.assert_array_equal(result[0, :2], [0, 1]); np.testing.assert_array_equal(result[1], values['anchor'][1])
    z0 = np.log(values['anchor'][2:]) - np.log1p(-values['anchor'][2:])
    z1 = np.log(result[2:]) - np.log1p(-result[2:])
    assert np.max(np.abs(z1 - z0)) <= .25 + 1e-14
    loss = residual.objective(model, torch.tensor(values['X']), torch.tensor(values['anchor']),
            torch.tensor(values['active']), torch.zeros_like(torch.tensor(labels)), 1.)
    assert torch.isfinite(loss)


def test_real_fit_reload_determinism_and_candidate_permutation(tmp_path):
    values, labels = sample(40)
    state = residual.fit(values, labels, .1, metadata(), tmp_path / 'a.pt')
    model, saved = residual.load(tmp_path / 'a.pt', metadata(), .1)
    p = residual.predict(values, model, saved)
    assert not np.array_equal(p, values['anchor']) and np.isfinite(p).all()
    assert saved['epochs'] == 32 and len(saved['objective_trace']) == 32
    permuted = {k: v[:, ::-1].copy() if k in ('X', 'anchor') else v for k, v in values.items()}
    np.testing.assert_array_equal(residual.predict(permuted, model, saved), p[:, ::-1])
    residual.fit(values, labels, .1, metadata(), tmp_path / 'b.pt')
    other, second = residual.load(tmp_path / 'b.pt')
    np.testing.assert_array_equal(residual.predict(values, other, second), p)
    with pytest.raises(ValueError): residual.load(tmp_path / 'a.pt', metadata(changed=True))
    with pytest.raises(ValueError): residual.load(tmp_path / 'a.pt', penalty=10.)


def test_normalization_ignores_validation_features_and_rejects_invalid_views(tmp_path):
    fitting, y = sample(8); residual.fit(fitting, y, 1., metadata(), tmp_path / 'fit.pt')
    model, state = residual.load(tmp_path / 'fit.pt'); expected = deepcopy(state['normalization'])
    held = deepcopy(fitting); held['X'] *= 1e6
    residual.predict(held, model, state)
    assert state['normalization'] == expected
    bad = deepcopy(fitting); bad['X'][0, 0, 0] = np.nan
    with pytest.raises(ValueError): residual.normalize_fit(bad)
    with pytest.raises(ValueError): residual.fit(fitting, y, .5, metadata(), tmp_path / 'bad.pt')


def test_future_labels_do_not_enter_historical_forecasts(tmp_path, monkeypatch):
    n = 144; times = protocol.JAN20 + np.arange(12, 12 + n) * protocol.HOUR
    raw = dict(t=times, u=np.arange(n) % 17, y=np.zeros((n, 10), dtype=np.int8))
    raw['y'][:, 0] = 1
    b = protocol.INNER[0]
    for start in protocol.BLOCKS:
        if start >= b: continue
        ix = first_indices(raw['u'], raw['t'], start, start + 6 * protocol.HOUR)
        for c, d, end in protocol.pairs(start):
            path = assets.view_path(tmp_path, c, d, end, start, start + 6 * protocol.HOUR)
            values, _ = sample(len(ix)); values['indices'] = ix
            save_arrays(path, **values); write_new(path.with_suffix('.receipt.json'), {'sha256': digest(path)})
    monkeypatch.setattr(workflow, 'completed', lambda *_: True)
    before, y, _ = workflow.fit_data(tmp_path, tmp_path, raw, b)
    raw['y'][raw['t'] >= b] = 0
    after, changed, _ = workflow.fit_data(tmp_path, tmp_path, raw, b)
    np.testing.assert_array_equal(y, changed)
    for key in before: np.testing.assert_array_equal(before[key], after[key])


def inner_rows():
    row = dict(baseline_auc=.8, candidate_auc=.801, auc_diff=.001, ci_lower=.0001, ci_upper=.002, resamples=1000)
    return {f'{p:g}': {str(b) + '/' + regime: {anchor: deepcopy(row) for anchor in ('V6', 'V7')}
            for b in protocol.INNER for regime in ('freshest', 'oldest')} for p in protocol.PENALTIES}


def test_inner_selection_tie_break_and_worst_regime():
    rows = inner_rows(); assert workflow.choose_penalty(rows) == 10.
    next(iter(rows['10'].values()))['V7']['ci_lower'] = -.0001
    assert workflow.choose_penalty(rows) == 1.
    for value in rows.values(): next(iter(value.values()))['V7'].update(candidate_auc=.799, auc_diff=-.001)
    assert workflow.choose_penalty(rows) is None


def test_inner_selection_refuses_missing_invalid_and_uninformative_results():
    rows = inner_rows(); rows['0.1'].pop(next(iter(rows['0.1'])))
    with pytest.raises(ValueError): workflow.choose_penalty(rows)
    rows = inner_rows(); next(iter(rows['1'].values()))['V6']['resamples'] = 989
    with pytest.raises(ValueError): workflow.choose_penalty(rows)
    rows = inner_rows()
    for p in rows.values():
        for case in p.values():
            for row in case.values(): row.update(candidate_auc=.8, auc_diff=0, ci_lower=0)
    assert workflow.choose_penalty(rows) is None


def test_worker_and_internal_cli_refuse_unleased_execution(tmp_path):
    with pytest.raises(FileNotFoundError): v76.worker(tmp_path, 'innerfit', 'no-claim')
    with patch.object(sys, 'argv', ['runner.py', 'innerfit', '--root', str(tmp_path)]):
        with pytest.raises(ValueError, match='controller lease'): v76.main()
    assert not (tmp_path / 'artifacts/rebuild/compute-ledger.json').exists()


def test_closed_candidate_prevents_new_reservations(tmp_path):
    work = tmp_path / v76.WORK; write_new(work / 'selection.json', {'passed': False})
    with pytest.raises(ValueError, match='terminal'): v76.launch(tmp_path, 'feature', 'new-feature')
    assert not (work / 'attempts').exists() and not (tmp_path / 'artifacts/rebuild/compute-ledger.json').exists()


def test_failed_final_accounting_cannot_mark_task_or_publish(tmp_path, monkeypatch):
    work = tmp_path / v76.WORK; parent = tmp_path / 'artifacts/rebuild'
    write_new(parent / 'genesis.json', {'generation': 'fixture'}); write_new(parent / 'compute-ledger.json', {'used_seconds': 0})
    write_new(work / 'manifest.json', {}); source = {'sha256': 'a' * 64}
    class FailedAccounting:
        def __init__(self, *_args, **_kwargs): self.state = {'attempts': []}
        def __enter__(self): return self
        def run(self, command, cwd):
            token = command[-1]; claim = work / 'attempts' / (token + '.claim.json')
            write_new(work / 'attempts' / (token + '.validated.json'),
                {'manifest_sha256': digest(work / 'manifest.json'), 'source_sha256': source['sha256'], 'claim_sha256': digest(claim), 'files': {}})
            return 0
        def __exit__(self, *_): raise OSError('final accounting failed')
    monkeypatch.setattr(v76, 'completed', lambda *_: False)
    monkeypatch.setattr(v76, 'initialize', lambda *_: (work, {}))
    monkeypatch.setattr(v76.rebuild, 'freeze', lambda *_: (work / 'snapshots/fixture', source))
    monkeypatch.setattr(v76.rebuild, 'ReconstructionBudget', FailedAccounting)
    with pytest.raises(OSError, match='accounting'): v76.launch(tmp_path, 'preflight', 'preflight')
    assert not (work / 'tasks/preflight.json').exists() and not (work / 'readiness.json').exists()


def test_publication_requires_settled_promotion_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'completed', lambda *_: False)
    with pytest.raises(ValueError, match='settle'): workflow.publish(tmp_path, tmp_path / 'work')
    assert not (tmp_path / 'artifacts/submissions/v7_6.csv').exists()


def test_synthetic_or_wrong_cohort_state_cannot_be_used_as_research_release():
    manifest = {'penalty': 1., 'source_sha256': 'a', 'selection_sha256': 'b'}
    state = {'penalty': 1., 'metadata': {'purpose': 'research', 'boundary': 1, 'source_sha256': 'a', 'selection_sha256': 'b'}}
    workflow.require_residual_role(state, manifest, 1)
    for change in ({'purpose': 'synthetic_validation'}, {'boundary': 2}, {'source_sha256': 'other'}, {'selection_sha256': 'other'}):
        bad = deepcopy(state); bad['metadata'].update(change)
        with pytest.raises(ValueError): workflow.require_residual_role(bad, manifest, 1)


def test_graph_neutral_features_and_zero_mass_are_finite():
    raw = dict(u=np.array([0, 1]), t=np.array([0, 20]), h=np.full((2, 20), -1))
    X = assets.make_features(raw, np.arange(2), 10, np.zeros((2, 10)), np.zeros((2, 10, 7)))
    assert not X['active'].any() and np.isfinite(X['X']).all()
    np.testing.assert_allclose(X['X'][:, :, 9], .1)
    np.testing.assert_array_equal(X['X'][:, :, 12], np.array([[False] * 10, [True] * 10]))


@pytest.mark.parametrize('changed', ['claim', 'source', 'output'])
def test_task_completion_binds_claim_source_output_and_accounting(tmp_path, changed):
    work = tmp_path / v76.WORK; token = 'test-token'; source = {'sha256': 'a' * 64}
    write_new(work / 'manifest.json', {})
    write_new(work / 'scientific-source.json', {'source': source})
    claim = work / 'attempts' / (token + '.claim.json')
    write_new(claim, {'task': 'test', 'source': source, 'manifest_sha256': digest(work / 'manifest.json')})
    output = work / 'output.json'; write_new(output, {})
    validation = work / 'attempts' / (token + '.validated.json')
    write_new(validation, {'manifest_sha256': digest(work / 'manifest.json'), 'source_sha256': source['sha256'],
                         'claim_sha256': digest(claim), 'files': {'output.json': digest(output)}})
    write_new(work / 'tasks/test.json', {'accounted': True, 'manifest_sha256': digest(work / 'manifest.json'),
              'token': token, 'validation_sha256': digest(validation)})
    write_new(tmp_path / 'artifacts/rebuild/genesis.json', {'generation': 'fixture'})
    write_new(tmp_path / 'artifacts/rebuild/compute-ledger.json', {'attempts': [{'owner': 'fixture/v76-' + token, 'status': 'complete'}]})
    assert v76.completed(tmp_path, work, 'test')
    if changed == 'claim': claim.write_text('{}')
    elif changed == 'source': (work / 'scientific-source.json').write_text(json.dumps({'source': {'sha256': 'b'}}))
    else: output.write_text('changed')
    with pytest.raises((ValueError, KeyError)): v76.completed(tmp_path, work, 'test')


@pytest.fixture
def export_fixture(tmp_path, monkeypatch):
    values, labels = sample(4); work = tmp_path / 'work'; path = workflow.residual_path(work, protocol.JAN20 + 6 * protocol.DAY, 1.)
    residual.fit(values, labels, 1., metadata(), path)
    rowids = np.array([[f'row-{i}-{j}' for j in range(10)] for i in range(4)])
    evaluation = {'u': np.arange(4), 'rowids': rowids}
    values.update(rowids=rowids, members=np.full((5, 4, 10), .06))
    example = tmp_path / 'data/raw/example_submission.csv'; example.parent.mkdir(parents=True)
    example.write_text('rowid,predicted_probability\n' + ''.join(f'{v},0\n' for v in rowids.ravel()))
    write_new(tmp_path / 'artifacts/rebuild/genesis.json', {'generation': 'synthetic'})
    write_new(work / 'run-manifest.json', {'models': {}, 'penalty': 1.})
    monkeypatch.setattr(workflow, 'EVAL_EVENTS', 4); monkeypatch.setattr(workflow, 'EVAL_ROWS', 40)
    monkeypatch.setattr(workflow, 'require_gates', lambda *_: {})
    monkeypatch.setattr(workflow, 'frozen', lambda *_: {'models': {}, 'penalty': 1.})
    monkeypatch.setattr(workflow, 'completed', lambda *_: True)
    # These small synthetic exports exercise replay/CSV mechanics, never release eligibility.
    monkeypatch.setattr(workflow, 'require_residual_role', lambda *_: None)
    monkeypatch.setattr(workflow.dataset, 'load', lambda *_: evaluation)
    calls = []
    def fresh(*args, **kwargs):
        assert kwargs['fresh'] is True; calls.append(True)
        return values, {}, {'cutoff': np.asarray(protocol.FULL)}
    monkeypatch.setattr(assets, 'views', fresh)
    return tmp_path, work, values, calls


def test_independent_fresh_exports_match_vectors_csv_and_owners(export_fixture):
    root, work, values, calls = export_fixture
    for number in (1, 2): workflow.export(root, work, {}, number, {'token': f'export-{number}', 'source': {'sha256': 'a' * 64}})
    a, b = [workflow.exported(root, work, n) for n in (1, 2)]
    assert len(calls) == 2 and a['compute_owner'] != b['compute_owner']
    assert a['csv_sha256'] == b['csv_sha256']
    assert not os.path.samefile(work / a['csv'], work / b['csv'])
    assert not (work / 'readiness.json').exists()


def test_replay_refuses_member_difference_even_when_csv_matches(export_fixture):
    root, work, values, calls = export_fixture
    claim = {'token': 'first', 'source': {'sha256': 'a' * 64}}
    workflow.export(root, work, {}, 1, claim)
    values['members'][0, 0, 0] += .001
    with pytest.raises(ValueError, match='predictions differ'):
        workflow.export(root, work, {}, 2, {'token': 'second', 'source': {'sha256': 'a' * 64}})
    assert not (work / 'exports/2.json').exists()


def test_export_refuses_before_inference_if_any_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'require_gates', lambda *_: (_ for _ in ()).throw(ValueError('gate failed')))
    with patch.object(assets, 'views') as inference:
        with pytest.raises(ValueError, match='gate failed'): workflow.export(tmp_path, tmp_path / 'work', {}, 1, {})
        inference.assert_not_called()
    assert not (tmp_path / 'work').exists()
