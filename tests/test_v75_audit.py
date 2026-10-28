"""Saved-score diagnostics and future promotion evidence; no research fitting."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments/v7.5-audit'), str(ROOT / 'tools'), str(ROOT / 'legacy'),
                str(ROOT / 'experiments/v7.5-history-age-v1'), str(ROOT / 'experiments/v7.4-rebuild-v1')]
import audit
import audit_metrics as metrics
import successor
from ml_workflow import promotion
from ml_workflow.contracts import canonical, digest, JAN20, DAY
from ml_workflow.storage import write_new


def fixture_scores():
    y = np.zeros((4, 10), dtype=np.int8); y[:3, 0] = 1
    p = np.array([[.3, .3] + [.01] * 8, [.2] + [.04] * 9, [.1] + [.01] * 9, [.15] * 10])
    return y, p


def test_auc_decomposition_includes_cross_event_negatives_and_ties():
    y, p = fixture_scores(); report = metrics.describe(y, p)
    positive = p[y == 1]; negative = p[y == 0]
    brute = ((positive[:, None] > negative) + .5 * (positive[:, None] == negative)).mean()
    assert report['auc'] == pytest.approx(brute)
    assert report['reconstructed_global_auc'] == pytest.approx(brute)
    assert report['within_event_auc'] == pytest.approx((8.5 / 9 + 1 + 1) / 3)
    assert report['event_score_sum_gt1_fraction'] == .25


def test_decomposition_handles_one_class_and_all_clicked():
    y, p = fixture_scores()
    assert metrics.describe(y * 0, p)['auc'] is None
    report = metrics.describe(y[:3], p[:3])
    assert report['negative_pair_weights']['no_click'] == 0
    assert report['auc'] == pytest.approx(report['reconstructed_global_auc'])
    with pytest.raises(ValueError): metrics.describe(y, p * 10)


def test_local_ranking_can_improve_while_global_auc_falls():
    y = np.zeros((2, 10)); y[0, 0] = 1
    before = np.array([[.7, .8] + [.1] * 8, [.01] * 10])
    after = np.array([[.2] + [.1] * 9, [.3] * 10])
    a, b = metrics.describe(y, before), metrics.describe(y, after)
    assert b['within_event_auc'] > a['within_event_auc']
    assert b['auc'] < a['auc']


def test_component_swaps_are_diagnostic_and_report_out_of_range():
    y, p = fixture_scores()
    np.testing.assert_allclose(metrics.component_swap(p, p), p)
    a = np.zeros((4, 10)); a[:, 0] = 1
    b = np.full((4, 10), .2)
    result = metrics.component_report(y, a, b, p, {'all': np.ones(4, bool)})
    assert result['diagnostic_only'] and not result['promotion_supported']
    assert result['variants']['A_allocation_V7_mass']['out_of_range_rows'] == 4
    with pytest.raises(ValueError): metrics.component_swap(np.zeros_like(p), p)
    assert not metrics.component_swap(np.zeros_like(p), np.zeros_like(p)).any()


def test_range_coverage_and_importance_separate_confounders():
    names = ['f' + str(i) for i in range(47)]
    X = np.ones((4, 47)); X[:, 45:] = 2
    coverage = metrics.range_coverage(np.zeros(47), np.ones(47), X, names)
    assert coverage['f45']['outside_fitting_range_fraction'] == 1
    assert coverage['f44']['outside_fitting_range_fraction'] == 0
    values = np.ones((5, 47))
    groups = metrics.group_importance(values, names)['groups']
    assert groups == {'original37': 37, 'graph7': 7, 'age': 1, 'prefix_size': 2}


def comparisons():
    row = dict(baseline_auc=.8, candidate_auc=.801, auc_diff=.001, ci_lower=.0001, ci_upper=.002, resamples=1000)
    return {g: {a: deepcopy(row) for a in ('V6', 'V7')} for g in ('primary', 'secondary', 'five_day', 'late')}


def test_all_eight_numerical_gates_and_no_eligibility_shortcut():
    good = comparisons(); checked = promotion.metric_gates(good)
    assert not checked['blockers'] and 'eligible' not in checked
    for name in good:
        for anchor in good[name]:
            bad = deepcopy(good)
            if name in ('primary', 'five_day'): bad[name][anchor]['ci_lower'] = 0
            else: bad[name][anchor].update(candidate_auc=.799, auc_diff=-.001)
            result = promotion.metric_gates(bad)
            assert not result['gates'][anchor][name] and len(result['blockers']) == 1


def test_missing_nan_and_insufficient_bootstrap_evidence_fail_closed():
    assert len(promotion.metric_gates({})['blockers']) == 8
    for change in ({'resamples': 989}, {'ci_lower': float('nan')}, {'candidate_auc': .9}):
        bad = comparisons(); bad['primary']['V7'].update(change)
        assert not promotion.metric_gates(bad)['gates']['V7']['primary']


@pytest.mark.parametrize('evidence', [None, [], {}, {'schema_version': True},
    {'schema_version': 1, 'model_role': 'rebuilt_baseline'},
    {'schema_version': 1, 'model_role': 'candidate', 'purpose': 'synthetic'}])
def test_malformed_synthetic_and_fallback_never_promote(tmp_path, evidence):
    result = promotion.verify_promotion(tmp_path, evidence)
    assert result['eligible'] is False and result['blockers']
    assert not any(g for anchor in result['gates'].values() for g in anchor.values())


def minimal_evidence():
    return dict(schema_version=1, model_role='candidate', purpose='research',
                baseline_lineage='rebuilt_v6_v7_recipe', bootstrap=promotion.BOOTSTRAP)


def test_promotion_rejects_changed_bytes_and_path_escape(tmp_path):
    path = tmp_path / 'file.json'; write_new(path, {'v': 1})
    evidence = minimal_evidence(); evidence['artifacts'] = {'file.json': '0' * 64}
    assert 'Artifact changed' in promotion.verify_promotion(tmp_path, evidence)['blockers'][0]
    evidence['artifacts'] = {'../file.json': digest(path)}
    assert 'Unsafe relative' in promotion.verify_promotion(tmp_path, evidence)['blockers'][0]
    link = tmp_path / 'link.json'; link.symlink_to(path)
    evidence['artifacts'] = {'link.json': digest(path)}
    assert 'Symlink' in promotion.verify_promotion(tmp_path, evidence)['blockers'][0]


def test_frozen_successor_is_design_only_and_detects_changes():
    value = successor.design(); result = successor.validate_design(value)
    assert result['valid'] and not result['training_executed']
    assert result['design_sha256'] == canonical(value)
    assert not value['promotion_supported'] and len(value['features']) == 13
    assert max(b[1] for b in value['inner_selection']['validation_blocks']) <= JAN20 + 2 * DAY
    for field in ('training_authorized_by_this_audit', 'promotion_supported'):
        changed = deepcopy(value); changed[field] = True
        with pytest.raises(ValueError): successor.validate_design(changed)


def test_forecast_schedule_enforces_historical_fit_and_prediction_prefixes():
    prefixes = [JAN20 + h * 3600 for h in (6, 12, 24, 48)]
    for hour in (12, 18, 24, 30, 48):
        start = JAN20 + hour * 3600; plan = successor.forecast_schedule(start, prefixes)
        assert plan['combiner_prefix'] <= start - 6 * 3600
        assert plan['fit_end'] == plan['target_start'] == start
        assert plan['prediction_prefix'] <= start
        assert plan['target_end'] == start + 6 * 3600
    with pytest.raises(ValueError): successor.forecast_schedule(JAN20, prefixes)


def test_audit_worker_without_claim_cannot_consume_or_publish(tmp_path):
    with pytest.raises(FileNotFoundError): audit.worker(tmp_path, 'no-lease')
    assert not (tmp_path / 'artifacts/rebuild/compute-ledger.json').exists()
    assert not (tmp_path / 'docs/results/v7.5-audit/audit.json').exists()


def test_publication_requires_settled_receipt_and_preserves_inputs(tmp_path):
    path = tmp_path / 'old-model.pt'; path.write_bytes(b'original')
    protected = {'old-model.pt': digest(path)}
    with patch.object(audit, 'verify', side_effect=ValueError('unsettled accounting')):
        with pytest.raises(ValueError): audit.publish(tmp_path)
    assert not (tmp_path / 'docs').exists()
    audit.unchanged(tmp_path, protected)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Protected'): audit.unchanged(tmp_path, protected)


@pytest.fixture
def evidence_chain(tmp_path, monkeypatch):
    """Small *evidence-chain* fixture, not proof of a production model's gates.

    Substitute corpus loading/cohort constants only; execute the real bootstrap,
    artifact, source, development, accounting, and independent CSV checks.
    """
    import json
    import shutil
    n = 12
    y = np.zeros((n, 10), dtype=np.int8); y[:, 0] = 1
    arrays = dict(u=np.arange(n), t=JAN20 + np.arange(n), y=y,
                  rowids=np.array([[f'r{i}-{j}' for j in range(10)] for i in range(n)]))
    monkeypatch.setattr(promotion, 'load_official_arrays', lambda *_: arrays)
    monkeypatch.setattr(promotion, 'COHORTS', {g: (JAN20, JAN20 + DAY, n) for g in ('primary', 'secondary', 'five_day')})
    monkeypatch.setattr(promotion, 'late_mask', lambda t, begin: np.ones(len(t), bool))
    monkeypatch.setattr(promotion, 'EXPECTED_ROWS', n * 10)
    def put(name, value):
        write_new(tmp_path / name, value)
        return name
    validator = 'legacy/submission_export.py'
    dest = tmp_path / 'snapshot/source' / validator; dest.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / validator, dest)
    archive = tmp_path / 'snapshot/source.tar.gz'; archive.write_bytes(b'fixture-source-archive')
    source = {'commit': 'a' * 40, 'sha256': digest(archive), 'files': {validator: digest(dest)}}
    put('source.json', source)
    put('engineering.json', dict(kind='source_readiness', git_commit=source['commit'],
         checks={k: 'success' for k in ('cpu_tests', 'model_contracts', 'source_integrity', 'source_archive')}))
    put('genesis.json', {'total_seconds': 43200})
    ledger = {'total_seconds': 43200, 'used_seconds': 2., 'carry_identity': digest(tmp_path / 'genesis.json'),
              'attempts': [{'owner': 'generation/export1', 'status': 'complete', 'seconds': 1.},
                           {'owner': 'generation/export2', 'status': 'complete', 'seconds': 1.}]}
    put('ledger.json', ledger); put('metadata.json', {'fixture': True})
    model = tmp_path / 'model.pt'; model.write_bytes(b'frozen-fixture-model')
    references = {}; predictions = {}; identities = {'V6': '6' * 64, 'V7': '7' * 64}
    baseline = np.full(y.shape, .1); candidate = np.full(y.shape, .05); candidate[:, 0] = .2
    for group in ('primary', 'secondary', 'five_day'):
        path = tmp_path / (group + '.npz')
        np.savez(path, indices=np.arange(n), rowids=arrays['rowids'], labels=y,
                 candidate=candidate, baseline=baseline, v7=baseline, V6_identity=identities['V6'], V7_identity=identities['V7'])
        predictions[group] = path.name; references[group] = path.name
    row = dict(baseline_auc=.5, candidate_auc=.6, auc_diff=.1)
    row['late'] = dict(baseline_auc=.5, candidate_auc=.6, auc_diff=.1, events=1)
    row['v7_comparison'] = deepcopy(row)
    put('development.json', dict(winner='A', passed=True, results={'A/old': row}))
    manifest = dict(candidate='A', baseline_lineage='rebuilt_v6_v7_recipe', comparators=identities,
         models={'model.pt': digest(model)}, source_sha256=source['sha256'], data_sha256=digest(tmp_path / 'metadata.json'),
         protocol=dict(release_policy='v6_v7_four_gate', candidate_order=['A'],
                       development=[dict(name='old', target_cutoff=JAN20, target_end=JAN20 + 3 * DAY)]),
         references=references, owner_prefix='generation/')
    put('manifest.json', manifest)
    csv = 'rowid,predicted_probability\n' + ''.join(f'{rid},0.2\n' for rid in arrays['rowids'].ravel())
    (tmp_path / 'example.csv').write_text(csv)
    exports = []
    for number in (1, 2):
        name = f'export{number}.csv'; (tmp_path / name).write_text(csv)
        receipt = dict(candidate='A', fresh_neural_features=True, fresh_graph=True,
             source_sha256=source['sha256'], manifest_sha256=digest(tmp_path / 'manifest.json'),
             model_sha256=canonical(manifest['models']), csv_sha256=digest(tmp_path / name), compute_owner=f'generation/export{number}')
        put(f'export{number}.json', receipt); exports.append(dict(csv=name, receipt=f'export{number}.json'))
    evidence = {**minimal_evidence(), 'candidate': 'A', 'development': dict(winner='A', passed=True, report='development.json'),
         'comparators': identities, 'run_manifest': 'manifest.json', 'source_record': 'source.json', 'source_directory': 'snapshot',
         'engineering_receipt': 'engineering.json', 'accounting_record': 'ledger.json', 'budget_genesis': 'genesis.json',
         'data_metadata': 'metadata.json', 'predictions': predictions, 'example_submission': 'example.csv', 'exports': exports}
    def refresh():
        evidence['artifacts'] = {p.relative_to(tmp_path).as_posix(): digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    refresh()
    def rewrite(name, transform):
        path = tmp_path / name; value = json.loads(path.read_text()); transform(value)
        path.write_text(json.dumps(value)); refresh()
    return tmp_path, evidence, refresh, rewrite


def test_complete_fixture_evidence_recomputes_all_eight_gates(evidence_chain):
    root, evidence, _, _ = evidence_chain
    result = promotion.verify_promotion(root, evidence)
    assert result['eligible'], result['blockers']
    assert all(v for anchor in result['gates'].values() for v in anchor.values())
    assert result['comparisons']['primary']['V7']['resamples'] == 1000


@pytest.mark.parametrize('kind', ['running', 'budget_reset', 'source', 'comparator', 'same_owner', 'alias', 'development', 'model', 'csv_order'])
def test_evidence_chain_refuses_artifact_and_provenance_failures(evidence_chain, kind):
    root, evidence, refresh, rewrite = evidence_chain
    if kind == 'running': rewrite('ledger.json', lambda v: v['attempts'][0].update(status='running'))
    elif kind == 'budget_reset': rewrite('ledger.json', lambda v: v.update(used_seconds=0))
    elif kind == 'source': (root / 'snapshot/source/legacy/submission_export.py').write_bytes(b'changed'); refresh()
    elif kind == 'comparator': evidence['comparators'] = {'V6': '0' * 64, 'V7': '7' * 64}
    elif kind == 'same_owner': rewrite('export2.json', lambda v: v.update(compute_owner='generation/export1'))
    elif kind == 'alias':
        (root / 'export2.csv').unlink(); (root / 'export2.csv').hardlink_to(root / 'export1.csv'); refresh()
    elif kind == 'development': rewrite('development.json', lambda v: v.update(passed=False))
    elif kind == 'model': (root / 'model.pt').write_bytes(b'changed'); refresh()
    elif kind == 'csv_order':
        path = root / 'export2.csv'; lines = path.read_text().splitlines(); lines[1], lines[2] = lines[2], lines[1]
        path.write_text('\n'.join(lines) + '\n')
        rewrite('export2.json', lambda v: v.update(csv_sha256=digest(path)))
    result = promotion.verify_promotion(root, evidence)
    assert not result['eligible'] and result['blockers']
