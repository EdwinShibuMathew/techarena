"""From-raw provenance, chronology and exact technical-resume contracts."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'experiments/v7.4-rebuild-v1'
sys.path[:0] = [str(DIRECTORY), str(ROOT / 'tools'), str(ROOT / 'legacy')]
import recipe
import data as dataset
from ml_workflow.contracts import digest


def raw_fixture(root):
    directory = root / 'data/raw'; directory.mkdir(parents=True)
    for name, times in [('train', [recipe.JAN20, recipe.JAN20 + 1, recipe.JAN20 + recipe.DAY]),
                        ('evaluation', [recipe.FULL])]:
        records = []
        for e, timestamp in enumerate(times):
            for j in range(10):
                row = dict(rowid=f'{name}-{e}-{j}', event_id=f'{e}', user_id='u' + str(e // 2),
                           item_id=f'item-{j:02}', timestamp=timestamp,
                           time_bucket=timestamp // 21600 % 4, day_type=int((timestamp // 86400 + 3) % 7 >= 5),
                           history_item_ids=json.dumps(['item-01', 'item-01']),
                           history_timestamps=json.dumps([timestamp - 60, timestamp - 30]))
                if name == 'train': row['clicked'] = int(j == 2)
                records.append(row)
        stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
        with zipfile.ZipFile(directory / (name + '.zip'), 'w') as z:
            z.writestr(name + '.csv', stream.getvalue())
    return {'train': 3, 'evaluation': 1}


def test_streaming_cache_and_causal_multiset(tmp_path):
    destination = tmp_path / 'cache'
    metadata = dataset.prepare(tmp_path, destination, raw_fixture(tmp_path))
    assert metadata['n_users'] == 2 and metadata['n_items'] == 10
    raw = dataset.load(destination, 'train')
    assert raw['rowids'][2, 9] == 'train-2-9'
    examples = dataset.prefix_examples(raw, recipe.JAN20 + recipe.DAY, 2, 10)
    assert examples['items'].tolist() == [1, 1, 2, 2]
    assert examples['users'].tolist() == [0, 0, 0, 0]
    assert dataset.verify(destination) == metadata
    with (destination / 'train-i.npy').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='input changed'): dataset.verify(destination)


def test_checkpoint_resume_restores_optimizer_and_all_randomness(tmp_path):
    torch = pytest.importorskip('torch')
    import train
    torch.manual_seed(42)
    model = train.PreferenceModel(4, 12, 8, torch.ones(4, dtype=torch.bool))
    opt = torch.optim.Adam(model.parameters(), lr=.012)
    users, items = torch.arange(4), torch.arange(4)
    train.preference_epoch(model, opt, users, items, 12)
    train.save(tmp_path / 'last.pt', model, opt, {'source': 'a'}, 'preference', 1)
    train.preference_epoch(model, opt, users, items, 12)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored = train.PreferenceModel(4, 12, 8, torch.ones(4, dtype=torch.bool))
    other_opt = torch.optim.Adam(restored.parameters(), lr=.99)
    train.restore(tmp_path / 'last.pt', restored, other_opt, {'source': 'a'})
    train.preference_epoch(restored, other_opt, users, items, 12)
    for key, value in restored.state_dict().items(): assert torch.equal(value, expected[key])
    with pytest.raises(ValueError, match='provenance'):
        train.restore(tmp_path / 'last.pt', restored, other_opt, {'source': 'changed'})


def test_synthetic_training_completion_is_immutable(tmp_path):
    torch = pytest.importorskip('torch')
    import train
    cache = tmp_path / 'cache'; folder = tmp_path / 'models'
    dataset.prepare(tmp_path, cache, raw_fixture(tmp_path))
    train.fit(folder, cache, recipe.JAN20 + recipe.DAY, 'dim8_seed42', 'test', 'cpu', synthetic=True)
    before = digest(folder / 'preference.pt'), digest(folder / 'event.pt')
    train.fit(folder, cache, recipe.JAN20 + recipe.DAY, 'dim8_seed42', 'test', 'cpu', synthetic=True)
    assert before == (digest(folder / 'preference.pt'), digest(folder / 'event.pt'))
    saved = torch.load(folder / 'event.pt', weights_only=True)
    assert saved['purpose'] == 'synthetic_validation'
    assert saved['preference_source_sha256'] == before[0]
    with pytest.raises(ValueError, match='Completed checkpoint differs'):
        train.fit(folder, cache, recipe.JAN20 + recipe.DAY, 'dim8_seed42', 'changed', 'cpu', synthetic=True)


def test_rebuild_schema_does_not_weaken_historical_contract():
    import study
    from ml_workflow.contracts import ExperimentSpec
    assert study.protocol()['schema_version'] == 5
    with pytest.raises(ValueError, match='Unsupported specification version'):
        ExperimentSpec.parse(study.protocol())


def test_cache_requires_identical_generation_and_output(tmp_path):
    import study
    path = tmp_path / 'scores.npz'
    result = study.bundle(tmp_path, path, {'generation': 'new'}, lambda: {'p': np.ones(3)})
    assert result['p'].tolist() == [1., 1., 1.]
    with pytest.raises(ValueError, match='provenance'):
        study.bundle(tmp_path, path, {'generation': 'old'}, lambda: {})
    with path.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='provenance'):
        study.bundle(tmp_path, path, {'generation': 'new'}, lambda: {})


def test_reconstruction_accounting_rejects_unfinished_or_reset_ledger(tmp_path):
    import study
    from ml_workflow.storage import write_new
    from consolidate import write_json
    write_new(tmp_path / 'genesis.json', {'recipe': recipe.RECIPE})
    ledger = {'total_seconds': 43200, 'carry_identity': digest(tmp_path / 'genesis.json'),
              'used_seconds': 12., 'attempts': [{'status': 'complete', 'seconds': 12.}]}
    write_json(tmp_path / 'compute-ledger.json', ledger)
    assert study.accounting(tmp_path) == ledger
    ledger['attempts'][0]['status'] = 'running'
    write_json(tmp_path / 'compute-ledger.json', ledger)
    with pytest.raises(ValueError, match='accounting'): study.accounting(tmp_path)
    ledger['attempts'][0]['status'] = 'complete'; ledger['total_seconds'] = 14400
    write_json(tmp_path / 'compute-ledger.json', ledger)
    with pytest.raises(ValueError, match='accounting'): study.accounting(tmp_path)


def test_promotion_requires_both_comparators_and_winner(tmp_path):
    import study
    def metric(delta=.001, lower=.0001):
        return {'baseline_auc': .8, 'candidate_auc': .8 + delta, 'auc_diff': delta,
                'ci_lower': lower, 'ci_upper': .002, 'resamples': 1000,
                'late': {'events': 1, 'baseline_auc': .8, 'candidate_auc': .8 + delta, 'auc_diff': delta}}
    development = {'winner': 'history_24h', 'passed': True, 'results': {}}
    for name in recipe.RECIPE['candidate_order']:
        for case in ('jan22_long', 'jan24_forward'):
            development['results'][name + '/' + case] = {**metric(), 'v7_comparison': metric()}
    screen = {'winner': 'history_24h', 'passed': True,
              'results': {case: {**metric(), 'v7_comparison': metric()} for case in ('primary', 'secondary')}}
    stress = {'winner': 'history_24h', 'passed': True,
              'results': {'five_day': {**metric(), 'v7_comparison': metric()}}}
    reports = dict(develop=development, screen=screen, stress=stress)
    with patch.object(study, 'checked', side_effect=lambda work, stage: reports[stage]):
        assert all(all(group.values()) for group in study.verify_gates(tmp_path)[1].values())
        screen['results']['primary']['v7_comparison']['ci_lower'] = 0.
        with pytest.raises(ValueError, match='eight'): study.publish(tmp_path, tmp_path)
    assert not (tmp_path / 'configs/v7_4.json').exists()
    assert not (tmp_path / 'readiness.json').exists()


def test_fixed_combiner_and_user_crossfit(tmp_path):
    pytest.importorskip('catboost')
    import study
    from ml_workflow.storage import write_new, save_arrays
    write_new(tmp_path / 'genesis.json', {'runtime': 'test'})
    write_new(tmp_path / 'scientific-source.json', {'source': 'test'})
    n = 25; rng = np.random.default_rng(42)
    raw = {'u': np.arange(n), 'y': np.zeros((n, 10), dtype=np.int8)}; raw['y'][:, 0] = 1
    ix = np.arange(n); X = rng.normal(size=(n * 10, 37)).astype(np.float32)
    cutoff = recipe.JAN20 + recipe.DAY
    path = tmp_path / 'features' / (study.feature_key(cutoff, cutoff, cutoff + recipe.DAY, 'dim8_seed42') + '.npz')
    save_arrays(path, X=X, feature_names=np.array(['f' + str(i) for i in range(37)]), indices=ix)
    write_new(path.with_suffix('.receipt.json'), {'sha256': digest(path)})
    study.fit_combiner(tmp_path, raw, cutoff, 'dim8_seed42', crossfit=True)
    output = tmp_path / 'oof' / str(cutoff) / 'dim8_seed42.npz'
    with np.load(output) as saved:
        assert np.isfinite(saved['scores']).all() and saved['scores'].shape == (n, 10)
        assert set(saved['folds']) == set(range(5))
    before = digest(output)
    study.fit_combiner(tmp_path, raw, cutoff, 'dim8_seed42', crossfit=True)
    assert digest(output) == before


def test_task_success_follows_durable_budget_accounting(tmp_path):
    import run
    from ml_workflow.storage import write_new, read
    from ml_workflow import budget as budgets
    write_new(tmp_path / 'genesis.json', {'generation': 'test'})
    source = {'sha256': 'a' * 64, 'files': {}}
    snapshot = tmp_path / 'snapshot'; snapshot.mkdir()
    real_write = budgets.write_json
    count = [0]
    def fail_final(path, value):
        count[0] += 1
        if count[0] == 2: raise OSError('durable accounting failed')
        real_write(path, value)
    def worker_stub(self, command, cwd):
        claim = read(next((tmp_path / 'attempts').glob('*.claim.json')))
        write_new(tmp_path / 'attempts' / (claim['token'] + '.validated.json'),
                  {'genesis_sha256': digest(tmp_path / 'genesis.json'), 'source_sha256': source['sha256'], 'files': {}})
        return 0
    with patch.object(run, 'initialize', return_value=(tmp_path, {'generation': 'test'})), \
         patch.object(run, 'freeze', return_value=(snapshot, source)), \
         patch.object(budgets.Budget, 'run', worker_stub), patch.object(budgets, 'write_json', fail_final):
        with pytest.raises(OSError, match='accounting failed'):
            run.launch(tmp_path, 'prepare', 'prepare', 20)
    assert not (tmp_path / 'tasks/prepare.success.json').exists()
    ledger = read(tmp_path / 'compute-ledger.json')
    assert ledger['used_seconds'] == 20 and ledger['attempts'][0]['status'] == 'running'


def test_stale_reservation_is_charged_in_full_on_recovery(tmp_path):
    import run
    from ml_workflow.storage import read, write_new
    path = tmp_path / 'ledger.json'
    write_new(path, {'total_seconds': 43200, 'carry_identity': 'generation', 'used_seconds': 100,
                     'attempts': [{'owner': 'old', 'status': 'running', 'reserved_seconds': 100}]})
    with run.ReconstructionBudget(path, 'new', 20, 43200, carry_identity='generation'):
        pass
    ledger = read(path)
    assert ledger['attempts'][0]['status'] == 'abandoned_reserved'
    assert ledger['attempts'][0]['seconds'] == 100 and ledger['used_seconds'] >= 100


def test_validated_recovery_requires_same_source_task_and_artifacts(tmp_path):
    import run
    from ml_workflow.storage import write_new
    source = {'sha256': 'source'}
    claim = dict(task='score-history_24h-primary', action='score', arguments={'candidate': 'history_24h'},
                 source=source, genesis_sha256='generation')
    artifact = tmp_path / 'result.json'; write_new(artifact, {'passed': False})
    write_new(tmp_path / 'attempts/old.claim.json', claim)
    write_new(tmp_path / 'attempts/old.validated.json', {'genesis_sha256': 'generation', 'source_sha256': 'source',
                                                       'files': {'result.json': digest(artifact)}})
    state = {'attempts': [{'owner': 'generation/old'}, {'owner': 'generation/new'}]}
    assert run.recover_validation(tmp_path, claim, state)['technical_recovery_from'] == 'old'
    assert run.recover_validation(tmp_path, {**claim, 'arguments': {'candidate': 'history_72h'}}, state) is None
    artifact.write_text('{}')
    with pytest.raises(ValueError, match='Recovery artifact'):
        run.recover_validation(tmp_path, claim, state)


def test_real_graph_scoring_binds_predictions_and_causal_state(tmp_path):
    pytest.importorskip('scipy')
    import study
    import run
    from ml_workflow.storage import write_new, read
    from ml_workflow.contracts import first_indices
    write_new(tmp_path / 'genesis.json', {'runtime': 'test'})
    write_new(tmp_path / 'scientific-source.json', {'source': 'test'})
    write_new(tmp_path / 'data/metadata.json', {'synthetic': True})
    n = 168; times = recipe.JAN20 + np.arange(n) * 3600
    history = np.full((n, 20), -1); history[:, :3] = [0, 1, 0]
    ht = np.zeros((n, 20), dtype=np.int64); ht[:, :3] = times[:, None] - [90, 60, 30]
    raw = dict(u=np.arange(n), t=times, i=np.tile(np.arange(10), (n, 1)), h=history, ht=ht,
               tb=times // 21600 % 4, dt=((times // recipe.DAY + 3) % 7 >= 5).astype(np.int8),
               y=np.zeros((n, 10), dtype=np.int8), rowids=np.array([[f'{e}-{j}' for j in range(10)] for e in range(n)]))
    raw['y'][:, 9] = 1; raw['y'][::4] = 0
    meta, target, end = recipe.CASES['jan22_long']
    fitting = first_indices(raw['u'], times, meta, meta + recipe.DAY)
    valid = first_indices(raw['u'], times, target, end)
    rng = np.random.default_rng(2026)
    oof, baseline = rng.uniform(.01, .1, (len(fitting), 10)), rng.uniform(.01, .1, (len(valid), 10))
    with patch.object(study, 'references', return_value=(fitting, valid, oof, baseline, {})):
        study.score(tmp_path, raw, 'jan22_long', 'history_24h')
    row = read(tmp_path / 'results/history_24h-jan22_long.json')
    assert row['calibrator_metadata']['maximum_label_timestamp'] < target
    assert row['baseline_lineage'] == 'rebuilt_v6_v7_recipe'
    with np.load(tmp_path / 'predictions/history_24h-jan22_long.npz') as saved:
        np.testing.assert_allclose(saved['candidate'].sum(1), saved['v7'].sum(1), atol=1e-14, rtol=0)
    with patch.object(run, 'completed', return_value=True):
        study.validate_row(tmp_path, row)
        graph = tmp_path / 'graphs' / (str(meta) + '.npz')
        with graph.open('ab') as stream: stream.write(b'changed')
        with pytest.raises(ValueError, match='dependency changed'): study.validate_row(tmp_path, row)
