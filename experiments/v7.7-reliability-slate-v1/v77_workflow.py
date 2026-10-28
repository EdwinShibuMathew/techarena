"""Fixed four-ablation campaign and artifact-backed, restartable release."""
from dataclasses import asdict
import os
from pathlib import Path
import numpy as np
from ml_workflow.contracts import canonical, digest, first_indices, inside, require, DAY, JAN20
from ml_workflow.storage import read, write_new, save_arrays
from ml_workflow import policy, promotion
from ml_workflow.historical import PROFILE, FLOORS, floor_gates
from experiment_runner import compute_roc_auc, PairedBootstrapEvaluator
from recipe import MEMBERS, COMBINER, RECIPE, CASES, FULL
import data as dataset
import v77_assets as assets
import v77_features as features
import v77_model as model
from v77_protocol import SPEC, CANDIDATES, METAS, prefixes
# Import by unique module name, including when the CLI is executed as __main__.
import v77_runner as runner


def path(work, relative): return inside(work, relative)
def relative(work, value): return value.relative_to(work).as_posix()
def arrays(filename):
    with np.load(filename, allow_pickle=False) as saved: return {k: saved[k] for k in saved.files}


def result(root, work, action, arguments=None):
    value = runner.completed(root, work, action, arguments)
    require(value is not None, 'Missing settled dependency: ' + action)
    return value


def point(y, before, after):
    a, b = compute_roc_auc(y, before), compute_roc_auc(y, after)
    return {'baseline_auc': a, 'candidate_auc': b, 'auc_diff': b - a}


def head_role(work, meta, candidate, dependencies):
    return {'purpose': 'research', 'metadata_day': meta, 'fit_end_exclusive': meta + DAY,
            'candidate': candidate, 'manifest_sha256': digest(work / 'manifest.json'), 'dependencies': dependencies}


def load_head(root, work, meta, candidate):
    saved = result(root, work, 'head', {'meta': meta, 'candidate': candidate})
    require(saved['metadata'] == head_role(work, meta, candidate, saved['metadata']['dependencies']), 'Head deployment role differs')
    return model.load(path(work, saved['model']), saved['metadata'])


def view(root, work, case):
    saved = result(root, work, 'view', {'case': case})
    return arrays(path(work, saved['view']))


def run_manifest(root, work):
    record = result(root, work, 'freeze')
    filename = path(work, record['manifest']); manifest = read(filename)
    for name, expected in manifest['models'].items(): require(digest(inside(root, name)) == expected, 'Frozen inference model changed')
    return filename, manifest


def preflight(root, work, directory, claim):
    import torch
    manifest = read(work / 'manifest.json')
    dataset.verify(root / 'artifacts/rebuild/data')
    require(torch.cuda.is_available(), 'Production CUDA unavailable')
    for name, value in {**manifest['protected_inputs'], **manifest['foundation_files']}.items():
        require(digest(inside(root, name)) == value, 'Protected asset changed: ' + name)
    engineering = inside(root, manifest['engineering_receipt'])
    receipt = read(engineering)
    require(digest(engineering) == manifest['engineering_sha256'] and receipt['kind'] == 'source_readiness'
            and receipt['git_commit'] == claim['source']['commit'] and all(receipt['checks'].get(k) == 'success'
                for k in ('cpu_tests', 'model_contracts', 'source_integrity', 'source_archive')), 'Engineering source checks differ')
    floors = {'schema_version': 1, 'profile': PROFILE, 'floors': {}}
    for name, (filename, keys, count, value) in FLOORS.items():
        require(digest(root / filename) == claim['source']['files'][filename], 'Historical evidence source changed')
        saved = read(root / filename)
        for k in keys: saved = saved[k]
        require(saved['candidate_auc'] == value, 'Historical point threshold differs')
        floors['floors'][name] = {'path': filename, 'keys': keys, 'events': count, 'auc': value, 'sha256': digest(root / filename)}
    write_new(directory / 'historical-floors.json', floors)
    return {'passed': True, 'source_sha256': claim['source']['sha256'], 'floors': relative(work, directory / 'historical-floors.json'),
            'device': torch.cuda.get_device_name(), 'prior_attempts': manifest['prior_attempts']}


def teacher(root, work, directory, raw, args):
    meta, prefix, member = args['meta'], args['prefix'], args['member']
    require(meta in METAS and prefix in prefixes(meta) and member in MEMBERS, 'Undeclared OOF role')
    data = assets.matrix(root, read(work / 'manifest.json'), raw, prefix, meta, meta + DAY, member)
    assets.oof(data, raw, directory)
    return {'oof': relative(work, directory / 'oof.npz'), 'receipt': relative(work, directory / 'oof.json'), 'role': args}


def training_view(root, work, directory, raw, args):
    meta, prefix = args['meta'], args['prefix']; values = []; identities = {}
    require(prefix in prefixes(meta), 'Undeclared fitting prefix')
    ix = first_indices(raw['u'], raw['t'], meta, meta + DAY)
    for member in MEMBERS:
        saved = result(root, work, 'teacher', dict(meta=meta, prefix=prefix, member=member))
        data = arrays(path(work, saved['oof']))
        require(saved['role'] == dict(meta=meta, prefix=prefix, member=member)
                and np.array_equal(data['indices'], ix) and np.array_equal(data['folds'], __import__('v77_protocol').folds(raw['u'][ix])), 'OOF alignment differs')
        values.append(data['probabilities']); identities[saved['oof']] = digest(path(work, saved['oof']))
    selected = features.cohort(raw, prefix, meta)
    positions = np.searchsorted(ix, selected)
    require(np.array_equal(ix[positions], selected), 'First-return views must be nested in metadata first-user cohort')
    data = assets.make_view(root, read(work / 'manifest.json'), raw, raw, selected, prefix, np.stack(values)[:, positions])
    strata = np.column_stack([np.full(len(selected), (meta - prefix) // DAY), (raw['t'][selected] - meta) // (6 * 3600)])
    save_arrays(directory / 'training.npz', **data, strata=strata)
    return {'data': relative(work, directory / 'training.npz'), 'teacher_predictions': identities, 'role': args}


def fit_head(root, work, directory, raw, args):
    meta, candidate = args['meta'], args['candidate']; datasets = []; dependencies = {}
    require(meta in METAS and candidate in CANDIDATES, 'Undeclared head role')
    for prefix in prefixes(meta):
        saved = result(root, work, 'training-view', dict(meta=meta, prefix=prefix))
        require(saved['role'] == dict(meta=meta, prefix=prefix), 'Training view role differs')
        datasets.append(arrays(path(work, saved['data']))); dependencies[saved['data']] = digest(path(work, saved['data']))
    data = assets.pool(datasets, raw); metadata = head_role(work, meta, candidate, dependencies)
    state = model.fit(data, data['labels'], data['weights'], candidate, metadata, directory / 'head.json')
    return {'model': relative(work, directory / 'head.json'), 'metadata': metadata,
            'unique_users': data['unique_users'], 'coverage': data['coverage'],
            'statistics': {k: state[k] for k in ('fit_views', 'unique_events', 'effective_views', 'effective_events', 'objective', 'identity_objective', 'iterations')}}


def deploy(root, work, directory, raw, args):
    meta, member = args['meta'], args['member']; manifest = read(work / 'manifest.json')
    require(meta in METAS and member in MEMBERS, 'Undeclared deployment combiner role')
    ix = first_indices(raw['u'], raw['t'], meta, meta + DAY)
    old = root / 'artifacts/v7_5/baseline-combiners' / str(meta) / (member + '.cbm')
    filename = directory / 'combiner.cbm'; inherited = old.relative_to(root).as_posix()
    from ml_workflow.cache import arrays_identity
    if inherited in manifest['protected_inputs']:
        require(digest(old) == manifest['protected_inputs'][inherited], 'Borrowed combiner changed')
        identity = read(old.with_suffix('.receipt.json'))['identity']
        require(identity['parameters'] == COMBINER and identity['meta'] == meta and identity['member'] == member
                and identity['labels'] == arrays_identity({'y': raw['y'][ix]}), 'Borrowed combiner role differs')
        os.link(old, filename)
    else:
        data = assets.matrix(root, manifest, raw, meta, meta, meta + DAY, member)
        require(np.array_equal(data['indices'], ix), 'Deploy fitting cohort differs')
        fitted = assets.fit_teacher(data['X'], data['feature_names'], raw['y'][ix].ravel())
        fitted.save_model(str(filename)); inherited = None
    return {'model': relative(work, filename), 'role': args, 'parameters': COMBINER,
            'label_identity': arrays_identity({'y': raw['y'][ix]}), 'borrowed_from': inherited}


def prediction_view(root, work, directory, raw, args, evaluation=None):
    case = args['case']; meta, cutoff, end = CASES[case] if case != 'evaluation' else (JAN20 + 5 * DAY, FULL, FULL + 7 * DAY)
    manifest = read(work / 'manifest.json'); probabilities = []; dependencies = {}
    for member in MEMBERS:
        saved = result(root, work, 'deploy', dict(meta=meta, member=member))
        require(saved['role'] == dict(meta=meta, member=member) and saved['parameters'] == COMBINER, 'Deploy role differs')
        data = assets.matrix(root, manifest, raw, cutoff, cutoff, end, member, evaluation=evaluation, fresh=evaluation is not None)
        probabilities.append(assets.predict_teacher(path(work, saved['model']), data))
        dependencies[saved['model']] = digest(path(work, saved['model']))
    target = raw if evaluation is None else evaluation
    ix = first_indices(raw['u'], raw['t'], cutoff, end) if evaluation is None else np.arange(len(target['u']))
    values = assets.make_view(root, manifest, raw, target, ix, cutoff, np.stack(probabilities))
    save_arrays(directory / 'view.npz', **values, v7=values['anchor'])
    return {'view': relative(work, directory / 'view.npz'), 'role': args, 'dependencies': dependencies}


def choose(results): return policy.dual_winner(SPEC, results)


def selection(root, work, directory, raw):
    rows = {}
    for candidate in CANDIDATES:
        for name in ('jan22_long', 'jan24_forward'):
            data = view(root, work, name); meta, cutoff, _ = CASES[name]
            p = model.predict(data, load_head(root, work, meta, candidate)); ix = data['indices']; y = raw['y'][ix]
            row = point(y, data['baseline'], p); row['v7_comparison'] = point(y, data['anchor'], p)
            if name == 'jan22_long':
                age = (raw['t'][ix] - cutoff) / 3600; aged = (age >= 48) & (age < 72)
                require(aged.sum() > 0, 'Missing aged development cohort')
                row['late'] = dict(point(y[aged], data['baseline'][aged], p[aged]), events=int(aged.sum()))
                row['v7_comparison']['late'] = point(y[aged], data['anchor'][aged], p[aged])
            rows[candidate + '/' + name] = row
    winner = choose(rows)
    write_new(directory / 'development.json', {'passed': winner is not None, 'winner': winner, 'results': rows})
    return {'passed': winner is not None, 'winner': winner, 'report': relative(work, directory / 'development.json'), 'results': rows}


def freeze(root, work, directory, claim):
    selection = result(root, work, 'selection'); require(selection['passed'], 'No development winner')
    candidate = selection['winner']; original = read(work / 'manifest.json')
    baseline_models = dict(original['foundation_files']); models = {}; references = {}
    for meta in METAS:
        head = result(root, work, 'head', dict(meta=meta, candidate=candidate))
        filename = path(work, head['model']); model.load(filename, head['metadata'])
        models[filename.relative_to(root).as_posix()] = digest(filename)
        for member in MEMBERS:
            saved = result(root, work, 'deploy', dict(meta=meta, member=member)); filename = path(work, saved['model'])
            baseline_models[filename.relative_to(root).as_posix()] = digest(filename)
    for name in ('primary', 'secondary', 'five_day'):
        saved = result(root, work, 'view', dict(case=name))
        references[name] = path(work, saved['view']).relative_to(root).as_posix()
    models.update(baseline_models)
    baseline = {'data_sha256': original['data_sha256'], 'recipe': RECIPE, 'models': baseline_models}
    manifest = {'schema_version': 2, 'candidate': candidate, 'protocol': SPEC, 'models': models, 'references': references,
                'baseline_lineage': SPEC['baseline_lineage'], 'baseline_inputs': baseline,
                'comparators': {a: canonical({'anchor': a, 'baseline': baseline}) for a in ('V6', 'V7')},
                'source_sha256': claim['source']['sha256'], 'data_sha256': original['data_sha256'],
                'owner_prefix': read(root / 'artifacts/rebuild/genesis.json')['generation'] + '/v77-'}
    write_new(directory / 'run-manifest.json', manifest)
    return {'manifest': relative(work, directory / 'run-manifest.json'), 'candidate': candidate}


def predict_case(root, work, directory, raw, args):
    name = args['case']; _, manifest = run_manifest(root, work); data = view(root, work, name)
    meta = CASES[name][0]; p = model.predict(data, load_head(root, work, meta, manifest['candidate']))
    save_arrays(directory / 'predictions.npz', indices=data['indices'], rowids=data['rowids'], labels=raw['y'][data['indices']],
                candidate=p, baseline=data['baseline'], v7=data['anchor'],
                **{a + '_identity': np.asarray(identity) for a, identity in manifest['comparators'].items()})
    return {'predictions': relative(work, directory / 'predictions.npz'), 'case': name}


def gates(root, work, directory, raw, stage):
    _, manifest = run_manifest(root, work); comparisons = {}; evaluator = PairedBootstrapEvaluator()
    names = ('primary', 'secondary') if stage == 'screen' else ('five_day',)
    for name in names:
        saved = result(root, work, 'predict', dict(case=name)); data = arrays(path(work, saved['predictions']))
        ix = first_indices(raw['u'], raw['t'], CASES[name][1], CASES[name][2]); y = raw['y'][ix]
        require(len(ix) == promotion.COHORTS[name][2] and np.array_equal(data['indices'], ix)
                and np.array_equal(data['labels'], y) and np.array_equal(data['rowids'], raw['rowids'][ix]), 'Gate cohort differs')
        comparisons[name] = {}
        for anchor, field in (('V6', 'baseline'), ('V7', 'v7')):
            require(str(data[anchor + '_identity'].item()) == manifest['comparators'][anchor], 'Gate comparator differs')
            row = point(y, data[field], data['candidate'])
            if name != 'secondary': row.update(asdict(evaluator.evaluate(data['candidate'], data[field], y, raw['u'][ix])))
            comparisons[name][anchor] = row
            if name == 'five_day':
                late = promotion.late_mask(raw['t'][ix], CASES[name][1])
                comparisons.setdefault('late', {})[anchor] = point(y[late], data[field][late], data['candidate'][late])
    numerical = promotion.metric_gates(comparisons)
    applicable = ('primary', 'secondary') if stage == 'screen' else ('five_day', 'late')
    checks = {a: {g: numerical['gates'][a][g] for g in applicable} for a in ('V6', 'V7')}
    floors = floor_gates(comparisons, {g: FLOORS[g][3] for g in applicable})
    return {'passed': all(all(v.values()) for v in checks.values()) and all(floors.values()),
            'gates': checks, 'historical_floors': floors, 'comparisons': comparisons}


def require_gates(root, work):
    require(result(root, work, 'selection')['passed'] and all(result(root, work, s)['passed'] for s in ('screen', 'stress')), 'Twelve release checks have not passed')


def export(root, work, directory, raw, claim):
    from submission_export import export_submission, validate_csv
    from row_auc_final_utils import verify_csv_probabilities
    require_gates(root, work); manifest_path, manifest = run_manifest(root, work)
    evaluation = dataset.load(root / 'artifacts/rebuild/data', 'evaluation')
    require('y' not in evaluation and len(evaluation['u']) == 52939, 'Official evaluation corpus differs')
    saved = prediction_view(root, work, directory, raw, dict(case='evaluation'), evaluation)
    data = arrays(path(work, saved['view']))
    p = model.predict(data, load_head(root, work, JAN20 + 5 * DAY, manifest['candidate']))
    csv = directory / 'v7_7.csv'; example = root / 'data/raw/example_submission.csv'
    export_submission(evaluation['rowids'].ravel(), p, example, csv, 529390)
    validate_csv(csv, example, 529390); verify_csv_probabilities(csv, evaluation['rowids'].ravel(), p)
    save_arrays(directory / 'predictions.npz', candidate=p, rowids=evaluation['rowids'], members=data['members'], anchor=data['anchor'])
    number = claim['arguments']['number']
    if number > 1:
        first = result(root, work, 'export', dict(number=1))
        old = arrays(path(work, first['vectors']))
        require(digest(csv) == digest(path(work, first['csv']))
                and np.allclose(data['members'], old['members'], rtol=0, atol=1e-15)
                and np.allclose(p, old['candidate'], rtol=0, atol=1e-15), 'Independent replay differs')
    receipt = {'candidate': manifest['candidate'], 'number': number, 'purpose': 'research',
               'fresh_neural_features': True, 'fresh_context_features': True, 'fresh_slate_features': True,
               'source_sha256': claim['source']['sha256'], 'manifest_sha256': digest(manifest_path),
               'model_sha256': canonical(manifest['models']), 'csv_sha256': digest(csv), 'compute_owner': claim['owner'],
               'vectors_sha256': digest(directory / 'predictions.npz'), 'feature_sha256': digest(path(work, saved['view']))}
    write_new(directory / 'receipt.json', receipt)
    return {'csv': relative(work, csv), 'receipt': relative(work, directory / 'receipt.json'),
            'vectors': relative(work, directory / 'predictions.npz'), 'number': number}


def seal(root, work):
    state = read(root / 'artifacts/rebuild/compute-ledger.json'); manifest = read(work / 'manifest.json')
    require(state['total_seconds'] == 43200 and state['carry_identity'] == manifest['genesis_sha256']
            and 0 <= state['used_seconds'] <= 43200 and not any(a['status'] == 'running' for a in state['attempts'])
            and abs(sum(a['seconds'] for a in state['attempts']) - state['used_seconds']) < 1e-6
            and canonical(state['attempts'][:manifest['prior_attempts']]) == manifest['prior_attempts_sha256'], 'Accounting unsettled or prior charges changed')
    filename = work / 'accounting' / (canonical(state) + '.json'); runner.same(filename, state)
    return filename


def evidence(root, work):
    """Create ONCE, with a settled pre-verification seal; resume reuses bytes."""
    filename = work / 'promotion-evidence.json'
    if filename.exists():
        saved = read(filename)
        for name, value in saved['artifacts'].items(): require(digest(inside(root, name)) == value, 'Frozen verification evidence changed')
        return filename
    require_gates(root, work); manifest_path, manifest = run_manifest(root, work)
    original = read(work / 'manifest.json'); development = result(root, work, 'selection'); ledger = seal(root, work)
    floors = path(work, result(root, work, 'preflight')['floors'])
    exports = []
    for number in (1, 2, 3):
        saved = result(root, work, 'export', dict(number=number))
        exports.append({k: path(work, saved[k]).relative_to(root).as_posix() for k in ('csv', 'receipt', 'vectors')})
    e = {'schema_version': 2, 'release_profile': PROFILE, 'purpose': 'research', 'model_role': 'candidate',
         'candidate': manifest['candidate'], 'baseline_lineage': SPEC['baseline_lineage'], 'bootstrap': SPEC['bootstrap'],
         'comparators': manifest['comparators'], 'run_manifest': manifest_path.relative_to(root).as_posix(),
         'development': {'winner': manifest['candidate'], 'passed': True, 'report': path(work, development['report']).relative_to(root).as_posix()},
         'source_record': (work / 'scientific-source-record.json').relative_to(root).as_posix(),
         'source_directory': (work / 'snapshot').relative_to(root).as_posix(), 'engineering_receipt': original['engineering_receipt'],
         'accounting_record': ledger.relative_to(root).as_posix(), 'budget_genesis': 'artifacts/rebuild/genesis.json',
         'data_metadata': 'artifacts/rebuild/data/metadata.json', 'example_submission': 'data/raw/example_submission.csv',
         'historical_floors': floors.relative_to(root).as_posix(), 'exports': exports,
         'predictions': {n: path(work, result(root, work, 'predict', dict(case=n))['predictions']).relative_to(root).as_posix()
                         for n in ('primary', 'secondary', 'five_day')}}
    paths = set(manifest['models']) | set(manifest['references'].values()) | {v[0] for v in FLOORS.values()} | set(e['predictions'].values())
    paths.update(e[k] for k in ('run_manifest', 'source_record', 'engineering_receipt', 'accounting_record', 'budget_genesis',
                               'data_metadata', 'example_submission', 'historical_floors'))
    paths.add(e['development']['report'])
    for row in exports: paths.update(row.values())
    for marker in (work / 'tasks').glob('*.json'):
        token = read(marker)['token']; validation = work / 'attempts' / (token + '.validated.json')
        paths.update(p.relative_to(root).as_posix() for p in (marker, validation, work / 'attempts' / (token + '.claim.json')))
        paths.update((work / p).relative_to(root).as_posix() for p in read(validation)['files'])
    e['artifacts'] = {name: digest(inside(root, name)) for name in sorted(paths)}
    write_new(filename, e)
    return filename


def verify_release(root, work):
    from row_auc_final_utils import verify_csv_probabilities
    require_gates(root, work)
    for number in (1, 2, 3):
        saved = result(root, work, 'export', dict(number=number)); data = arrays(path(work, saved['vectors']))
        verify_csv_probabilities(path(work, saved['csv']), data['rowids'].ravel(), data['candidate'])
    decision = promotion.verify_promotion(root, read(work / 'promotion-evidence.json'))
    require(decision['eligible'], 'Artifact-backed verification refused: ' + '; '.join(decision['blockers']))
    return decision


def publish(root, work):
    require_gates(root, work); decision = result(root, work, 'verify-release'); e = read(evidence(root, work))
    require(decision['eligible'] and decision['evidence_sha256'] == canonical(e)
            and all(decision['historical_floors'].values()) and not promotion.metric_gates(decision['comparisons'])['blockers'], 'Paid verification does not support publication')
    ledger_path = seal(root, work); ledger = read(ledger_path); before = read(root / e['accounting_record'])
    require(ledger['attempts'][:len(before['attempts'])] == before['attempts'], 'Final accounting does not extend verified seal')
    saved = result(root, work, 'export', dict(number=1)); source = path(work, saved['csv'])
    output = root / 'artifacts/submissions/v7_7.csv'; output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): require(digest(output) == digest(source), 'Preserve different existing submission')
    else: os.link(source, output)
    record = {'schema_version': 2, 'model_role': 'promoted_v7_7', 'candidate': e['candidate'], 'release_profile': PROFILE,
              'baseline_lineage': SPEC['baseline_lineage'], 'promotion_supported': True, 'submission_ready': True,
              'upload_ready': True, 'competition_upload_performed': False, 'rows': 529390,
              'submission': output.relative_to(root).as_posix(), 'sha256': digest(output),
              'evidence_sha256': canonical(e), 'decision_sha256': canonical(decision),
              'accounting_record': ledger_path.relative_to(root).as_posix(), 'accounting_sha256': digest(ledger_path),
              'historical_v6_v7_reproduced': False}
    runner.same(output.with_suffix('.csv.receipt.json'), record)
    checksum = output.with_suffix('.csv.sha256'); checksum_text = record['sha256'] + '  ' + output.name + '\n'
    if checksum.exists(): require(checksum.read_text() == checksum_text, 'Checksum sidecar differs')
    else:
        with checksum.open('x') as stream: stream.write(checksum_text)
    # Public config and readiness flags are the LAST writes.
    runner.same(root / 'configs/v7_7.json', record); runner.same(work / 'readiness.json', record)


def verify(root, work):
    ready = work / 'readiness.json'
    if not ready.exists(): return {'v7_7_promoted': False, 'submission_ready': False}
    record = read(ready); decision = result(root, work, 'verify-release'); e = read(evidence(root, work))
    require_gates(root, work)
    csv = inside(root, record['submission'])
    require(record == read(root / 'configs/v7_7.json') == read(csv.with_suffix('.csv.receipt.json'))
            and record['sha256'] == digest(csv) and record['evidence_sha256'] == canonical(e)
            and record['decision_sha256'] == canonical(decision) and decision['eligible']
            and digest(inside(root, record['accounting_record'])) == record['accounting_sha256']
            and csv.with_suffix('.csv.sha256').read_text() == record['sha256'] + '  ' + csv.name + '\n', 'Published release changed')
    return {'v7_7_promoted': True, 'submission_ready': True, 'submission': record['submission'], 'sha256': record['sha256']}


def report(root, work):
    ledger = seal(root, work)
    record = {'schema_version': 1, 'protocol': SPEC, 'manifest_sha256': digest(work / 'manifest.json'),
              'scientific_source': read(work / 'scientific-source.json')['source']['commit'], 'status': verify(root, work),
              'stages': {name: runner.completed(root, work, name) for name in ('selection', 'screen', 'stress')},
              'accounting_sha256': digest(ledger), 'cumulative_seconds': read(ledger)['used_seconds'],
              'competition_upload_performed': False, 'historical_dates_reused': True}
    destination = root / 'docs/results/v7.7-reliability-slate-v1'
    runner.same(destination / 'results.json', record); runner.same(destination / 'accounting.json', read(ledger))


def prepare_meta(root, work, meta, candidates, phase):
    for prefix in prefixes(meta):
        for member in MEMBERS: runner.launch(root, work, 'teacher', dict(meta=meta, prefix=prefix, member=member), phase)
        runner.launch(root, work, 'training-view', dict(meta=meta, prefix=prefix), phase)
    for candidate in candidates: runner.launch(root, work, 'head', dict(meta=meta, candidate=candidate), phase)
    for member in MEMBERS: runner.launch(root, work, 'deploy', dict(meta=meta, member=member), phase)


def controller(root, work, action):
    if action == 'verify': print(verify(root, work)); return
    if action == 'replay':
        require_gates(root, work)
        runner.launch(root, work, 'export', dict(number=3), 'promotion', final=True); return
    runner.scientific_open(root, work)
    runner.launch(root, work, 'preflight')
    if action == 'preflight': return
    for meta in METAS[:2]: prepare_meta(root, work, meta, list(CANDIDATES), 'develop')
    for name in ('jan22_long', 'jan24_forward'): runner.launch(root, work, 'view', dict(case=name), 'develop')
    selection = runner.launch(root, work, 'selection', phase='develop')
    if not selection['passed']: report(root, work); print('STOP: no preregistered development winner'); return
    if action == 'develop': return
    for meta in METAS[2:]: prepare_meta(root, work, meta, [selection['winner']], 'promotion')
    for name in ('primary', 'secondary', 'five_day'): runner.launch(root, work, 'view', dict(case=name), 'promotion')
    runner.launch(root, work, 'freeze', phase='promotion')
    for name in ('primary', 'secondary'): runner.launch(root, work, 'predict', dict(case=name), 'promotion')
    screen = runner.launch(root, work, 'screen', phase='promotion')
    if not screen['passed']: report(root, work); print('STOP: screen or historical floor failed'); return
    if action == 'screen': return
    runner.launch(root, work, 'predict', dict(case='five_day'), 'promotion')
    stress = runner.launch(root, work, 'stress', phase='promotion')
    if not stress['passed']: report(root, work); print('STOP: stress or historical floor failed'); return
    if action == 'stress': return
    for number in (1, 2, 3): runner.launch(root, work, 'export', dict(number=number), 'promotion', final=True)
    evidence(root, work)
    runner.launch(root, work, 'verify-release', phase='promotion', final=True)
    publish(root, work); report(root, work); print(verify(root, work))


def worker(root, work, directory, claim):
    action, args = claim['action'], claim['arguments']
    if action == 'preflight': return preflight(root, work, directory, claim)
    result(root, work, 'preflight')
    if action == 'verify-release': return verify_release(root, work)
    raw = dataset.load(root / 'artifacts/rebuild/data', 'train')
    if action == 'teacher': return teacher(root, work, directory, raw, args)
    if action == 'training-view': return training_view(root, work, directory, raw, args)
    if action == 'head': return fit_head(root, work, directory, raw, args)
    if action == 'deploy': return deploy(root, work, directory, raw, args)
    if action == 'view': return prediction_view(root, work, directory, raw, args)
    if action == 'selection': return selection(root, work, directory, raw)
    if action == 'freeze': return freeze(root, work, directory, claim)
    if action == 'predict': return predict_case(root, work, directory, raw, args)
    if action in ('screen', 'stress'): return gates(root, work, directory, raw, action)
    if action == 'export': return export(root, work, directory, raw, claim)
    raise ValueError('Undeclared worker action: ' + action)
