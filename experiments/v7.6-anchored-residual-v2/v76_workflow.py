"""Causal preparation, fixed selection, original gates and verified publication."""
from dataclasses import asdict
from pathlib import Path
import os
import numpy as np
from ml_workflow.contracts import canonical, compose, digest, first_indices, inside, require
from ml_workflow.storage import read, write_new, save_arrays
from ml_workflow import policy, promotion
from experiment_runner import FastAuc, PairedBootstrapEvaluator
from v76_protocol import SPEC, PREFIXES, BLOCKS, BOUNDARIES, INNER, PENALTIES, MEMBERS, CASES, FULL, JAN20, DAY, HOUR, pairs, validation_pairs, user_folds, fitting_indices
import v76_assets as assets
import residual
import data as dataset
import pipeline as prior

EVAL_EVENTS = 52939
EVAL_ROWS = 529390


def auc(y, p):
    ids = np.repeat(np.arange(len(y)), 10)
    return FastAuc(y, p, ids).value(np.ones(len(y), dtype=np.int64))


def point(y, reference, candidate):
    a, b = auc(y, reference), auc(y, candidate)
    return {'baseline_auc': a, 'candidate_auc': b, 'auc_diff': b - a}


def completed(root, work, task):
    from runner import completed as check
    return check(root, work, task)


def checked(root, work, stage):
    task = 'select' if stage == 'selection' else stage
    require(completed(root, work, task), 'Missing accounted stage: ' + stage)
    path = work / ('selection.json' if stage == 'selection' else 'stages/' + stage + '.json')
    result = read(path)
    require(result['manifest_sha256'] == digest(work / 'manifest.json'), 'Stage generation differs')
    if stage == 'selection': require(result['passed'] == (result['penalty'] is not None)
                                    and result['penalty'] == choose_penalty(result['results']), 'Selection outcome inconsistent')
    elif stage == 'develop': require(result['winner'] == policy.dual_winner(SPEC, result['results'])
                                     and result['passed'] == (result['winner'] is not None), 'Development outcome inconsistent')
    else: require(result['passed'] == policy.stage_passed(SPEC, stage, result['results']), 'Promotion outcome inconsistent')
    return result


def selected(root, work):
    result = checked(root, work, 'selection')
    require(result['passed'] and result['penalty'] in PENALTIES, 'Inner selection failed; V7.6 is closed')
    return result['penalty']


def require_gates(root, work):
    selected(root, work)
    stages = [checked(root, work, name) for name in ('develop', 'screen', 'stress')]
    require(all(r['passed'] and r['winner'] == 'residual' for r in stages), 'All original stages must pass')
    gates = policy.dual_gates(stages[1], stages[2])
    require(all(v for group in gates.values() for v in group.values()), 'All eight gates must pass')
    return gates


def preflight(root, work, claim):
    import torch
    m = read(work / 'manifest.json'); parent = root / 'artifacts/rebuild'
    engineering = read(inside(root, m['engineering_receipt']))
    require(digest(inside(root, m['engineering_receipt'])) == m['engineering_sha256']
            and engineering['git_commit'] == claim['source']['commit'] and engineering['kind'] == 'source_readiness'
            and all(engineering['checks'].get(k) == 'success' for k in ('cpu_tests', 'model_contracts', 'source_integrity', 'source_archive')),
            'Engineering checks must describe the frozen scientific source')
    require(torch.cuda.is_available(), 'CUDA feature extraction is required before research')
    dataset.verify(parent / 'data')
    for name, expected in m['protected_inputs'].items(): require(digest(inside(root, name)) == expected, 'Historical input changed: ' + name)
    raw = dataset.load(parent / 'data', 'train')
    require(len(raw['u']) == 686824 and int(raw['y'].sum()) == 432023, 'Official corpus differs')
    path = work / 'preflight.json'
    write_new(path, {'manifest_sha256': digest(work / 'manifest.json'), 'source_commit': claim['source']['commit'],
              'foundation_members': len(m['foundation_folders']), 'protected_inputs': len(m['protected_inputs']),
              'cuda': torch.cuda.get_device_name(0), 'source_sha256': claim['source']['sha256'], 'purpose': 'research', 'passed': True})
    return assets.hashes(work, [path])


def prepare_block(root, work, b):
    from runner import launch
    for c, d, end in pairs(b):
        for member in MEMBERS:
            launch(root, 'feature', f'feature-{c}-{c}-{end}-{member}', dict(d=c, begin=c, end=end, member=member))
            launch(root, 'combiner', f'combiner-{c}-{end}-{member}', dict(c=c, end=end, member=member))
            launch(root, 'feature', f'feature-{d}-{b}-{b+6*HOUR}-{member}', dict(d=d, begin=b, end=b + 6 * HOUR, member=member))
        launch(root, 'view', f'view-{c}-{d}-{end}-{b}', dict(c=c, d=d, fit_end=end, begin=b, end=b + 6 * HOUR))


def fit_data(root, work, raw, boundary, fold=None):
    keep = fitting_indices(raw, boundary, fold); values = [] ; dependencies = {}
    for b in BLOCKS:
        if not max(BLOCKS[0], boundary - 2 * DAY) <= b < boundary: continue
        for c, d, end in pairs(b):
            task = f'view-{c}-{d}-{end}-{b}'
            require(completed(root, work, task), 'Unaccounted historical view: ' + task)
            path = assets.view_path(work, c, d, end, b, b + 6 * HOUR); row = assets.matrix(path)
            mask = np.isin(row['indices'], keep)
            values.append({k: row[k][mask] for k in ('X', 'anchor', 'active', 'indices')})
            dependencies.update(assets.hashes(work, [path, path.with_suffix('.receipt.json')]))
    require(values, 'No fitting forecasts')
    result = {k: np.concatenate([v[k] for v in values]) for k in values[0]}
    unique = np.unique(result['indices'])
    require(np.array_equal(unique, np.sort(keep)) and (raw['t'][unique] < boundary).all(), 'Fitting history crosses its boundary or omits users')
    return result, raw['y'][unique], dependencies


def residual_path(work, boundary, penalty, fold=None):
    return work / ('inner-models' if fold is not None else 'residuals') / str(boundary) / f'{penalty:g}-{fold if fold is not None else "full"}.pt'


def fit_residual(root, work, raw, args, claim, inner):
    boundary, penalty = args['boundary'], args['penalty']; fold = args.get('fold')
    if inner:
        require(boundary in INNER and penalty in PENALTIES and type(fold) is int and 0 <= fold < 5, 'Undeclared inner model')
    else:
        require(boundary in BOUNDARIES and fold is None and penalty == selected(root, work), 'Outer recipe changed')
    values, labels, dependencies = fit_data(root, work, raw, boundary, fold)
    metadata = {'purpose': 'research', 'boundary': boundary, 'fold': fold, 'manifest_sha256': digest(work / 'manifest.json'),
                'source_sha256': claim['source']['sha256'], 'fitting_files': dependencies,
                'selection_sha256': None if inner else digest(work / 'selection.json'),
                'labels_sha256': canonical(labels.tolist())}
    path = residual_path(work, boundary, penalty, fold)
    state = residual.fit(values, labels, penalty, metadata, path)
    model, saved = residual.load(path, metadata, penalty)
    require(np.array_equal(residual.predict(values, model, saved), residual.predict(values, model, state)), 'Saved residual differs')
    outputs = [path]
    if inner:
        for regime, (c, d, end) in validation_pairs(boundary).items():
            data = assets.matrix(assets.view_path(work, c, d, end, boundary, boundary + 6 * HOUR))
            held = user_folds(raw['u'][data['indices']]) == fold
            validation = {k: data[k][held] for k in ('X', 'anchor', 'active', 'indices')}
            require(len(validation['indices']) > 0 and not np.isin(raw['u'][validation['indices']], raw['u'][np.unique(values['indices'])]).any(), 'Residual user fold leaked')
            p = residual.predict(validation, model, saved)
            out = path.with_name(path.stem + '-' + regime + '.npz')
            save_arrays(out, indices=validation['indices'], candidate=p, baseline=data['baseline'][held], v7=data['anchor'][held])
            outputs.append(out)
    print('residual-fit', boundary, penalty, fold, state['fit_events'], state['fit_views'], state['objective_trace'][-1], flush=True)
    return assets.hashes(work, outputs)


def choose_penalty(rows):
    eligible = []
    for penalty in PENALTIES:
        comparisons = rows[f'{penalty:g}']; groups = {'V6': [], 'V7': []}; lower = []
        for case in comparisons.values():
            for anchor in groups:
                row = case[anchor]; policy.validate_metric(row, confidence=True)
                groups[anchor].append(row['auc_diff']); lower.append(row['ci_lower'])
        require(set(comparisons) == {str(b) + '/' + r for b in INNER for r in ('freshest', 'oldest')}, 'Missing inner regime/block evidence')
        if min(v for gains in groups.values() for v in gains) >= 0 and all(np.mean(v) > 0 for v in groups.values()):
            eligible.append((penalty, min(lower)))
    if not eligible: return None
    best = max(v for _, v in eligible)
    return max(p for p, v in eligible if best - v <= 1e-12)


def select(root, work, raw):
    rows = {}; dependencies = {}
    for penalty in PENALTIES:
        results = {}
        for b in INNER:
            ix = first_indices(raw['u'], raw['t'], b, b + 6 * HOUR)
            for regime in ('freshest', 'oldest'):
                bundles = []
                for fold in range(5):
                    require(completed(root, work, f'innerfit-{b}-{penalty:g}-{fold}'), 'Incomplete inner folds')
                    path = residual_path(work, b, penalty, fold).with_name(f'{penalty:g}-{fold}-{regime}.npz')
                    with np.load(path, allow_pickle=False) as values: bundles.append({k: values[k] for k in values.files})
                    dependencies.update(assets.hashes(work, [path]))
                indices = np.concatenate([v['indices'] for v in bundles]); order = np.argsort(indices, kind='stable')
                require(np.array_equal(indices[order], ix), 'Inner validation coverage differs')
                scores = {k: np.concatenate([v[k] for v in bundles])[order] for k in ('candidate', 'baseline', 'v7')}
                results[str(b) + '/' + regime] = {a: asdict(PairedBootstrapEvaluator().evaluate(scores['candidate'], scores[k], raw['y'][ix], raw['u'][ix]))
                                                for a, k in (('V6', 'baseline'), ('V7', 'v7'))}
        rows[f'{penalty:g}'] = results
    winner = choose_penalty(rows)
    out = work / 'selection.json'
    write_new(out, {'manifest_sha256': digest(work / 'manifest.json'), 'penalty': winner, 'passed': winner is not None,
                   'results': rows, 'evidence': dependencies, 'bootstrap': SPEC['bootstrap'], 'outer_results_used': False})
    print('v76-selection', winner, 'passed', winner is not None, flush=True)
    return assets.hashes(work, [out])


def prepare_case(root, work, name):
    from runner import launch
    c, d, end = CASES[name]
    for member in MEMBERS:
        launch(root, 'feature', f'feature-{c}-{c}-{c+DAY}-{member}', dict(d=c, begin=c, end=c + DAY, member=member))
        launch(root, 'combiner', f'combiner-{c}-{c+DAY}-{member}', dict(c=c, end=c + DAY, member=member))
        launch(root, 'feature', f'feature-{d}-{d}-{end}-{member}', dict(d=d, begin=d, end=end, member=member))
    launch(root, 'reference', 'reference-' + name, {'case': name})


def reference(root, work, raw, name):
    require(name in CASES, 'Unknown outer case')
    c, d, end = CASES[name]
    values, dependencies, _ = assets.views(root, work, raw, c, d, c + DAY, d, end)
    old = root / 'artifacts/v7_5/predictions' / ('A-' + name + '.npz')
    if old.exists():
        with np.load(old, allow_pickle=False) as previous:
            require(np.array_equal(previous['indices'], values['indices'])
                    and np.allclose(values['baseline'], previous['baseline'], atol=1e-15, rtol=0)
                    and np.allclose(values['anchor'], previous['v7'], atol=1e-15, rtol=0), 'Existing development comparator differs')
    path = work / 'references' / (name + '.npz')
    save_arrays(path, **{k: values[k] for k in ('X', 'anchor', 'active', 'indices', 'baseline', 'rowids')})
    write_new(path.with_suffix('.receipt.json'), {'sha256': digest(path), 'identity': {'case': name, 'dependencies': dependencies,
               'manifest_sha256': digest(work / 'manifest.json')}})
    return assets.hashes(work, [path, path.with_suffix('.receipt.json')])


def freeze(root, work, claim):
    penalty = selected(root, work); models = dict(read(work / 'manifest.json')['foundation_files'])
    for boundary in BOUNDARIES:
        require(completed(root, work, 'outerfit-' + str(boundary)), 'Missing frozen residual state')
        path = residual_path(work, boundary, penalty)
        models[path.relative_to(root).as_posix()] = digest(path)
    for name in CASES:
        require(completed(root, work, 'reference-' + name), 'Missing frozen comparator')
        c = CASES[name][0]
        for member in MEMBERS:
            path = assets.combiner_path(work, c, c + DAY, member)
            models[path.relative_to(root).as_posix()] = digest(path)
    identities = {a: canonical({'recipe': 'rebuilt_v6_v7_recipe', 'models': models,
                               'data_sha256': read(work / 'manifest.json')['data_sha256'], 'gamma': gamma})
                  for a, gamma in (('V6', 1.), ('V7', 1.12))}
    references = {}
    for name in CASES:
        data = assets.matrix(work / 'references' / (name + '.npz'))
        path = work / 'frozen-references' / (name + '.npz')
        save_arrays(path, indices=data['indices'], rowids=data['rowids'], labels=dataset.load(root / 'artifacts/rebuild/data', 'train')['y'][data['indices']],
                    baseline=data['baseline'], v7=data['anchor'], V6_identity=np.asarray(identities['V6']), V7_identity=np.asarray(identities['V7']))
        references[name] = path.relative_to(root).as_posix()
    source_path = work / 'scientific-source-record.json'; write_new(source_path, claim['source'])
    genesis = read(root / 'artifacts/rebuild/genesis.json')
    manifest = {'schema_version': 1, 'candidate': 'residual', 'baseline_lineage': SPEC['baseline_lineage'],
        'comparators': identities, 'models': models, 'source_sha256': claim['source']['sha256'],
        'data_sha256': read(work / 'manifest.json')['data_sha256'], 'protocol': SPEC, 'references': references,
        'owner_prefix': genesis['generation'] + '/v76-', 'selection_sha256': digest(work / 'selection.json'),
        'generation_manifest_sha256': digest(work / 'manifest.json'), 'penalty': penalty,
        'reference_hashes': {p: digest(root / p) for p in references.values()}}
    path = work / 'run-manifest.json'; write_new(path, manifest)
    return assets.hashes(work, [path, source_path, *[root / p for p in references.values()]])


def frozen(root, work):
    require(completed(root, work, 'freeze'), 'Freeze model hashes before outer scoring')
    m = read(work / 'run-manifest.json')
    require(m['penalty'] == selected(root, work) and m['selection_sha256'] == digest(work / 'selection.json')
            and m['generation_manifest_sha256'] == digest(work / 'manifest.json'), 'Frozen selection changed')
    for path, expected in {**m['models'], **m['reference_hashes']}.items(): require(digest(inside(root, path)) == expected, 'Frozen inference dependency changed')
    return m


def require_residual_role(state, manifest, boundary):
    metadata = state['metadata']
    require(metadata['purpose'] == 'research' and metadata['boundary'] == boundary
            and metadata['selection_sha256'] == manifest['selection_sha256']
            and metadata['source_sha256'] == manifest['source_sha256']
            and state['penalty'] == manifest['penalty'], 'Residual source/cohort/purpose binding differs')


def score(root, work, raw, name):
    m = frozen(root, work)
    if name in ('primary', 'secondary'): require(checked(root, work, 'develop')['passed'], 'Development failure is terminal')
    if name == 'five_day': require(checked(root, work, 'screen')['passed'], 'Screening failure is terminal')
    c, d, end = CASES[name]; path = residual_path(work, c + DAY, m['penalty'])
    model, state = residual.load(path, penalty=m['penalty'])
    require_residual_role(state, m, c + DAY)
    values = assets.matrix(work / 'references' / (name + '.npz')); ix = first_indices(raw['u'], raw['t'], d, end)
    require(np.array_equal(values['indices'], ix), 'Official first-user cohort differs')
    prior.check_cohort_size(name, len(ix))
    p = residual.predict(values, model, state); y = raw['y'][ix]
    row = {'case': name, 'candidate': 'residual', 'status': 'valid', 'events': len(ix),
           'baseline_lineage': SPEC['baseline_lineage'], **point(y, values['baseline'], p),
           'v7_comparison': point(y, values['anchor'], p), 'residual_sha256': digest(path)}
    if name in ('primary', 'five_day'):
        for result, reference in ((row, values['baseline']), (row['v7_comparison'], values['anchor'])):
            result.update(asdict(PairedBootstrapEvaluator().evaluate(p, reference, y, raw['u'][ix])))
    age = (raw['t'][ix] - d) / HOUR; late = (age >= 48) & (age < 120)
    if late.any():
        row['late'] = {'events': int(late.sum()), **point(y[late], values['baseline'][late], p[late])}
        row['v7_comparison']['late'] = {'events': int(late.sum()), **point(y[late], values['anchor'][late], p[late])}
    if name == 'five_day': prior.check_cohort_size('late', int(late.sum()))
    vectors = work / 'predictions' / (name + '.npz')
    save_arrays(vectors, indices=ix, rowids=raw['rowids'][ix], labels=y, candidate=p, baseline=values['baseline'], v7=values['anchor'],
                V6_identity=np.asarray(m['comparators']['V6']), V7_identity=np.asarray(m['comparators']['V7']))
    output = work / 'results' / (name + '.json'); write_new(output, row)
    print('v76-score', name, 'vsV6', row['auc_diff'], 'vsV7', row['v7_comparison']['auc_diff'], flush=True)
    return assets.hashes(work, [vectors, output])


def summarize(root, work, stage):
    names = {'develop': ['jan22_long', 'jan24_forward'], 'screen': ['primary', 'secondary'], 'stress': ['five_day']}[stage]
    results = {}
    for name in names:
        require(completed(root, work, 'score-' + name), 'Scoring lacks settled ownership')
        results['residual/' + name if stage == 'develop' else name] = read(work / 'results' / (name + '.json'))
    winner = policy.dual_winner(SPEC, results) if stage == 'develop' else 'residual'
    passed = winner is not None if stage == 'develop' else policy.stage_passed(SPEC, stage, results)
    path = work / 'stages' / (stage + '.json')
    write_new(path, {'manifest_sha256': digest(work / 'manifest.json'), 'winner': winner, 'passed': bool(passed), 'results': results})
    print('v76-stage', stage, passed, flush=True)
    return assets.hashes(work, [path])


def export(root, work, raw, number, claim):
    from submission_export import export_submission, validate_csv
    from row_auc_final_utils import verify_csv_probabilities
    require_gates(root, work); m = frozen(root, work)
    evaluation = dataset.load(root / 'artifacts/rebuild/data', 'evaluation')
    require('y' not in evaluation and len(evaluation['u']) == EVAL_EVENTS, 'Official evaluation corpus differs')
    c = JAN20 + 5 * DAY
    values, _, graph = assets.views(root, work, raw, c, FULL, c + DAY, FULL, FULL + 7 * DAY, evaluation=evaluation, fresh=True)
    model, saved = residual.load(residual_path(work, c + DAY, m['penalty']), penalty=m['penalty'])
    require_residual_role(saved, m, c + DAY)
    p = residual.predict(values, model, saved)
    folder = work / 'exports' / (str(number) + '-' + claim['token']); folder.mkdir(parents=True)
    csv = folder / 'v7_6.csv'; example = root / 'data/raw/example_submission.csv'
    export_submission(evaluation['rowids'].ravel(), p, example, csv, EVAL_ROWS)
    validate_csv(csv, example, EVAL_ROWS); verify_csv_probabilities(csv, evaluation['rowids'].ravel(), p)
    vectors = folder / 'predictions.npz'
    save_arrays(vectors, candidate=p, rowids=evaluation['rowids'], members=values['members'], anchor=values['anchor'])
    graph_path = folder / 'graph.npz'; save_arrays(graph_path, **graph)
    if number > 1:
        previous = exported(root, work, 1)
        require(digest(csv) == digest(inside(work, previous['csv'])), 'Independent CSV bytes differ')
        with np.load(inside(work, previous['vectors']), allow_pickle=False) as old:
            require(np.allclose(values['members'], old['members'], rtol=0, atol=1e-15)
                    and np.allclose(p, old['candidate'], rtol=0, atol=1e-15), 'Independent predictions differ')
    owner = read(root / 'artifacts/rebuild/genesis.json')['generation'] + '/v76-' + claim['token']
    receipt = folder / 'receipt.json'
    write_new(receipt, {'candidate': 'residual', 'fresh_neural_features': True, 'fresh_graph': True,
        'source_sha256': claim['source']['sha256'], 'manifest_sha256': digest(work / 'run-manifest.json'),
        'model_sha256': canonical(m['models']), 'csv_sha256': digest(csv), 'compute_owner': owner,
        'csv': csv.relative_to(work).as_posix(), 'vectors': vectors.relative_to(work).as_posix(),
        'graph': graph_path.relative_to(work).as_posix(), 'purpose': 'research', 'number': number,
        'files': assets.hashes(work, [csv, vectors, graph_path])})
    index = work / 'exports' / (str(number) + '.json'); prior.write_same(index, {'receipt': receipt.relative_to(work).as_posix(), 'sha256': digest(receipt)})
    return assets.hashes(work, [csv, vectors, graph_path, receipt, index])


def exported(root, work, number):
    require(completed(root, work, 'export-' + str(number)), 'Export lacks completed ownership')
    index = read(work / 'exports' / (str(number) + '.json')); path = inside(work, index['receipt'])
    require(digest(path) == index['sha256'], 'Export receipt differs')
    result = read(path)
    for name, expected in result['files'].items(): require(digest(inside(work, name)) == expected, 'Export output differs')
    return result


def seal(root, work):
    parent = root / 'artifacts/rebuild'; state = read(parent / 'compute-ledger.json'); m = read(work / 'manifest.json')
    require(state['total_seconds'] == 43200 and state['carry_identity'] == m['genesis_sha256']
            and 0 <= state['used_seconds'] <= 43200 and not any(a['status'] == 'running' for a in state['attempts'])
            and abs(sum(a['seconds'] for a in state['attempts']) - state['used_seconds']) < 1e-6
            and canonical(state['attempts'][:m['prior_attempts']]) == m['prior_attempts_sha256'], 'Accounting unsettled or prior charges changed')
    path = work / 'accounting' / (digest(parent / 'compute-ledger.json') + '.json'); prior.write_same(path, state)
    return path


def evidence(root, work):
    m = frozen(root, work); record = read(work / 'manifest.json'); source = read(work / 'scientific-source.json')
    ledger = seal(root, work); paths = set(m['models']) | set(m['references'].values())
    exports = []
    for number in (1, 2, 3):
        exported(root, work, number)
        index = read(work / 'exports' / (str(number) + '.json')); receipt = inside(work, index['receipt'])
        row = read(receipt); exports.append({'csv': (work / row['csv']).relative_to(root).as_posix(), 'receipt': receipt.relative_to(root).as_posix()})
    e = {'schema_version': 1, 'purpose': 'research', 'model_role': 'candidate', 'candidate': 'residual',
         'baseline_lineage': SPEC['baseline_lineage'], 'bootstrap': SPEC['bootstrap'], 'comparators': m['comparators'],
         'run_manifest': (work / 'run-manifest.json').relative_to(root).as_posix(),
         'development': {'winner': 'residual', 'passed': True, 'report': (work / 'stages/develop.json').relative_to(root).as_posix()},
         'source_record': (work / 'scientific-source-record.json').relative_to(root).as_posix(),
         'source_directory': (work / source['snapshot_relative']).relative_to(root).as_posix(),
         'engineering_receipt': record['engineering_receipt'], 'accounting_record': ledger.relative_to(root).as_posix(),
         'budget_genesis': 'artifacts/rebuild/genesis.json', 'data_metadata': 'artifacts/rebuild/data/metadata.json',
         'example_submission': 'data/raw/example_submission.csv', 'exports': exports,
         'predictions': {n: (work / 'predictions' / (n + '.npz')).relative_to(root).as_posix() for n in ('primary', 'secondary', 'five_day')}}
    for value in (e['run_manifest'], e['development']['report'], e['source_record'], e['engineering_receipt'],
                  e['accounting_record'], e['budget_genesis'], e['data_metadata'], e['example_submission'], *e['predictions'].values()): paths.add(value)
    for task in (work / 'tasks').glob('*.json'):
        record = read(task); token = record['token']
        for path in (task, work / 'attempts' / (token + '.claim.json'), work / 'attempts' / (token + '.validated.json')): paths.add(path.relative_to(root).as_posix())
        for name in read(work / 'attempts' / (token + '.validated.json'))['files']: paths.add((work / name).relative_to(root).as_posix())
    e['artifacts'] = {p: digest(inside(root, p)) for p in sorted(paths)}
    path = work / 'promotion-evidence.json'; prior.write_same(path, e)


def verify_release(root, work):
    from row_auc_final_utils import verify_csv_probabilities
    require_gates(root, work); m = frozen(root, work)
    for number in (1, 2, 3):
        row = exported(root, work, number)
        require(row['purpose'] == 'research' and row['number'] == number, 'Replay role differs')
        with np.load(inside(work, row['vectors']), allow_pickle=False) as values:
            verify_csv_probabilities(inside(work, row['csv']), values['rowids'].ravel(), values['candidate'])
    result = promotion.verify_promotion(root, read(work / 'promotion-evidence.json'))
    require(result['eligible'] is True, 'Artifact-backed promotion refused: ' + '; '.join(result['blockers']))
    result['run_manifest_sha256'] = digest(work / 'run-manifest.json')
    path = work / 'promotion-decision.json'; write_new(path, result)
    return assets.hashes(work, [path])


def publish(root, work):
    require(completed(root, work, 'verify-release'), 'Paid promotion verification must settle before publication')
    require_gates(root, work); m = frozen(root, work); decision = read(work / 'promotion-decision.json')
    require(decision['eligible'] is True and not promotion.metric_gates(decision['comparisons'])['blockers']
            and decision['evidence_sha256'] == canonical(read(work / 'promotion-evidence.json'))
            and decision['run_manifest_sha256'] == digest(work / 'run-manifest.json'), 'Promotion evidence changed')
    ledger = seal(root, work); old_seal = read(root / read(work / 'promotion-evidence.json')['accounting_record'])
    state = read(ledger)
    require(state['attempts'][:len(old_seal['attempts'])] == old_seal['attempts'], 'Final accounting does not extend verified seal')
    row = exported(root, work, 1); source = inside(work, row['csv']); output = root / 'artifacts/submissions/v7_6.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): require(digest(output) == digest(source), 'Preserve different existing submission')
    else: os.link(source, output)
    record = {'schema_version': 1, 'model_role': 'promoted_v7_6', 'baseline_lineage': SPEC['baseline_lineage'],
              'promotion_supported': True, 'submission_ready': True, 'upload_ready': True,
              'competition_upload_performed': False, 'rows': 529390, 'submission': output.relative_to(root).as_posix(),
              'sha256': digest(output), 'run_manifest_sha256': digest(work / 'run-manifest.json'),
              'decision_sha256': digest(work / 'promotion-decision.json'), 'accounting_record': ledger.relative_to(work).as_posix(),
              'accounting_sha256': digest(ledger), 'historical_v6_v7_reproduced': False, 'penalty': m['penalty']}
    prior.write_same(root / 'configs/v7_6.json', record); prior.write_same(output.with_suffix('.csv.receipt.json'), record)
    checksum = output.with_suffix('.csv.sha256'); text = record['sha256'] + '  ' + output.name + '\n'
    if checksum.exists(): require(checksum.read_text() == text, 'Checksum sidecar differs')
    else:
        with checksum.open('x') as stream: stream.write(text)
    prior.write_same(work / 'readiness.json', record)


def verify(root, work):
    if (work / 'selection.json').exists(): selection = checked(root, work, 'selection')
    else: return {'v7_6_promoted': False, 'status': 'preparing'}
    for stage in ('develop', 'screen', 'stress'):
        if (work / 'stages' / (stage + '.json')).exists(): checked(root, work, stage)
    if not (work / 'readiness.json').exists(): return {'v7_6_promoted': False, 'selection_passed': selection['passed'], 'status': 'unpromoted'}
    require_gates(root, work); frozen(root, work)
    require(completed(root, work, 'verify-release'), 'Promotion lacks accounted verification')
    receipt = read(work / 'readiness.json'); csv = inside(root, receipt['submission'])
    require(receipt == read(root / 'configs/v7_6.json') == read(csv.with_suffix('.csv.receipt.json'))
            and digest(csv) == receipt['sha256'] and receipt['promotion_supported'] is True
            and digest(inside(work, receipt['accounting_record'])) == receipt['accounting_sha256']
            and csv.with_suffix('.csv.sha256').read_text() == receipt['sha256'] + '  ' + csv.name + '\n', 'Published release changed')
    for number in (1, 2, 3): exported(root, work, number)
    return {'v7_6_promoted': True, 'submission': receipt['submission'], 'sha256': receipt['sha256']}


def report(root, work):
    pointer = work / 'terminal-accounting.json'
    if pointer.exists():
        record = read(pointer); accounting = inside(work, record['path'])
        require(digest(accounting) == record['sha256'], 'Terminal accounting seal changed')
    else:
        accounting = seal(root, work)
        write_new(pointer, {'path': accounting.relative_to(work).as_posix(), 'sha256': digest(accounting)})
    results = {'schema_version': 1, 'scientific_source': read(work / 'scientific-source.json')['source']['commit'],
               'protocol': SPEC, 'manifest_sha256': digest(work / 'manifest.json'), 'status': verify(root, work),
               'selection': read(work / 'selection.json') if (work / 'selection.json').exists() else None,
               'stages': {s: read(work / 'stages' / (s + '.json')) for s in ('develop', 'screen', 'stress') if (work / 'stages' / (s + '.json')).exists()},
               'accounting_sha256': digest(accounting), 'cumulative_seconds': read(accounting)['used_seconds'],
               'original_fallback_sha256': '9320ca6465c5ce6576746c5fc8f00d13109e507d7a20d5d506a9e03f63fa2b85',
               'competition_upload_performed': False}
    output = root / 'docs/results/v7.6-anchored-residual-v2'
    prior.write_same(output / 'results.json', results); prior.write_same(output / 'accounting.json', read(accounting))


def worker(root, work, claim):
    action, args = claim['action'], claim['arguments']
    if action == 'preflight': return preflight(root, work, claim)
    require(completed(root, work, 'preflight'), 'Worker preflight prerequisite missing')
    dataset.verify(root / 'artifacts/rebuild/data')
    raw = dataset.load(root / 'artifacts/rebuild/data', 'train')
    if action == 'feature':
        d, begin, end, member = (args[k] for k in ('d', 'begin', 'end', 'member'))
        require(d in PREFIXES and member in MEMBERS and d <= begin < end <= FULL, 'Undeclared feature role')
        assets.features(root, work, raw, d, begin, end, member)
        p = assets.feature_path(work, d, begin, end, member); return assets.hashes(work, [p, p.with_suffix('.receipt.json')])
    if action == 'combiner':
        require(args['c'] in PREFIXES and args['member'] in MEMBERS, 'Undeclared anchor role')
        return assets.fit_combiner(root, work, raw, args['c'], args['end'], args['member'])
    if action == 'view':
        c, d, e, b, end = (args[k] for k in ('c', 'd', 'fit_end', 'begin', 'end'))
        require((c, d, e) in pairs(b) and end == b + 6 * HOUR, 'Undeclared forecast view')
        return assets.save_view(root, work, raw, c, d, e, b, end)
    if action in ('innerfit', 'outerfit'): return fit_residual(root, work, raw, args, claim, action == 'innerfit')
    if action == 'selection': return select(root, work, raw)
    if action == 'reference': selected(root, work); return reference(root, work, raw, args['case'])
    if action == 'freeze': return freeze(root, work, claim)
    if action == 'score': return score(root, work, raw, args['case'])
    if action == 'summarize': return summarize(root, work, args['stage'])
    if action == 'export': return export(root, work, raw, args['number'], claim)
    if action == 'verify-release': return verify_release(root, work)
    raise ValueError('Unknown V7.6 worker action')


def controller(root, work, action):
    from runner import launch
    if action == 'verify': print(verify(root, work)); return
    if action == 'replay':
        require_gates(root, work)
        number = 1 + max((int(p.stem) for p in (work / 'exports').glob('*.json')), default=0)
        require(number >= 3, 'Standalone replay follows two independent staged exports')
        launch(root, 'export', 'export-' + str(number), {'number': number}, final=True)
        print('fresh-replay-complete', number); return
    if action == 'run':
        for stage in ('preflight', 'prepare-inner', 'select', 'prepare-outer', 'freeze', 'develop', 'screen', 'stress', 'finalize'):
            controller(root, work, stage)
        return
    if action == 'preflight': launch(root, 'preflight', 'preflight'); return
    require(completed(root, work, 'preflight'), 'Preflight must complete first')
    if action == 'prepare-inner':
        for b in BLOCKS:
            if b <= INNER[-1]: prepare_block(root, work, b)
    elif action == 'select':
        if not completed(root, work, 'select'):
            for b in INNER:
                for penalty in PENALTIES:
                    for fold in range(5): launch(root, 'innerfit', f'innerfit-{b}-{penalty:g}-{fold}',
                                                {'boundary': b, 'penalty': penalty, 'fold': fold}, limit=900)
            launch(root, 'selection', 'select')
        if not checked(root, work, 'selection')['passed']: report(root, work)
        selected(root, work)
    elif action == 'prepare-outer':
        penalty = selected(root, work)
        for b in BLOCKS: prepare_block(root, work, b)
        for boundary in BOUNDARIES: launch(root, 'outerfit', 'outerfit-' + str(boundary), {'boundary': boundary, 'penalty': penalty}, limit=900)
        for name in CASES: prepare_case(root, work, name)
    elif action == 'freeze': selected(root, work); launch(root, 'freeze', 'freeze')
    elif action in ('develop', 'screen', 'stress'):
        selected(root, work)
        if action != 'develop': require(checked(root, work, 'develop')['passed'], 'Development failure is terminal')
        if action == 'stress': require(checked(root, work, 'screen')['passed'], 'Screening failure is terminal')
        if not completed(root, work, action):
            names = {'develop': ('jan22_long', 'jan24_forward'), 'screen': ('primary', 'secondary'), 'stress': ('five_day',)}[action]
            for name in names: launch(root, 'score', 'score-' + name, {'case': name})
            launch(root, 'summarize', action, {'stage': action})
        if not checked(root, work, action)['passed']: report(root, work)
        require(checked(root, work, action)['passed'], 'V7.6 gate failed; candidate closed and fallback preserved')
    elif action == 'finalize':
        if (work / 'readiness.json').exists(): print(verify(root, work)); return
        require_gates(root, work)
        for number in (1, 2): launch(root, 'export', 'export-' + str(number), {'number': number}, final=True)
        controller(root, work, 'replay') if not completed(root, work, 'export-3') else None
        evidence(root, work); launch(root, 'verify-release', 'verify-release', final=True)
        publish(root, work); report(root, work); print(verify(root, work))
    else: raise ValueError('Unknown public V7.6 action')
