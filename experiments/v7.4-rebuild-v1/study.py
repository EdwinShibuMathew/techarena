"""Schema-five reconstructed lineage: fixed recipes, dual gates, accounted release."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import os
import sys

import numpy as np

import data as dataset
from recipe import CASES, CUTOFFS, DAY, FULL, JAN20, MEMBERS, COMBINER, RECIPE, TOTAL_SECONDS
from ml_workflow.contracts import canonical, compose, context, digest, first_indices, inside, require
from ml_workflow.storage import read, save_arrays, write_new, verify_snapshot
from ml_workflow.cache import arrays_identity
from ml_workflow import engine, policy
from predict_v7 import calibrate_v7_probabilities

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = 'experiments/v7.4-history-graph-v1/adapter.py'


def adapter():
    spec = importlib.util.spec_from_file_location('rebuild_graph', ROOT / ADAPTER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def protocol():
    module = adapter()
    return {'schema_version': 5, 'baseline_lineage': 'rebuilt_v6_v7_recipe',
            'release_policy': 'v6_v7_four_gate', 'candidate_order': RECIPE['candidate_order'],
            'candidates': module.CANDIDATES,
            'development': [{'name': name, 'meta_cutoff': CASES[name][0],
                             'target_cutoff': CASES[name][1], 'target_end': CASES[name][2]}
                            for name in ('jan22_long', 'jan24_forward')]}


def bundle(work, path, identity, compute):
    """A cache hit requires identical inputs, source, runtime and output bytes."""
    receipt = path.with_suffix('.receipt.json')
    if path.exists() or receipt.exists():
        require(path.is_file() and receipt.is_file(), 'Partial cache; preserve evidence')
        record = read(receipt)
        require(record['identity'] == identity and record['sha256'] == digest(path), 'Cache provenance changed')
    else:
        save_arrays(path, **compute())
        write_new(receipt, {'identity': identity, 'sha256': digest(path)})
    with np.load(path, allow_pickle=False) as saved: values = {k: saved[k] for k in saved.files}
    return values


def identity(work, kind, values):
    return {'schema_version': 5, 'kind': kind, 'genesis_sha256': digest(work / 'genesis.json'),
            'scientific_source_sha256': digest(work / 'scientific-source.json'), **values}


def files(work, paths):
    return {p.relative_to(work).as_posix(): digest(p) for p in paths}


def feature_key(cutoff, begin, end, member, evaluation=False):
    return f'{cutoff}-{begin}-{end}-{member}' + ('-evaluation' if evaluation else '')


def features(work, raw, cutoff, begin, end, member, evaluation=False, fresh=False):
    import torch
    from build_submission import neural_pipeline
    from predict_v6 import portable_pipeline
    from run import completed
    require(completed(work, f'train-{cutoff}-{member}'), 'Verified prefix checkpoint required')
    target_raw = dataset.load(work / 'data', 'evaluation') if evaluation else raw
    index = np.arange(len(target_raw['u'])) if evaluation else first_indices(target_raw['u'], target_raw['t'], begin, end)
    require(len(index) and (target_raw['t'][index] >= cutoff).all(), 'Feature target precedes fitting cutoff')
    folder = work / 'baselines' / str(cutoff) / member
    checkpoint, positive = folder / 'event.pt', folder / 'positive.npz'
    expected = identity(work, 'features', {'checkpoint_sha256': digest(checkpoint), 'positive_sha256': digest(positive),
                         'data_sha256': digest(work / 'data/metadata.json'), 'member': member,
                         'cutoff': cutoff, 'begin': begin, 'end': end, 'evaluation': evaluation,
                         'indices': arrays_identity({'indices': index})})
    def compute():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(state['purpose'] == 'research' and state['epoch'] == 8 and state['stage'] == 'event'
                and state['cutoff'] == cutoff and state['dimension'] == MEMBERS[member][0]
                and state['positive_sha256'] == digest(positive)
                and state['preference_source_sha256'] == digest(folder / 'preference.pt'), 'Invalid reconstructed checkpoint')
        fitting = context(raw, np.flatnonzero(raw['t'] < cutoff), True)
        target = context(target_raw, index, False)
        # rowids are for alignment only, never neural or CatBoost input.
        fitting.pop('rowids'); target.pop('rowids')
        X, names, raw_probs, probs, _, _ = portable_pipeline(neural_pipeline)(checkpoint, fitting, target, positive, torch.device('cuda'))
        require(X.shape == (len(index) * 10, 37) and X.dtype == np.float32 and np.isfinite(X).all(), 'Invalid 37-feature schema')
        return {'X': X, 'feature_names': np.asarray(names), 'indices': index, 'probs': probs, 'raw_probs': raw_probs}
    if fresh: return compute()
    return bundle(work, work / 'features' / (feature_key(cutoff, begin, end, member, evaluation) + '.npz'), expected, compute)


def load_features(work, cutoff, begin, end, member, evaluation=False):
    path = work / 'features' / (feature_key(cutoff, begin, end, member, evaluation) + '.npz')
    require(read(path.with_suffix('.receipt.json'))['sha256'] == digest(path), 'Features changed')
    with np.load(path, allow_pickle=False) as saved: return {k: saved[k] for k in saved.files}


def fit_combiner(work, raw, cutoff, member, crossfit=False):
    from catboost import CatBoostClassifier, Pool
    matrix = load_features(work, cutoff, cutoff, cutoff + DAY, member)
    ix, names, X = matrix['indices'], matrix['feature_names'].tolist(), matrix['X']
    labels = raw['y'][ix]
    folds = adapter().user_folds(raw['u'][ix]) if crossfit else np.full(len(ix), -1)
    require(not crossfit or set(folds) == set(range(5)), 'Incomplete user folds')
    models = work / ('oof-models' if crossfit else 'combiners') / str(cutoff)
    scores = np.empty((len(ix), 10), dtype=np.float64)
    outputs = []
    for fold in (range(5) if crossfit else [-1]):
        fitting = folds != fold if crossfit else np.ones(len(ix), dtype=bool)
        held = ~fitting if crossfit else fitting
        require(not crossfit or not np.intersect1d(raw['u'][ix][fitting], raw['u'][ix][held]).size, 'Cross-fit user leakage')
        path = models / (member + (f'-{fold}' if crossfit else '') + '.cbm'); path.parent.mkdir(parents=True, exist_ok=True)
        receipt = path.with_suffix('.receipt.json')
        expected = identity(work, 'combiner', {'cutoff': cutoff, 'member': member, 'fold': fold, 'crossfit': crossfit,
                            'parameters': COMBINER, 'features': arrays_identity(matrix),
                            'labels': arrays_identity({'labels': labels, 'folds': folds})})
        if path.exists() or receipt.exists():
            require(path.is_file() and receipt.is_file() and read(receipt)['identity'] == expected
                    and read(receipt)['sha256'] == digest(path), 'Combiner provenance changed')
            model = CatBoostClassifier(); model.load_model(str(path))
        else:
            model = CatBoostClassifier(**deepcopy(COMBINER), verbose=False)
            model.fit(Pool(X[np.repeat(fitting, 10)], label=labels[fitting].ravel(), feature_names=names))
            stage = path.with_suffix('.staging'); model.save_model(str(stage)); os.link(stage, path); stage.unlink()
            write_new(receipt, {'identity': expected, 'sha256': digest(path)})
        require(model.tree_count_ == 545 and model.feature_names_ == names, 'Combiner recipe differs')
        scores[held] = model.predict_proba(Pool(X[np.repeat(held, 10)], feature_names=names), thread_count=4)[:, 1].reshape(-1, 10)
        outputs.extend([path, receipt]); print('combiner', cutoff, member, fold, flush=True)
    if crossfit:
        output = work / 'oof' / str(cutoff) / (member + '.npz')
        expected = identity(work, 'oof', {'models': files(work, outputs), 'folds': arrays_identity({'folds': folds})})
        bundle(work, output, expected, lambda: {'scores': scores, 'indices': ix, 'folds': folds})
        outputs.extend([output, output.with_suffix('.receipt.json')])
    return files(work, outputs)


def references(work, raw, case):
    from build_submission import load_combiner, predict_combiner
    meta, target, end = CASES[case]
    fitting, predictions = {}, {}
    fit_ix = first_indices(raw['u'], raw['t'], meta, meta + DAY)
    valid = first_indices(raw['u'], raw['t'], target, end)
    dependencies = {}
    for member in MEMBERS:
        path = work / 'oof' / str(meta) / (member + '.npz')
        require(read(path.with_suffix('.receipt.json'))['sha256'] == digest(path), 'OOF changed')
        with np.load(path, allow_pickle=False) as saved:
            require(np.array_equal(saved['indices'], fit_ix), 'OOF event order differs')
            fitting[member] = saved['scores']
        matrix = load_features(work, target, target, end, member)
        require(np.array_equal(matrix['indices'], valid), 'Target event order differs')
        model = work / 'combiners' / str(meta) / (member + '.cbm')
        require(read(model.with_suffix('.receipt.json'))['sha256'] == digest(model), 'Reference combiner changed')
        predictions[member] = predict_combiner(load_combiner(model), matrix['X'], matrix['feature_names'].tolist()).reshape(-1, 10)
        dependencies.update(files(work, [path, model]))
    return fit_ix, valid, compose(fitting), compose(predictions), dependencies


def graph(work, raw, cutoff, fresh=False):
    module = adapter()
    expected = identity(work, 'graph', {'data_sha256': digest(work / 'data/metadata.json'),
                                      'cutoff': cutoff, 'parameters': module.CONFIG})
    compute = lambda: module.build_graph(context(raw, np.flatnonzero(raw['t'] < cutoff), True), cutoff)
    values = compute() if fresh else bundle(work, work / 'graphs' / (str(cutoff) + '.npz'), expected, compute)
    module.validate_graph(values); require(int(values['cutoff']) == cutoff, 'Graph cutoff differs')
    return values


def score(work, raw, case, candidate):
    module = adapter(); params = deepcopy(module.CANDIDATES[candidate])
    meta, target, end = CASES[case]
    fit, valid, oof, before, dependencies = references(work, raw, case)
    if case in ('primary', 'secondary', 'five_day'):
        require(len(valid) == {'primary': 43964, 'secondary': 35138, 'five_day': 51488}[case], 'Official cohort differs')
    fit_features = module.affinities(graph(work, raw, meta), context(raw, fit, False), params['half_life_hours'])
    anchor_fit = calibrate_v7_probabilities(oof, 1.12)
    metadata = {'spec_sha256': canonical(protocol()), 'candidate': candidate, 'parameters_sha256': canonical(params),
                'meta_cutoff': meta, 'maximum_label_timestamp': int(raw['t'][fit].max()), 'fitting_events': len(fit),
                'fitting_identity': canonical(arrays_identity({'anchor': anchor_fit, 'features': fit_features,
                                                              'labels': raw['y'][fit], 'users': raw['u'][fit]})),
                'graph_identity': digest(work / 'graphs' / (str(meta) + '.npz')),
                'genesis_sha256': digest(work / 'genesis.json'),
                'baseline_lineage': 'rebuilt_v6_v7_recipe'}
    calibrator = work / 'calibrators' / candidate / (str(meta) + '.json')
    if calibrator.exists(): saved = module.load_calibrator(calibrator, deepcopy(metadata), deepcopy(params))
    else:
        calibrator.parent.mkdir(parents=True, exist_ok=True)
        saved = module.fit_calibrator(anchor_fit.copy(), fit_features.copy(), raw['y'][fit].copy(), deepcopy(metadata), calibrator, deepcopy(params))
    target_features = module.affinities(graph(work, raw, target), context(raw, valid, False), params['half_life_hours'])
    anchor = calibrate_v7_probabilities(before, 1.12)
    after = module.predict_calibrator(anchor.copy(), target_features.copy(), deepcopy(saved))
    require(after.shape == before.shape and np.isfinite(after).all() and ((after >= 0) & (after <= 1)).all()
            and np.allclose(after.sum(1), anchor.sum(1), rtol=0, atol=1e-14), 'Invalid candidate or event mass')
    prediction = work / 'predictions' / (candidate + '-' + case + '.npz')
    expected = identity(work, 'prediction', {'case': case, 'candidate': candidate, 'calibrator_sha256': digest(calibrator),
                                           'references': dependencies})
    bundle(work, prediction, expected, lambda: {'indices': valid, 'rowids': raw['rowids'][valid], 'labels': raw['y'][valid],
                                               'baseline': before, 'v7': anchor, 'candidate': after})
    # Receipt creation is deterministic; a completed case is never scientifically retried.
    row = metrics(raw, valid, target, before, anchor, after, case)
    row.update({'case': case, 'candidate': candidate, 'alpha': saved['alpha'], 'calibrator': calibrator.relative_to(work).as_posix(),
                'calibrator_metadata': metadata, 'parameters': params, 'prediction_sha256': digest(prediction),
                'baseline_lineage': 'rebuilt_v6_v7_recipe', 'fitting_events': len(fit),
                'references': dependencies, 'calibrator_sha256': digest(calibrator)})
    output = work / 'results' / (candidate + '-' + case + '.json')
    write_new(output, row)
    print('gate-input', case, candidate, 'vs rebuilt V6', row['auc_diff'], 'vs rebuilt V7', row['v7_comparison']['auc_diff'], 'alpha', saved['alpha'], flush=True)
    return files(work, [output, prediction, prediction.with_suffix('.receipt.json'), calibrator])


def metrics(raw, valid, cutoff, before, anchor, after, case):
    y = raw['y'][valid]
    row = {'events': len(valid), **engine.point(y, before, after), 'v7_comparison': engine.point(y, anchor, after)}
    if case in ('primary', 'secondary', 'five_day'):
        for result, comparator in ((row, before), (row['v7_comparison'], anchor)):
            result.update(asdict(engine.PairedBootstrapEvaluator().evaluate(after, comparator, y, raw['u'][valid])))
            policy.validate_metric(result, confidence=True)
    age = (raw['t'][valid] - cutoff) / 3600
    late = (age >= 48) & (age < 120)
    if late.any():
        row['late'] = {'events': int(late.sum()), **engine.point(y[late], before[late], after[late])}
        row['v7_comparison']['late'] = {'events': int(late.sum()), **engine.point(y[late], anchor[late], after[late])}
    if case == 'five_day': require(int(late.sum()) == 7596, 'Official late cohort differs')
    return row


def checked(work, stage):
    from run import completed
    require(completed(work, stage), 'Stage must complete with durable accounting: ' + stage)
    report = read(work / 'stages' / (stage + '.json'))
    require(report['genesis_sha256'] == digest(work / 'genesis.json')
            and report['protocol_sha256'] == canonical(protocol()), 'Scientific generation changed')
    for name, expected in report['files'].items(): require(digest(inside(work, name)) == expected, 'Stage evidence changed')
    return report


def summarize(work, stage):
    results, paths = {}, []
    if stage == 'develop':
        for candidate in RECIPE['candidate_order']:
            for case in ('jan22_long', 'jan24_forward'):
                path = work / 'results' / (candidate + '-' + case + '.json')
                results[candidate + '/' + case] = read(path); paths.append(path)
        winner = policy.dual_winner(protocol(), results); passed = winner is not None
    else:
        previous = checked(work, 'develop'); winner = previous['winner']
        require(previous['passed'] and winner in RECIPE['candidate_order'], 'Development failure is terminal')
        if stage == 'stress': require(checked(work, 'screen')['passed'], 'Screen failure is terminal')
        for case in (('primary', 'secondary') if stage == 'screen' else ('five_day',)):
            path = work / 'results' / (winner + '-' + case + '.json'); results[case] = read(path); paths.append(path)
        passed = policy.stage_passed(protocol(), stage, results)
    output = work / 'stages' / (stage + '.json')
    write_new(output, {'schema_version': 5, 'baseline_lineage': 'rebuilt_v6_v7_recipe',
                       'genesis_sha256': digest(work / 'genesis.json'), 'protocol_sha256': canonical(protocol()),
                       'winner': winner, 'passed': bool(passed), 'results': results, 'files': files(work, paths),
                       'validation_status': 'retrospective reused dates; not independent holdout evidence'})
    print('stage', stage, 'passed', passed, 'winner', winner, flush=True)
    return files(work, [output])


def verify_gates(work):
    development, screen, stress = [checked(work, stage) for stage in ('develop', 'screen', 'stress')]
    winner = policy.dual_winner(protocol(), development['results'])
    require(winner is not None and all(r['passed'] and r['winner'] == winner for r in (development, screen, stress)),
            'All stages must pass for the same predeclared winner')
    gates = policy.dual_gates(screen, stress)
    require(all(value for group in gates.values() for value in group.values()), 'All eight comparator checks must pass')
    return winner, gates


def export(work, root, raw, number):
    """Fresh neural extraction and graph reconstruction on every independent run."""
    from build_submission import load_combiner, predict_combiner
    from submission_export import export_submission, validate_csv
    from row_auc_final_utils import verify_csv_probabilities
    winner, gates = verify_gates(work)
    secondary = checked(work, 'screen')['results']['secondary']; module = adapter()
    calibrator = inside(work, secondary['calibrator'])
    require(digest(calibrator) == secondary['calibrator_sha256'], 'Final calibrator changed')
    state = module.load_calibrator(calibrator, deepcopy(secondary['calibrator_metadata']), deepcopy(secondary['parameters']))
    require(state['metadata']['meta_cutoff'] == JAN20 + 5 * DAY
            and state['metadata']['maximum_label_timestamp'] < JAN20 + 6 * DAY, 'Final calibrator chronology differs')
    evaluation = dataset.load(work / 'data', 'evaluation'); members = {}; dependencies = {}
    for member in MEMBERS:
        matrix = features(work, raw, FULL, FULL, FULL + 7 * DAY, member, evaluation=True, fresh=True)
        model = work / 'combiners' / str(JAN20 + 5 * DAY) / (member + '.cbm')
        require(read(model.with_suffix('.receipt.json'))['sha256'] == digest(model), 'Final reference combiner changed')
        members[member] = predict_combiner(load_combiner(model), matrix['X'], matrix['feature_names'].tolist()).reshape(-1, 10)
        dependencies.update(files(work, [model, work / 'baselines' / str(FULL) / member / 'event.pt']))
    baseline = compose(members); anchor = calibrate_v7_probabilities(baseline, 1.12)
    evidence = module.affinities(graph(work, raw, FULL, fresh=True), evaluation, secondary['parameters']['half_life_hours'])
    after = module.predict_calibrator(anchor.copy(), evidence, deepcopy(state))
    require(np.allclose(after.sum(1), anchor.sum(1), rtol=0, atol=1e-14), 'Final correction changed event mass')
    folder = work / 'exports' / str(number); folder.mkdir(parents=True, exist_ok=False)
    ids = evaluation['rowids'].ravel(); example = root / 'data/raw/example_submission.csv'
    outputs = []
    for name, p in [('rebuilt_v6', baseline), ('rebuilt_v7', anchor), ('v7_4', after)]:
        output = folder / (name + '.csv')
        export_submission(ids, p, example, output, 529390); validate_csv(output, example, 529390)
        verify_csv_probabilities(output, ids, p); outputs.append(output)
        if number > 1: require(digest(output) == digest(work / 'exports/1' / output.name), 'Independent replay differs')
    vectors = folder / 'predictions.npz'
    save_arrays(vectors, rowids=evaluation['rowids'], members=np.stack(list(members.values())),
                rebuilt_v6=baseline, rebuilt_v7=anchor, candidate=after)
    outputs.append(vectors)
    receipt = folder / 'validation.json'
    write_new(receipt, {'schema_version': 5, 'baseline_lineage': 'rebuilt_v6_v7_recipe', 'winner': winner,
                        'gates': gates, 'genesis_sha256': digest(work / 'genesis.json'), 'files': files(work, outputs),
                        'dependencies': dependencies, 'calibrator_sha256': digest(calibrator), 'fresh_graph': True,
                        'fresh_neural_features': True, 'independent_replay': number > 1})
    return files(work, outputs + [receipt])


def worker(root, work, claim):
    dataset.verify(work / 'data')
    raw = dataset.load(work / 'data', 'train'); action, args = claim['action'], claim['arguments']
    if action == 'feature':
        cutoff, begin, end, member = [args[k] for k in ('cutoff', 'begin', 'end', 'member')]
        features(work, raw, cutoff, begin, end, member)
        path = work / 'features' / (feature_key(cutoff, begin, end, member) + '.npz')
        return files(work, [path, path.with_suffix('.receipt.json')])
    if action in ('combiner', 'crossfit'):
        return fit_combiner(work, raw, args['cutoff'], args['member'], action == 'crossfit')
    if action == 'score': return score(work, raw, args['case'], args['candidate'])
    if action == 'summarize': return summarize(work, args['stage'])
    if action == 'export': return export(work, root, raw, args['number'])
    raise ValueError('Unsupported scientific worker')


def prepare_case(root, work, case, launch):
    from run import completed
    meta, target, end = CASES[case]
    for member in MEMBERS:
        for cutoff, begin, stop in [(meta, meta, meta + DAY), (target, target, end)]:
            key = feature_key(cutoff, begin, stop, member)
            launch(root, 'feature', 'feature-' + key, 1200, dict(cutoff=cutoff, begin=begin, end=stop, member=member))
        for action in ('combiner', 'crossfit'):
            launch(root, action, f'{action}-{meta}-{member}', 1200, dict(cutoff=meta, member=member))


def controller(root, action, launch):
    from run import workspace, completed
    work = workspace(root)
    if action in ('develop', 'screen', 'stress'):
        if completed(work, action):
            report = checked(work, action); require(report['passed'], 'Recorded gate failure is terminal'); return
        if action != 'develop': require(checked(work, 'develop')['passed'], 'Development failure is terminal')
        if action == 'stress': require(checked(work, 'screen')['passed'], 'Screen failure is terminal')
        candidates = RECIPE['candidate_order'] if action == 'develop' else [checked(work, 'develop')['winner']]
        cases = {'develop': ('jan22_long', 'jan24_forward'), 'screen': ('primary', 'secondary'), 'stress': ('five_day',)}[action]
        for case in cases:
            prepare_case(root, work, case, launch)
            for candidate in candidates:
                launch(root, 'score', f'score-{candidate}-{case}', 1200, dict(case=case, candidate=candidate))
        launch(root, 'summarize', action, 1200, {'stage': action})
        require(checked(work, action)['passed'], 'Gate failed; no further training or promotion permitted')
    elif action == 'finalize':
        verify_gates(work)
        for member in MEMBERS:
            launch(root, 'train', f'train-{FULL}-{member}', 3600, dict(cutoff=FULL, member=member))
        for number in (1, 2): launch(root, 'export', f'export-{number}', 1200, {'number': number})
        publish(root, work)
    elif action == 'replay':
        require((work / 'readiness.json').exists(), 'Publish an accounted release first')
        number = 3
        launch(root, 'export', f'export-{number}', 1200, {'number': number})
    else: raise ValueError('Unknown controller action')


def accounting(work):
    ledger = read(work / 'compute-ledger.json')
    genesis = read(work / 'genesis.json')
    require(ledger['total_seconds'] == TOTAL_SECONDS and ledger['carry_identity'] == digest(work / 'genesis.json')
            and genesis['recipe'] == RECIPE and 0 <= ledger['used_seconds'] <= TOTAL_SECONDS
            and not any(a['status'] == 'running' for a in ledger['attempts'])
            and abs(sum(a['seconds'] for a in ledger['attempts']) - ledger['used_seconds']) < 1e-6,
            'Final accounting must be durable, reconciled and within the approved cap')
    return ledger


def publish(root, work):
    from run import completed
    winner, gates = verify_gates(work)
    for number in (1, 2): require(completed(work, f'export-{number}'), 'Both independently accounted exports required')
    ledger = accounting(work)
    outputs = files(work, [work / 'exports/1' / (name + '.csv') for name in ('rebuilt_v6', 'rebuilt_v7', 'v7_4')])
    for name in ('rebuilt_v6', 'rebuilt_v7', 'v7_4'):
        require(digest(work / 'exports/1' / (name + '.csv')) == digest(work / 'exports/2' / (name + '.csv')), 'Replay fingerprint differs')
    config = {'schema_version': 5, 'release_name': 'v7_4', 'baseline_lineage': 'rebuilt_v6_v7_recipe',
              'historical_v6_v7_reproduced': False, 'submission_ready': True, 'promotion_supported': True,
              'release_policy': 'v6_v7_four_gate', 'winner': winner, 'gates': gates,
              'genesis_sha256': digest(work / 'genesis.json'), 'protocol_sha256': canonical(protocol()),
              'submission_sha256': outputs['exports/1/v7_4.csv'],
              'rebuilt_baseline_sha256': {'V6': outputs['exports/1/rebuilt_v6.csv'], 'V7': outputs['exports/1/rebuilt_v7.csv']},
              'sharpening_gamma': 1.12, 'validation_status': 'retrospective reused dates; not independent holdout evidence',
              'compute_seconds_at_publication': ledger['used_seconds'],
              'replay_command': 'python experiments/v7.4-rebuild-v1/run.py replay'}
    path = root / 'configs/v7_4.json'
    if path.exists(): require(read(path) == config, 'Preserve a different release configuration')
    else: write_new(path, config)
    output = root / 'artifacts/submission_v7_4.csv'
    if output.exists(): require(digest(output) == config['submission_sha256'], 'Preserve different submission bytes')
    else: os.link(work / 'exports/1/v7_4.csv', output)
    receipt = {'schema_version': 5, **config, 'config_sha256': digest(path), 'files': outputs,
               'accounting_sha256_at_publication': digest(work / 'compute-ledger.json')}
    if (work / 'readiness.json').exists(): require(read(work / 'readiness.json') == receipt, 'Preserve different readiness record')
    else: write_new(work / 'readiness.json', receipt)


def verify(root, work):
    winner, gates = verify_gates(work); ledger = accounting(work)
    receipt = read(work / 'readiness.json')
    require(receipt['submission_ready'] is True and receipt['winner'] == winner and receipt['gates'] == gates
            and digest(root / 'configs/v7_4.json') == receipt['config_sha256']
            and digest(root / 'artifacts/submission_v7_4.csv') == receipt['submission_sha256'], 'Release evidence changed')
    for name, expected in receipt['files'].items(): require(digest(inside(work, name)) == expected, 'Release output changed')
    return {'verified': True, 'baseline_lineage': 'rebuilt_v6_v7_recipe', 'gates': gates, 'used_seconds': ledger['used_seconds'],
            'submission_sha256': receipt['submission_sha256'], 'historical_v6_v7_reproduced': False}
