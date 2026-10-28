"""Causal asset bridge and independent competition inference for V7.5."""
from copy import deepcopy
from pathlib import Path
import os
import time
import inspect
import importlib.util
from dataclasses import asdict

import numpy as np

import data as dataset
from recipe import RECIPE
from protocol import MEMBERS, COMBINER, DAY, JAN20, FULL, PROTOCOL, PREFIXES, EXTRA_CUTOFFS, CANDIDATES, ADDED_NAMES, CASES, fitting_prefixes
from ml_workflow.contracts import canonical, compose, context, digest, first_indices, inside, require
from ml_workflow.storage import read, write_new, save_arrays
from ml_workflow.cache import arrays_identity
from predict_v7 import calibrate_v7_probabilities
from ml_workflow import engine, policy

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = 'experiments/v7.4-history-graph-v1/adapter.py'


def hashes(work, paths): return {p.relative_to(work).as_posix(): digest(p) for p in paths}


def write_same(path, value):
    if path.exists(): require(read(path) == value, 'Technical retry changed frozen evidence')
    else: write_new(path, value)


def cache(work, path, identity, compute):
    receipt = path.with_suffix('.receipt.json')
    if path.exists() or receipt.exists():
        require(path.is_file() and receipt.is_file() and read(receipt)['identity'] == identity
                and read(receipt)['sha256'] == digest(path), 'Cache identity or bytes changed')
    else:
        save_arrays(path, **compute()); write_new(receipt, {'identity': identity, 'sha256': digest(path)})
    with np.load(path, allow_pickle=False) as saved: return {k: saved[k] for k in saved.files}


def model_paths(root, work, cutoff, member):
    from campaign import completed
    parent = root / 'artifacts/rebuild'; m = read(work / 'manifest.json')
    folder = work / 'models' / str(cutoff) / member
    if folder.exists():
        require(completed(work, f'train-{cutoff}-{member}'), 'Require an accounted new checkpoint')
    else:
        folder = parent / 'baselines' / str(cutoff) / member
        for name in ('event.pt', 'preference.pt', 'positive.npz'):
            relative = (folder / name).relative_to(root).as_posix()
            require(relative in m['inherited_models'] and digest(folder / name) == m['inherited_models'][relative], 'Inherited checkpoint changed')
    return folder


def feature_path(work, cutoff, begin, end, member):
    return work / 'features37' / f'{cutoff}-{begin}-{end}-{member}.npz'


def features37(root, work, raw, cutoff, begin, end, member, evaluation=False, fresh=False):
    import torch
    from build_submission import neural_pipeline
    from predict_v6 import portable_pipeline
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    folder = model_paths(root, work, cutoff, member)
    target_raw = dataset.load(root / 'artifacts/rebuild/data', 'evaluation') if evaluation else raw
    ix = np.arange(len(target_raw['u'])) if evaluation else first_indices(target_raw['u'], target_raw['t'], begin, end)
    require(len(ix) and (target_raw['t'][ix] >= cutoff).all(), 'Feature target precedes checkpoint cutoff')
    identity = {'schema_version': 6, 'manifest_sha256': digest(work / 'manifest.json'), 'member': member,
                'cutoff': cutoff, 'begin': begin, 'end': end, 'evaluation': evaluation,
                'checkpoint_sha256': digest(folder / 'event.pt'), 'positive_sha256': digest(folder / 'positive.npz'),
                'indices': arrays_identity({'indices': ix}),
                'extractor': canonical({'functions': [inspect.getsource(features37), inspect.getsource(model_paths)],
                   'files': {p.relative_to(Path(__file__).resolve().parents[2]).as_posix(): digest(p)
                             for p in [*sorted((Path(__file__).resolve().parents[2] / 'legacy').glob('*.py')),
                                       Path(__file__).resolve().parents[2] / 'tools/predict_v6.py']}})}
    def compute():
        state = torch.load(folder / 'event.pt', map_location='cpu', weights_only=True)
        require(state['cutoff'] == cutoff and state['epoch'] == 8 and state['stage'] == 'event'
                and state['purpose'] == 'research' and state['dimension'] == MEMBERS[member][0]
                and state['seed'] == MEMBERS[member][1] and state['recipe_sha256'] == canonical(RECIPE)
                and state['positive_sha256'] == digest(folder / 'positive.npz')
                and state['preference_source_sha256'] == digest(folder / 'preference.pt'), 'Checkpoint provenance differs')
        fitting = context(raw, np.flatnonzero(raw['t'] < cutoff), True); fitting.pop('rowids')
        target = context(target_raw, ix, False); target.pop('rowids')
        X, names, raw_probs, probs, _, _ = portable_pipeline(neural_pipeline)(folder / 'event.pt', fitting, target, folder / 'positive.npz', torch.device('cuda'))
        require(X.shape == (len(ix) * 10, 37) and X.dtype == np.float32 and np.isfinite(X).all(), '37-feature schema differs')
        return {'X': X, 'feature_names': np.asarray(names), 'indices': ix, 'probs': probs, 'raw_probs': raw_probs}
    if fresh: return compute()
    return cache(work, feature_path(work, cutoff, begin, end, member), identity, compute)


def matrix(work, cutoff, begin, end, member):
    path = feature_path(work, cutoff, begin, end, member)
    require(read(path.with_suffix('.receipt.json'))['sha256'] == digest(path), 'Feature bytes changed')
    with np.load(path, allow_pickle=False) as saved: return {k: saved[k] for k in saved.files}


def fit_model(work, path, X, names, labels, identity, weights=None):
    from catboost import CatBoostClassifier, Pool
    receipt = path.with_suffix('.receipt.json'); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or receipt.exists():
        require(path.is_file() and receipt.is_file() and read(receipt)['identity'] == identity
                and read(receipt)['sha256'] == digest(path), 'Combiner provenance changed')
        model = CatBoostClassifier(); model.load_model(str(path))
    else:
        model = CatBoostClassifier(**deepcopy(COMBINER), verbose=False)
        model.fit(Pool(X, label=labels, weight=weights, feature_names=names))
        stage = path.with_suffix('.staging'); model.save_model(str(stage)); os.link(stage, path); stage.unlink()
        write_new(receipt, {'identity': identity, 'sha256': digest(path)})
    require(model.tree_count_ == 545 and model.feature_names_ == names, 'Tree count or feature schema differs')
    return model


def baseline_combiner(work, raw, member, meta):
    data = matrix(work, meta, meta, meta + DAY, member)
    path = work / 'baseline-combiners' / str(meta) / (member + '.cbm')
    identity = {'manifest_sha256': digest(work / 'manifest.json'), 'member': member, 'meta': meta,
                'features': arrays_identity(data), 'labels': arrays_identity({'y': raw['y'][data['indices']]}), 'parameters': COMBINER}
    fit_model(work, path, data['X'], data['feature_names'].tolist(), raw['y'][data['indices']].ravel(), identity)
    return hashes(work, [path, path.with_suffix('.receipt.json')])


def predict_model(path, X, names, width):
    from catboost import CatBoostClassifier, Pool
    require(len(names) == width and read(path.with_suffix('.receipt.json'))['sha256'] == digest(path), 'Model identity/schema changed')
    model = CatBoostClassifier(); model.load_model(str(path))
    require(model.tree_count_ == 545 and model.feature_names_ == names, 'Saved feature order differs')
    p = model.predict_proba(Pool(X, feature_names=names), thread_count=4)[:, 1]
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid combiner predictions')
    return p.reshape(-1, 10)


def baseline_export(root, work, raw, number):
    from submission_export import export_submission, validate_csv
    from row_auc_final_utils import verify_csv_probabilities
    members = {}; dependencies = {}
    for member in MEMBERS:
        data = features37(root, work, raw, FULL, FULL, FULL + 7 * DAY, member, evaluation=True, fresh=True)
        path = work / 'baseline-combiners' / str(JAN20 + 5 * DAY) / (member + '.cbm')
        members[member] = predict_model(path, data['X'], data['feature_names'].tolist(), 37)
        dependencies.update(hashes(work, [path, path.with_suffix('.receipt.json')]))
    evaluation = dataset.load(root / 'artifacts/rebuild/data', 'evaluation')
    before = compose(members); anchor = calibrate_v7_probabilities(before, 1.12)
    folder = work / 'baseline-exports' / ('attempt-' + str(number) + '-' + str(time.time_ns()))
    folder.mkdir(parents=True); final = work / 'baseline-exports' / str(number)
    vectors = folder / 'predictions.npz'
    save_arrays(vectors, rowids=evaluation['rowids'], members=np.stack(list(members.values())), rebuilt_v6=before, rebuilt_v7=anchor)
    outputs = [vectors]
    for name, p in [('rebuilt_v6', before), ('rebuilt_v7', anchor)]:
        path = folder / (name + '.csv'); example = root / 'data/raw/example_submission.csv'
        export_submission(evaluation['rowids'].ravel(), p, example, path, 529390)
        validate_csv(path, example, 529390); verify_csv_probabilities(path, evaluation['rowids'].ravel(), p)
        if number > 1: require(digest(path) == digest(work / 'baseline-exports/1' / path.name), 'Independent baseline CSV differs')
        outputs.append(path)
    if number > 1:
        with np.load(work / 'baseline-exports/1/predictions.npz', allow_pickle=False) as previous:
            require(np.allclose(np.stack(list(members.values())), previous['members'], rtol=0, atol=1e-15), 'Independent member replay differs')
    receipt = folder / 'validation.json'
    write_new(receipt, {'schema_version': 6, 'manifest_sha256': digest(work / 'manifest.json'), 'number': number,
                        'dependencies': dependencies, 'fresh_neural_features': True,
                        'files': {(final / p.name).relative_to(work).as_posix(): digest(p) for p in outputs}})
    if final.exists(): require(all(digest(final / p.name) == digest(p) for p in outputs + [receipt]), 'Technical baseline replay differs')
    else: folder.rename(final)
    return hashes(work, [final / p.name for p in outputs + [receipt]])


def publish_baseline(root, work):
    from campaign import completed
    for number in (1, 2): require(completed(work, f'baseline-export-{number}'), 'Two accounted independent baseline exports required')
    if (work / 'baseline-delivery.json').exists():
        previous = read(work / 'baseline-delivery.json')
        target = inside(root, previous['submission'])
        require(digest(target) == previous['sha256']
                and read(target.with_suffix('.csv.receipt.json')) == previous, 'Published baseline changed')
        return
    accounting_path = seal_accounting(root, work)
    source = work / 'baseline-exports/1/rebuilt_v6.csv'
    destination = root / 'artifacts/submissions/rebuilt_v6_submission.csv'; destination.parent.mkdir(parents=True, exist_ok=True)
    require(digest(source) == digest(work / 'baseline-exports/2/rebuilt_v6.csv'), 'Baseline export hashes differ')
    if destination.exists(): require(digest(destination) == digest(source), 'Preserve different submission bytes')
    else: os.link(source, destination)
    receipt = {'schema_version': 6, 'model_role': 'rebuilt_baseline', 'baseline_lineage': 'rebuilt_v6_v7_recipe',
               'csv_valid': True, 'replay_verified': True, 'promotion_supported': False, 'historical_v6_reproduced': False,
               'manifest_sha256': digest(work / 'manifest.json'), 'rows': 529390,
               'submission': destination.relative_to(root).as_posix(), 'sha256': digest(destination),
               'accounting_sha256_at_publication': digest(accounting_path),
               'accounting_record': accounting_path.relative_to(work).as_posix()}
    write_same(destination.with_suffix('.csv.receipt.json'), receipt)
    checksum = destination.with_suffix('.csv.sha256'); content = receipt['sha256'] + '  ' + destination.name + '\n'
    if checksum.exists(): require(checksum.read_text() == content, 'Different checksum sidecar')
    else:
        with checksum.open('x') as stream: stream.write(content)
    write_same(work / 'baseline-delivery.json', receipt)


def worker(root, work, claim):
    parent = root / 'artifacts/rebuild'; dataset.verify(parent / 'data')
    raw = dataset.load(parent / 'data', 'train'); action = claim['action']; args = claim['arguments']
    if action == 'train':
        import train
        from protocol import PREFIXES
        cutoff, member = args['cutoff'], args['member']
        require(cutoff in PREFIXES + [FULL] and member in MEMBERS, 'Undeclared model role')
        folder = work / 'models' / str(cutoff) / member
        identity = canonical({p: v for p, v in claim['source']['files'].items()
                              if p.startswith('legacy/') or p.startswith('experiments/v7.4-rebuild-v1/') and Path(p).name in ('train.py', 'recipe.py', 'data.py')})
        train.fit(folder, parent / 'data', cutoff, member, identity)
        return hashes(work, [folder / name for name in ('event.pt', 'preference.pt', 'positive.npz')])
    if action == 'baseline-feature':
        meta = args.get('meta', JAN20 + 5 * DAY); member = args['member']
        features37(root, work, raw, meta, meta, meta + DAY, member)
        path = feature_path(work, meta, meta, meta + DAY, member)
        return hashes(work, [path, path.with_suffix('.receipt.json')])
    if action == 'baseline-combiner': return baseline_combiner(work, raw, args['member'], args.get('meta', JAN20 + 5 * DAY))
    if action == 'baseline-export': return baseline_export(root, work, raw, args['number'])
    return research_worker(root, work, raw, claim)


def research_worker(root, work, raw, claim):
    action, args = claim['action'], claim['arguments']
    if action == 'feature':
        c, begin, end, member = [args[k] for k in ('cutoff', 'begin', 'end', 'member')]
        if args.get('width', 47) == 37:
            get37(root, work, raw, c, begin, end, member)
            path = feature_path(work, c, begin, end, member)
        else:
            features47(root, work, raw, c, begin, end, member)
            path = path47(work, c, begin, end, member)
        return hashes(work, [path, path.with_suffix('.receipt.json')])
    if action == 'combiner': return fit_candidate(work, raw, args['meta'], args['member'], args['family'])
    if action == 'score': return score(root, work, raw, args['case'], args['candidate'])
    if action == 'summarize': return summarize(work, args['stage'])
    if action == 'export': return candidate_export(root, work, raw, args['number'])
    raise ValueError('Unsupported scientific worker action')


def controller(root, work, action, launch):
    from campaign import completed, baseline
    if action == 'verify':
        print(verify(root, work)); return
    require((work / 'baseline-delivery.json').exists(), 'Deliver the baseline CSV before candidate research')
    if action == 'prepare':
        for cutoff in EXTRA_CUTOFFS:
            for member in MEMBERS:
                launch(root, 'train', f'train-{cutoff}-{member}', 900, dict(cutoff=cutoff, member=member), research=True)
        return
    if action in ('develop', 'screen', 'stress'):
        if completed(work, action):
            require(checked(work, action)['passed'], 'Scientific failure is terminal'); return
        if action != 'develop': require(checked(work, 'develop')['passed'], 'Development failure is terminal')
        if action == 'stress': require(checked(work, 'screen')['passed'], 'Screening failure is terminal')
        for cutoff in EXTRA_CUTOFFS:
            for member in MEMBERS: require(completed(work, f'train-{cutoff}-{member}'), 'Prepare chronological prefixes first')
        candidates = list(CANDIDATES) if action == 'develop' else [checked(work, 'develop')['winner']]
        cases = {'develop': ('jan22_long', 'jan24_forward'), 'screen': ('primary', 'secondary'), 'stress': ('five_day',)}[action]
        for case in cases:
            prepare_case(root, work, case, candidates, launch)
            for candidate in candidates:
                launch(root, 'score', f'score-{candidate}-{case}', arguments=dict(case=case, candidate=candidate), research=True)
        launch(root, 'summarize', action, arguments={'stage': action}, research=True)
        require(checked(work, action)['passed'], 'Gate failed; baseline remains available and V7.5 stops')
    elif action == 'finalize':
        verify_gates(work)
        for number in (1, 2, 3): launch(root, 'export', f'export-{number}', arguments={'number': number})
        publish_candidate(root, work)
        print(verify(root, work))
    elif action == 'replay':
        require((work / 'readiness.json').exists(), 'Only replay a published promoted release')
        number = max(int(p.stem.split('-')[1]) for p in (work / 'tasks').glob('export-*.json')) + 1
        launch(root, 'export', f'export-{number}', arguments={'number': number})
        print(verify(root, work))
    else: raise ValueError('Unknown phase')


def graph_module():
    spec = importlib.util.spec_from_file_location('v75_graph_adapter', ROOT / ADAPTER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def get37(root, work, raw, cutoff, begin, end, member):
    """Import immutable historical numerical caches without relabeling their producer."""
    path = feature_path(work, cutoff, begin, end, member)
    if path.exists(): return matrix(work, cutoff, begin, end, member)
    parent = root / 'artifacts/rebuild'
    original = parent / 'features' / path.name
    if original.exists():
        import run as prior_runner
        require(prior_runner.completed(parent, 'feature-' + original.stem), 'Borrowed feature lacks original accounted evidence')
        receipt = original.with_suffix('.receipt.json'); record = read(receipt); old = record['identity']
        folder = model_paths(root, work, cutoff, member)
        expected_indices = first_indices(raw['u'], raw['t'], begin, end)
        require(record['sha256'] == digest(original) and old['cutoff'] == cutoff and old['begin'] == begin
                and old['end'] == end and old['member'] == member and old['data_sha256'] == digest(parent / 'data/metadata.json')
                and old['checkpoint_sha256'] == digest(folder / 'event.pt')
                and old['positive_sha256'] == digest(folder / 'positive.npz'), 'Borrowed feature provenance differs')
        producer = read(parent / 'scientific-source.json')['source']
        require(old['scientific_source_sha256'] == digest(parent / 'scientific-source.json'), 'Borrowed producer changed')
        for name, value in producer['files'].items():
            if name.startswith('legacy/') or name == 'tools/predict_v6.py':
                require(digest(ROOT / name) == value, 'Borrowed extractor dependency changed')
        with np.load(original, allow_pickle=False) as saved:
            require(np.array_equal(saved['indices'], expected_indices), 'Borrowed event alignment differs')
        path.parent.mkdir(parents=True, exist_ok=True); os.link(original, path)
        write_new(path.with_suffix('.receipt.json'), {'sha256': digest(path), 'identity': {
            'manifest_sha256': digest(work / 'manifest.json'), 'borrowed_from': original.relative_to(root).as_posix(),
            'producer_receipt_sha256': digest(receipt), 'producer_source_sha256': digest(parent / 'scientific-source.json')}})
        return matrix(work, cutoff, begin, end, member)
    return features37(root, work, raw, cutoff, begin, end, member)


def graph(root, work, raw, cutoff, fresh=False):
    module = graph_module(); parent = root / 'artifacts/rebuild'
    identity = {'manifest_sha256': digest(work / 'manifest.json'), 'cutoff': cutoff,
                'parameters': module.CONFIG, 'adapter_sha256': digest(ROOT / ADAPTER)}
    def compute():
        original = parent / 'graphs' / (str(cutoff) + '.npz')
        if original.exists() and not fresh:
            import study as prior_study
            prior_study.checked(parent, 'develop')
            record = read(original.with_suffix('.receipt.json'))
            producer = read(parent / 'scientific-source.json')['source']
            require(record['sha256'] == digest(original) and record['identity']['data_sha256'] == digest(parent / 'data/metadata.json')
                    and producer['files'][ADAPTER] == digest(ROOT / ADAPTER), 'Borrowed graph changed')
            with np.load(original, allow_pickle=False) as saved: return {k: saved[k] for k in saved.files}
        return module.build_graph(context(raw, np.flatnonzero(raw['t'] < cutoff), True), cutoff)
    values = compute() if fresh else cache(work, work / 'graphs' / (str(cutoff) + '.npz'), identity, compute)
    module.validate_graph(values); require(int(values['cutoff']) == cutoff, 'Graph cutoff changed')
    return values


def extra_features(values, target, prefix_events, prefix_users):
    module = graph_module()
    a, b = [module.affinities(values, target, hours) for hours in (24, 72)]
    n = len(target['u']); items = values['items']; h = target['h']; present = h >= 0
    support = np.zeros_like(a)
    if len(items):
        pos = np.searchsorted(items, target['i']); bounded = pos.clip(max=len(items) - 1)
        found = (pos < len(items)) & (items[bounded] == target['i'])
        support = np.where(found, values['counts'][bounded], 0)
    history_coverage = (present & np.isin(h, items)).sum(1) / np.maximum(present.sum(1), 1)
    edge_coverage = np.zeros_like(a); keys = values['keys']
    if len(keys):
        for start in range(0, n, 2048):
            stop = start + 2048; histories = h[start:stop]
            pair = target['i'][start:stop, :, None].astype(np.int64) * 50000 + histories[:, None, :].clip(min=0)
            pos = np.searchsorted(keys, pair); bounded = pos.clip(max=len(keys) - 1)
            found = (pos < len(keys)) & (keys[bounded] == pair) & present[start:stop, None, :]
            edge_coverage[start:stop] = found.sum(2) / np.maximum(present[start:stop].sum(1)[:, None], 1)
    age = (target['t'] - int(values['cutoff'])) / 3600
    require((age >= 0).all(), 'Target precedes graph prefix')
    broadcast = lambda x: np.broadcast_to(np.asarray(x)[:, None], a.shape)
    extras = [a, b, a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True), np.log1p(support),
              broadcast(history_coverage), edge_coverage, broadcast(np.log1p(np.minimum(age, 168))),
              np.full_like(a, np.log1p(prefix_events)), np.full_like(a, np.log1p(prefix_users))]
    output = np.stack(extras, axis=2).reshape(n * 10, 10).astype(np.float32)
    require(np.isfinite(output).all(), 'Nonfinite graph features')
    return output


def path47(work, cutoff, begin, end, member): return work / 'features47' / f'{cutoff}-{begin}-{end}-{member}.npz'


def features47(root, work, raw, cutoff, begin, end, member):
    base = get37(root, work, raw, cutoff, begin, end, member)
    values = graph(root, work, raw, cutoff)
    ix = base['indices']; fitting = raw['t'] < cutoff
    evidence_path = work / 'graph-features' / f'{cutoff}-{begin}-{end}.npz'
    evidence = cache(work, evidence_path, {'manifest_sha256': digest(work / 'manifest.json'),
                     'source_sha256': digest(work / 'scientific-source.json'), 'cutoff': cutoff, 'begin': begin, 'end': end,
                     'graph_sha256': digest(work / 'graphs' / (str(cutoff) + '.npz'))},
                     lambda: {'X': extra_features(values, context(raw, ix, False), int(fitting.sum()), len(np.unique(raw['u'][fitting]))), 'indices': ix})
    require(np.array_equal(evidence['indices'], ix), 'Graph context alignment changed')
    identity = {'manifest_sha256': digest(work / 'manifest.json'), 'base_sha256': digest(feature_path(work, cutoff, begin, end, member)),
                'evidence_sha256': digest(evidence_path), 'source_sha256': digest(work / 'scientific-source.json')}
    return cache(work, path47(work, cutoff, begin, end, member), identity,
                 lambda: {'X': np.column_stack([base['X'], evidence['X']]).astype(np.float32),
                          'indices': ix, 'feature_names': np.concatenate([base['feature_names'], ADDED_NAMES])})


def matrix47(work, cutoff, begin, end, member):
    path = path47(work, cutoff, begin, end, member)
    require(read(path.with_suffix('.receipt.json'))['sha256'] == digest(path), '47-feature cache changed')
    with np.load(path, allow_pickle=False) as saved: result = {k: saved[k] for k in saved.files}
    require(result['X'].shape == (len(result['indices']) * 10, 47) and result['feature_names'][-10:].tolist() == ADDED_NAMES,
            'V7.5 feature schema differs')
    return result


def candidate_model(work, meta, member, family): return work / 'combiners47' / family / str(meta) / (member + '.cbm')


def replica_training(matrices, labels):
    require(matrices and all(np.array_equal(m['indices'], matrices[0]['indices'])
                            and np.array_equal(m['feature_names'], matrices[0]['feature_names']) for m in matrices), 'Replica alignment differs')
    k = len(matrices)
    X = np.concatenate([m['X'] for m in matrices]); y = np.tile(labels.ravel(), k)
    weights = np.full(len(y), 1. / k, dtype=np.float64)
    require(np.allclose(weights.reshape(k, -1).sum(0), 1., rtol=0, atol=1e-15), 'Replica event weight changed')
    return X, y, weights


def fit_candidate(work, raw, meta, member, family):
    require(family in ('current', 'pooled'), 'Undeclared fitting family')
    cutoffs = fitting_prefixes(meta, family == 'pooled')
    matrices = [matrix47(work, c, meta, meta + DAY, member) for c in cutoffs]
    require(all(c <= meta for c in cutoffs), 'Future fitting prefix')
    ix = matrices[0]['indices']; X, y, weights = replica_training(matrices, raw['y'][ix])
    require((raw['t'][ix] >= meta).all() and (raw['t'][ix] < meta + DAY).all(), 'Fitting labels cross metadata day')
    paths = [path47(work, c, meta, meta + DAY, member) for c in cutoffs]
    identity = {'manifest_sha256': digest(work / 'manifest.json'), 'meta': meta, 'member': member,
                'family': family, 'prefixes': cutoffs, 'features': hashes(work, paths), 'parameters': COMBINER,
                'labels_weights': arrays_identity({'y': y, 'weight': weights}),
                'source_sha256': digest(work / 'scientific-source.json')}
    path = candidate_model(work, meta, member, family)
    fit_model(work, path, X, matrices[0]['feature_names'].tolist(), y, identity, weights)
    print('v75-combiner', meta, member, family, len(cutoffs), len(y), flush=True)
    return hashes(work, [path, path.with_suffix('.receipt.json')])


def prepare_case(root, work, case, candidates, launch):
    meta, target, end = CASES[case]
    families = sorted({'pooled' if CANDIDATES[c]['pool'] else 'current' for c in candidates})
    prefixes = sorted(set(c for family in families for c in fitting_prefixes(meta, family == 'pooled')))
    for member in MEMBERS:
        # The original reference uses the metadata-day checkpoint and original 37 features.
        launch(root, 'feature', f'feature37-{meta}-{member}', arguments=dict(cutoff=meta, begin=meta, end=meta + DAY, member=member, width=37), research=True)
        launch(root, 'baseline-combiner', f'reference-combiner-{meta}-{member}', arguments=dict(meta=meta, member=member), research=True)
        for c, begin, stop in [(c, meta, meta + DAY) for c in prefixes] + [(target, target, end)]:
            launch(root, 'feature', f'feature47-{c}-{begin}-{stop}-{member}',
                   arguments=dict(cutoff=c, begin=begin, end=stop, member=member), research=True)
        for family in families:
            launch(root, 'combiner', f'combiner47-{family}-{meta}-{member}', arguments=dict(meta=meta, member=member, family=family), research=True)


def references_and_candidate(root, work, raw, case, candidate):
    meta, target, end = CASES[case]; before, proposed, dependencies = {}, {}, {}
    family = 'pooled' if CANDIDATES[candidate]['pool'] else 'current'
    valid = first_indices(raw['u'], raw['t'], target, end)
    for member in MEMBERS:
        data = matrix47(work, target, target, end, member)
        require(np.array_equal(data['indices'], valid), 'Target indices differ')
        baseline_path = work / 'baseline-combiners' / str(meta) / (member + '.cbm')
        base = matrix(work, target, target, end, member)
        before[member] = predict_model(baseline_path, base['X'], base['feature_names'].tolist(), 37)
        candidate_path = candidate_model(work, meta, member, family)
        proposed[member] = predict_model(candidate_path, data['X'], data['feature_names'].tolist(), 47)
        dependencies.update(hashes(work, [baseline_path, baseline_path.with_suffix('.receipt.json'), candidate_path,
                                         candidate_path.with_suffix('.receipt.json'), path47(work, target, target, end, member)]))
    baseline = compose(before); p = compose(proposed)
    old = root / 'artifacts/rebuild/predictions' / ('history_24h-' + case + '.npz')
    if old.exists():
        with np.load(old, allow_pickle=False) as previous:
            require(np.array_equal(previous['indices'], valid) and np.allclose(previous['baseline'], baseline, rtol=0, atol=1e-15),
                    'Rebuilt comparator changed from V7.4')
    return valid, baseline, calibrate_v7_probabilities(baseline, 1.12), p, dependencies


def score(root, work, raw, case, candidate):
    valid, before, anchor, p, dependencies = references_and_candidate(root, work, raw, case, candidate)
    output = work / 'results' / (candidate + '-' + case + '.json')
    if CANDIDATES[candidate]['gamma'] != 1:
        try: p = calibrate_v7_probabilities(p, CANDIDATES[candidate]['gamma'])
        except ValueError as error:
            if 'Sharpening produced scores outside' not in str(error): raise
            row = {'case': case, 'candidate': candidate, 'status': 'invalid', 'reason': str(error), 'evidence': dependencies}
            write_same(output, row); return hashes(work, [output])
    meta, target, end = CASES[case]
    check_cohort_size(case, len(valid))
    labels = raw['y'][valid]
    row = {'case': case, 'candidate': candidate, 'status': 'valid', 'events': len(valid),
           'baseline_lineage': 'rebuilt_v6_v7_recipe', **engine.point(labels, before, p),
           'v7_comparison': engine.point(labels, anchor, p), 'evidence': dependencies}
    if case in ('primary', 'secondary', 'five_day'):
        for comparison, baseline in ((row, before), (row['v7_comparison'], anchor)):
            comparison.update(asdict(engine.PairedBootstrapEvaluator().evaluate(p, baseline, labels, raw['u'][valid])))
            policy.validate_metric(comparison, confidence=True)
    age = (raw['t'][valid] - target) / 3600; late = (age >= 48) & (age < 120)
    if late.any():
        row['late'] = {'events': int(late.sum()), **engine.point(labels[late], before[late], p[late])}
        row['v7_comparison']['late'] = {'events': int(late.sum()), **engine.point(labels[late], anchor[late], p[late])}
    if case == 'five_day': check_cohort_size('late', int(late.sum()))
    prediction = work / 'predictions' / (candidate + '-' + case + '.npz')
    if prediction.exists():
        with np.load(prediction, allow_pickle=False) as saved:
            require(np.array_equal(saved['candidate'], p) and np.array_equal(saved['baseline'], before), 'Scientific retry changed scores')
    else: save_arrays(prediction, indices=valid, rowids=raw['rowids'][valid], labels=labels, baseline=before, v7=anchor, candidate=p)
    row['evidence'].update(hashes(work, [prediction]))
    row['input_sha256'] = digest(root / 'artifacts/rebuild/data/metadata.json')
    write_same(output, row)
    print('v75-score', case, candidate, 'vs_V6', row['auc_diff'], 'vs_V7', row['v7_comparison']['auc_diff'], 'late', row.get('late', {}).get('auc_diff'), flush=True)
    return hashes(work, [output, prediction])


def check_cohort_size(case, size):
    expected = {'primary': 43964, 'secondary': 35138, 'five_day': 51488, 'late': 7596}
    if case in expected: require(size == expected[case], 'Official cohort differs: ' + case)


def validate_row(work, row):
    from campaign import completed
    require(row['candidate'] in CANDIDATES and row['case'] in CASES
            and completed(work, 'score-' + row['candidate'] + '-' + row['case']), 'Scientific result lacks accounted ownership')
    for name, value in row['evidence'].items(): require(digest(inside(work, name)) == value, 'Scientific dependency changed')
    for name in row['evidence']:
        if name.endswith('.cbm.receipt.json'):
            fitted = read(inside(work, name))['identity'].get('features', {})
            for relative, value in fitted.items():
                if relative.startswith('features47/'):
                    require(digest(inside(work, relative)) == value, 'Fitting feature changed')
    if row['status'] == 'valid':
        require(row['baseline_lineage'] == 'rebuilt_v6_v7_recipe', 'Comparator lineage differs')
        for result in (row, row['v7_comparison']): policy.validate_metric(result, confidence=row['case'] in ('primary', 'secondary', 'five_day'))


def checked(work, stage):
    from campaign import completed
    require(completed(work, stage), 'Stage lacks durable accounting: ' + stage)
    result = read(work / 'stages' / (stage + '.json'))
    require(result['manifest_sha256'] == digest(work / 'manifest.json'), 'Stage generation differs')
    for name, value in result['files'].items(): require(digest(inside(work, name)) == value, 'Stage evidence changed')
    for row in result['results'].values(): validate_row(work, row)
    return result


def select_winner(results):
    valid = [c for c in CANDIDATES if all(results[c + '/' + case['name']]['status'] == 'valid' for case in PROTOCOL['development'])]
    spec = {**PROTOCOL, 'candidate_order': valid}
    return policy.dual_winner(spec, results)


def summarize(work, stage):
    rows = {}; paths = []
    winner = None
    if stage == 'develop':
        keys = [(c, case['name']) for c in CANDIDATES for case in PROTOCOL['development']]
    else:
        previous = checked(work, 'develop'); require(previous['passed'], 'Development failure is terminal')
        winner = previous['winner']
        if stage == 'stress': require(checked(work, 'screen')['passed'], 'Screening failure is terminal')
        keys = [(winner, case) for case in (('primary', 'secondary') if stage == 'screen' else ('five_day',))]
    for candidate, case in keys:
        path = work / 'results' / (candidate + '-' + case + '.json'); row = read(path); validate_row(work, row)
        rows[candidate + '/' + case if stage == 'develop' else case] = row; paths.append(path)
    if stage == 'develop': winner = select_winner(rows); passed = winner is not None
    else: passed = all(r['status'] == 'valid' for r in rows.values()) and policy.stage_passed(PROTOCOL, stage, rows)
    output = work / 'stages' / (stage + '.json')
    write_same(output, {'schema_version': 6, 'manifest_sha256': digest(work / 'manifest.json'), 'winner': winner,
                       'passed': bool(passed), 'results': rows, 'files': hashes(work, paths),
                       'validation_status': 'retrospective reused dates; not independent holdout evidence'})
    print('v75-stage', stage, 'passed', passed, 'winner', winner, flush=True)
    return hashes(work, [output])


def verify_gates(work):
    develop, screen, stress = [checked(work, name) for name in ('develop', 'screen', 'stress')]
    winner = select_winner(develop['results'])
    require(winner is not None and all(r['passed'] and r['winner'] == winner for r in (develop, screen, stress)), 'All stages must pass for the frozen winner')
    gates = policy.dual_gates(screen, stress)
    require(all(value for group in gates.values() for value in group.values()), 'All eight promotion gates must pass')
    return winner, gates


def candidate_export(root, work, raw, number):
    winner, gates = verify_gates(work)
    evaluation = dataset.load(root / 'artifacts/rebuild/data', 'evaluation')
    values = graph(root, work, raw, FULL, fresh=True)
    evidence = extra_features(values, evaluation, len(raw['u']), len(np.unique(raw['u'])))
    members = {}
    family = 'pooled' if CANDIDATES[winner]['pool'] else 'current'
    dependencies = {}
    for member in MEMBERS:
        base = features37(root, work, raw, FULL, FULL, FULL + 7 * DAY, member, evaluation=True, fresh=True)
        X = np.column_stack([base['X'], evidence]).astype(np.float32)
        names = base['feature_names'].tolist() + ADDED_NAMES
        path = candidate_model(work, JAN20 + 5 * DAY, member, family)
        members[member] = predict_model(path, X, names, 47)
        dependencies.update(hashes(work, [path, path.with_suffix('.receipt.json')]))
    p = compose(members)
    if CANDIDATES[winner]['gamma'] != 1: p = calibrate_v7_probabilities(p, CANDIDATES[winner]['gamma'])
    from submission_export import export_submission, validate_csv
    from row_auc_final_utils import verify_csv_probabilities
    folder = work / 'exports' / ('attempt-' + str(number) + '-' + str(time.time_ns())); folder.mkdir(parents=True)
    final = work / 'exports' / str(number); output = folder / 'v7_5.csv'
    export_submission(evaluation['rowids'].ravel(), p, root / 'data/raw/example_submission.csv', output, 529390)
    validate_csv(output, root / 'data/raw/example_submission.csv', 529390)
    verify_csv_probabilities(output, evaluation['rowids'].ravel(), p)
    vectors = folder / 'predictions.npz'; save_arrays(vectors, candidate=p, rowids=evaluation['rowids'])
    graph_path = folder / 'graph.npz'; save_arrays(graph_path, **values)
    if number > 1: require(digest(output) == digest(work / 'exports/1/v7_5.csv'), 'Fresh independent candidate CSV differs')
    receipt = folder / 'validation.json'; outputs = [output, vectors, graph_path]
    write_new(receipt, {'schema_version': 6, 'winner': winner, 'gates': gates, 'manifest_sha256': digest(work / 'manifest.json'),
                        'fresh_neural_features': True, 'fresh_graph': True, 'number': number, 'dependencies': dependencies,
                        'files': {(final / p.name).relative_to(work).as_posix(): digest(p) for p in outputs}})
    if final.exists(): require(all(digest(final / p.name) == digest(p) for p in outputs + [receipt]), 'Technical final replay differs')
    else: folder.rename(final)
    return hashes(work, [final / p.name for p in outputs + [receipt]])


def publish_candidate(root, work):
    from campaign import completed
    winner, gates = verify_gates(work)
    for number in (1, 2, 3): require(completed(work, f'export-{number}'), 'Independent accounted candidate exports and replay required')
    accounting_path = seal_accounting(root, work)
    source = work / 'exports/1/v7_5.csv'; require(digest(source) == digest(work / 'exports/2/v7_5.csv'), 'Candidate export hash differs')
    output = root / 'artifacts/submissions/v7_5.csv'
    if output.exists(): require(digest(output) == digest(source), 'Preserve different competition submission')
    else: os.link(source, output)
    record = {'schema_version': 6, 'model_role': 'promoted_v7_5', 'baseline_lineage': 'rebuilt_v6_v7_recipe',
              'csv_valid': True, 'replay_verified': True, 'promotion_supported': True, 'submission_ready': True,
              'historical_v6_v7_reproduced': False, 'winner': winner, 'gates': gates,
              'manifest_sha256': digest(work / 'manifest.json'), 'submission': output.relative_to(root).as_posix(),
              'sha256': digest(output), 'rows': 529390,
              'accounting_record': accounting_path.relative_to(work).as_posix(),
              'accounting_sha256_at_publication': digest(accounting_path)}
    write_same(root / 'configs/v7_5.json', record)
    write_same(output.with_suffix('.csv.receipt.json'), record)
    checksum = output.with_suffix('.csv.sha256'); text = record['sha256'] + '  ' + output.name + '\n'
    if checksum.exists(): require(checksum.read_text() == text, 'Checksum changed')
    else:
        with checksum.open('x') as stream: stream.write(text)
    write_same(work / 'readiness.json', record)


def verify(root, work):
    from campaign import completed
    baseline = read(work / 'baseline-delivery.json')
    for member in MEMBERS:
        require(completed(work, f'train-{FULL}-{member}'), 'Baseline model evidence changed')
    for number in (1, 2): require(completed(work, f'baseline-export-{number}'), 'Baseline inference evidence changed')
    baseline_path = inside(root, baseline['submission'])
    require(digest(inside(root, baseline['submission'])) == baseline['sha256'] and baseline['model_role'] == 'rebuilt_baseline'
            and baseline['promotion_supported'] is False, 'Baseline delivery identity changed')
    require(digest(inside(work, baseline['accounting_record'])) == baseline['accounting_sha256_at_publication'], 'Baseline accounting seal changed')
    require(read(baseline_path.with_suffix('.csv.receipt.json')) == baseline
            and baseline_path.with_suffix('.csv.sha256').read_text() == baseline['sha256'] + '  ' + baseline_path.name + '\n',
            'Baseline sidecars changed')
    if not (work / 'readiness.json').exists(): return {'baseline_verified': True, 'v7_5_promoted': False, 'baseline_submission': baseline['submission']}
    winner, gates = verify_gates(work); record = read(work / 'readiness.json')
    require(record == read(root / 'configs/v7_5.json') and record['winner'] == winner and record['gates'] == gates
            and record['promotion_supported'] is True and digest(inside(root, record['submission'])) == record['sha256'], 'Promoted receipt differs')
    require(digest(inside(work, record['accounting_record'])) == record['accounting_sha256_at_publication'], 'Publication accounting seal changed')
    candidate_path = inside(root, record['submission'])
    require(read(candidate_path.with_suffix('.csv.receipt.json')) == record
            and candidate_path.with_suffix('.csv.sha256').read_text() == record['sha256'] + '  ' + candidate_path.name + '\n',
            'Candidate sidecars changed')
    for number in (1, 2, 3): require(completed(work, f'export-{number}'), 'Final evidence changed')
    return {'baseline_verified': True, 'v7_5_promoted': True, 'submission': record['submission'], 'sha256': record['sha256']}


def seal_accounting(root, work):
    ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
    require(ledger['total_seconds'] == 43200 and ledger['carry_identity'] == digest(root / 'artifacts/rebuild/genesis.json')
            and 0 <= ledger['used_seconds'] <= 43200 and not any(a['status'] == 'running' for a in ledger['attempts'])
            and abs(sum(a['seconds'] for a in ledger['attempts']) - ledger['used_seconds']) < 1e-6,
            'Accounting must settle and reconcile before publication')
    path = work / 'publications' / (canonical(ledger) + '.json')
    write_same(path, ledger)
    return path
