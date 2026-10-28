"""Causal asset bridge and independent competition inference for V7.5."""
from copy import deepcopy
from pathlib import Path
import os
import time
import inspect

import numpy as np

import data as dataset
from recipe import RECIPE
from protocol import MEMBERS, COMBINER, DAY, JAN20, FULL, PROTOCOL
from ml_workflow.contracts import canonical, compose, context, digest, first_indices, inside, require
from ml_workflow.storage import read, write_new, save_arrays
from ml_workflow.cache import arrays_identity
from predict_v7 import calibrate_v7_probabilities


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
    ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
    require(not any(a['status'] == 'running' for a in ledger['attempts']), 'Accounting must settle before publication')
    source = work / 'baseline-exports/1/rebuilt_v6.csv'
    destination = root / 'artifacts/submissions/rebuilt_v6_submission.csv'; destination.parent.mkdir(parents=True, exist_ok=True)
    require(digest(source) == digest(work / 'baseline-exports/2/rebuilt_v6.csv'), 'Baseline export hashes differ')
    if destination.exists(): require(digest(destination) == digest(source), 'Preserve different submission bytes')
    else: os.link(source, destination)
    receipt = {'schema_version': 6, 'model_role': 'rebuilt_baseline', 'baseline_lineage': 'rebuilt_v6_v7_recipe',
               'csv_valid': True, 'replay_verified': True, 'promotion_supported': False, 'historical_v6_reproduced': False,
               'manifest_sha256': digest(work / 'manifest.json'), 'rows': 529390,
               'submission': destination.relative_to(root).as_posix(), 'sha256': digest(destination),
               'accounting_sha256_at_publication': digest(root / 'artifacts/rebuild/compute-ledger.json')}
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
    raise ValueError('Scientific implementation is not frozen yet')


def controller(root, work, action, launch):
    raise ValueError('Scientific implementation is not frozen yet')
