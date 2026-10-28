"""Verified foundation reuse and new caches; historical workspaces stay read-only."""
from pathlib import Path
import inspect
import numpy as np
from ml_workflow.contracts import canonical, compose, context, digest, first_indices, inside, require
from ml_workflow.storage import read, write_new, save_arrays
from ml_workflow.cache import arrays_identity
from predict_v6 import portable_pipeline
from predict_v7 import calibrate_v7_probabilities
from v76_protocol import SPEC, MEMBERS, COMBINER, FULL, DAY, FEATURES
from recipe import RECIPE
import pipeline as prior


def hashes(work, paths): return {p.relative_to(work).as_posix(): digest(p) for p in paths}


def feature_path(work, d, begin, end, member):
    return work / 'features' / f'{d}-{begin}-{end}-{member}.npz'


def matrix(path):
    require(read(path.with_suffix('.receipt.json'))['sha256'] == digest(path), 'Cache bytes changed')
    with np.load(path, allow_pickle=False) as values: return {k: values[k] for k in values.files}


def foundation(root, work, cutoff, member):
    m = read(work / 'manifest.json'); relative = m['foundation_folders'][str(cutoff) + '/' + member]
    folder = inside(root, relative)
    for name in ('event.pt', 'preference.pt', 'positive.npz'):
        p = folder / name
        require(digest(p) == m['foundation_files'][p.relative_to(root).as_posix()], 'Foundation asset changed')
    return folder


def features(root, work, raw, cutoff, begin, end, member, evaluation=None, fresh=False):
    import torch
    from build_submission import neural_pipeline
    folder = foundation(root, work, cutoff, member)
    target_raw = raw if evaluation is None else evaluation
    ix = first_indices(raw['u'], raw['t'], begin, end) if evaluation is None else np.arange(len(evaluation['u']))
    require(len(ix) and (target_raw['t'][ix] >= cutoff).all(), 'Invalid causal feature interval')
    identity = {'manifest_sha256': digest(work / 'manifest.json'), 'cutoff': cutoff, 'begin': begin, 'end': end,
                'member': member, 'evaluation': evaluation is not None, 'checkpoint_sha256': digest(folder / 'event.pt'),
                'indices': arrays_identity({'indices': ix}), 'extractor': canonical(inspect.getsource(features)),
                'dtype': 'float32', 'device': 'cuda', 'tf32': False}
    def compute():
        state = torch.load(folder / 'event.pt', map_location='cpu', weights_only=True)
        require(state['cutoff'] == cutoff and state['epoch'] == 8 and state['stage'] == 'event'
                and state['purpose'] == 'research' and state['dimension'] == MEMBERS[member][0]
                and state['seed'] == MEMBERS[member][1] and state['recipe_sha256'] == canonical(RECIPE)
                and state['positive_sha256'] == digest(folder / 'positive.npz')
                and state['preference_source_sha256'] == digest(folder / 'preference.pt'), 'Checkpoint provenance differs')
        # Exact interval caches only: slicing a longer first-user cohort would lose events.
        old = root / 'artifacts/v7_5/features37' / feature_path(work, cutoff, begin, end, member).name
        m = read(work / 'manifest.json'); relative = old.relative_to(root).as_posix()
        if not fresh and evaluation is None and relative in m['protected_inputs']:
            require(digest(old) == m['protected_inputs'][relative], 'Inherited feature changed')
            values = matrix(old)
            require(np.array_equal(values['indices'], ix), 'Inherited feature cohort differs')
            return values
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        require(torch.cuda.is_available(), 'CUDA feature extraction is required')
        fitting = context(raw, np.flatnonzero(raw['t'] < cutoff), True); fitting.pop('rowids')
        target = context(target_raw, ix, False); target.pop('rowids')
        X, names, *_ = portable_pipeline(neural_pipeline)(folder / 'event.pt', fitting, target, folder / 'positive.npz', torch.device('cuda'))
        require(X.shape == (len(ix) * 10, 37) and X.dtype == np.float32 and np.isfinite(X).all(), 'Feature schema differs')
        return {'X': X, 'feature_names': np.asarray(names), 'indices': ix}
    if fresh: return compute()
    return prior.cache(work, feature_path(work, cutoff, begin, end, member), identity, compute)


def combiner_path(work, c, end, member): return work / 'combiners' / f'{c}-{end}' / (member + '.cbm')


def fit_combiner(root, work, raw, c, end, member):
    require(c < end <= c + DAY, 'Combiner fitting labels exceed permitted interval')
    path = combiner_path(work, c, end, member)
    data = matrix(feature_path(work, c, c, end, member))
    expected_ix = first_indices(raw['u'], raw['t'], c, end)
    require(np.array_equal(data['indices'], expected_ix), 'Combiner fitting cohort differs')
    identity = {'manifest_sha256': digest(work / 'manifest.json'), 'c': c, 'end': end, 'member': member,
                'features_sha256': digest(feature_path(work, c, c, end, member)),
                'labels_sha256': arrays_identity({'y': raw['y'][expected_ix]}), 'parameters': COMBINER}
    # Reuse exact old models only; new receipts record the true original producer.
    old = root / 'artifacts/v7_5/baseline-combiners' / str(c) / (member + '.cbm')
    m = read(work / 'manifest.json'); relative = old.relative_to(root).as_posix()
    if end == c + DAY and relative in m['protected_inputs'] and not path.exists():
        require(digest(old) == m['protected_inputs'][relative], 'Inherited combiner changed')
        old_identity = read(old.with_suffix('.receipt.json'))['identity']
        require(old_identity['parameters'] == COMBINER and old_identity['meta'] == c
                and old_identity['member'] == member
                and old_identity['labels'] == arrays_identity({'y': raw['y'][expected_ix]}), 'Inherited combiner contract differs')
        import os
        path.parent.mkdir(parents=True, exist_ok=True); os.link(old, path)
        write_new(path.with_suffix('.receipt.json'), {'identity': identity, 'sha256': digest(path),
                  'borrowed_from': relative, 'producer_receipt_sha256': digest(old.with_suffix('.receipt.json'))})
    else:
        prior.fit_model(work, path, data['X'], data['feature_names'].tolist(), raw['y'][expected_ix].ravel(), identity)
    return hashes(work, [path, path.with_suffix('.receipt.json')])


def graph(root, work, raw, cutoff, fresh=False):
    module = prior.graph_module()
    identity = {'manifest_sha256': digest(work / 'manifest.json'), 'cutoff': cutoff, 'parameters': module.CONFIG}
    def compute(): return module.build_graph(context(raw, np.flatnonzero(raw['t'] < cutoff), True), cutoff)
    values = compute() if fresh else prior.cache(work, work / 'graphs' / (str(cutoff) + '.npz'), identity, compute)
    module.validate_graph(values); require(int(values['cutoff']) == cutoff, 'Graph cutoff changed')
    return values


def make_features(raw, ix, cutoff, anchor, graph_columns):
    require(anchor.shape == (len(ix), 10) and graph_columns.shape == (len(ix), 10, 7)
            and np.isfinite(anchor).all() and ((anchor >= 0) & (anchor <= 1)).all(), 'Invalid residual feature inputs')
    mass = anchor.sum(1, keepdims=True)
    q = np.divide(anchor, mass, out=np.full_like(anchor, .1), where=mass > 0)
    logq = np.log(np.maximum(q, 1e-300)); entropy = -(q * logq).sum(1) / np.log(10)
    safe = np.clip(anchor, 1e-6, 1 - 1e-6)
    history = (raw['h'][ix] >= 0).sum(1)
    unseen = ~np.isin(raw['u'][ix], raw['u'][raw['t'] < cutoff])
    # Evaluation users are not in the evaluation prefix. Caller supplies the actual fitting users below.
    X = np.concatenate([graph_columns.astype(np.float64), (np.log(safe) - np.log1p(-safe))[:, :, None],
                        np.broadcast_to(np.log1p(mass), anchor.shape)[:, :, None], q[:, :, None],
                        np.broadcast_to(entropy[:, None], anchor.shape)[:, :, None],
                        np.broadcast_to((history / 20)[:, None], anchor.shape)[:, :, None],
                        np.broadcast_to(unseen[:, None], anchor.shape)[:, :, None]], axis=2)
    return {'X': X, 'anchor': anchor, 'active': (history > 0) & (graph_columns[:, :, 6].max(1) > 0), 'indices': ix}


def views(root, work, raw, c, d, fit_end, begin, end, evaluation=None, fresh=False):
    require(c < fit_end <= min(c + DAY, d) and d <= begin < end, 'Noncausal forecast view')
    target_raw = raw if evaluation is None else evaluation
    ix = first_indices(raw['u'], raw['t'], begin, end) if evaluation is None else np.arange(len(evaluation['u']))
    predictions = {}; dependencies = {}
    for member in MEMBERS:
        data = (features(root, work, raw, d, begin, end, member, evaluation, True) if fresh
                else matrix(feature_path(work, d, begin, end, member)))
        require(np.array_equal(data['indices'], ix), 'View feature order differs')
        model_path = combiner_path(work, c, fit_end, member)
        predictions[member] = prior.predict_model(model_path, data['X'], data['feature_names'].tolist(), 37)
        dependencies.update(hashes(work, [model_path, model_path.with_suffix('.receipt.json')]))
        if not fresh: dependencies.update(hashes(work, [feature_path(work, d, begin, end, member)]))
    before = compose(predictions); anchor = calibrate_v7_probabilities(before, 1.12)
    values = graph(root, work, raw, d, fresh=fresh)
    columns = prior.extra_features(values, context(target_raw, ix, False), int((raw['t'] < d).sum()),
                                   len(np.unique(raw['u'][raw['t'] < d]))).reshape(-1, 10, 10)[:, :, :7]
    result = make_features(target_raw, ix, d, anchor, columns)
    # Warm/cold is relative to the actual neural fitting corpus, including for evaluation.
    result['X'][:, :, 12] = (~np.isin(target_raw['u'][ix], raw['u'][raw['t'] < d]))[:, None]
    result.update(baseline=before, members=np.stack(list(predictions.values())), rowids=target_raw['rowids'][ix])
    return result, dependencies, values


def view_path(work, c, d, fit_end, begin, end): return work / 'views' / f'{c}-{d}-{fit_end}-{begin}-{end}.npz'


def save_view(root, work, raw, c, d, fit_end, begin, end):
    path = view_path(work, c, d, fit_end, begin, end)
    data, dependencies, _ = views(root, work, raw, c, d, fit_end, begin, end)
    save_arrays(path, **{k: data[k] for k in ('X', 'anchor', 'active', 'indices', 'baseline')})
    write_new(path.with_suffix('.receipt.json'), {'sha256': digest(path), 'identity': {
        'manifest_sha256': digest(work / 'manifest.json'), 'c': c, 'd': d, 'fit_end': fit_end,
        'begin': begin, 'end': end, 'dependencies': dependencies, 'feature_names': FEATURES}})
    return hashes(work, [path, path.with_suffix('.receipt.json')])
