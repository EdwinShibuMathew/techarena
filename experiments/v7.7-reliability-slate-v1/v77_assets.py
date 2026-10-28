"""Portable 37-feature foundation, user-disjoint teachers and slate evidence.

Only explicit, hash-pinned predecessor assets are borrowed. No predecessor
campaign, selection, training or publication controller is imported.
"""
import numpy as np
from ml_workflow.contracts import canonical, compose, context, digest, first_indices, inside, require, DAY
from ml_workflow.cache import arrays_identity
from ml_workflow.storage import read, write_new, save_arrays
from predict_v6 import portable_pipeline
from predict_v7 import calibrate_v7_probabilities
from recipe import RECIPE, MEMBERS, COMBINER
from v77_protocol import folds
import v77_features as slate


def foundation(root, manifest, cutoff, member):
    folder = inside(root, manifest['foundation_folders'][str(cutoff) + '/' + member])
    for name in ('event.pt', 'preference.pt', 'positive.npz'):
        path = folder / name
        require(digest(path) == manifest['foundation_files'][path.relative_to(root).as_posix()], 'Foundation changed')
    return folder


def matrix(root, manifest, raw, cutoff, begin, end, member, evaluation=None, fresh=False):
    import torch
    from build_submission import neural_pipeline
    folder = foundation(root, manifest, cutoff, member)
    target = raw if evaluation is None else evaluation
    ix = first_indices(raw['u'], raw['t'], begin, end) if evaluation is None else np.arange(len(target['u']))
    require(len(ix) and (target['t'][ix] >= cutoff).all(), 'Noncausal feature cohort')
    state = torch.load(folder / 'event.pt', map_location='cpu', weights_only=True)
    require(state['cutoff'] == cutoff and state['epoch'] == 8 and state['stage'] == 'event'
            and state['purpose'] == 'research' and state['dimension'] == MEMBERS[member][0]
            and state['seed'] == MEMBERS[member][1] and state['recipe_sha256'] == canonical(RECIPE)
            and state['positive_sha256'] == digest(folder / 'positive.npz')
            and state['preference_source_sha256'] == digest(folder / 'preference.pt'), 'Foundation provenance differs')
    old = root / 'artifacts/v7_5/features37' / f'{cutoff}-{begin}-{end}-{member}.npz'
    relative = old.relative_to(root).as_posix()
    if not fresh and evaluation is None and relative in manifest['protected_inputs']:
        require(digest(old) == manifest['protected_inputs'][relative]
                and read(old.with_suffix('.receipt.json'))['sha256'] == digest(old), 'Borrowed cache changed')
        with np.load(old, allow_pickle=False) as saved:
            result = {k: saved[k] for k in ('X', 'feature_names', 'indices')}
        require(np.array_equal(result['indices'], ix), 'Borrowed first-user cohort differs')
    else:
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        require(torch.cuda.is_available(), 'Production feature extraction requires CUDA')
        fit = context(raw, np.flatnonzero(raw['t'] < cutoff), True); fit.pop('rowids')
        query = context(target, ix, False); query.pop('rowids')
        X, names, *_ = portable_pipeline(neural_pipeline)(folder / 'event.pt', fit, query, folder / 'positive.npz', torch.device('cuda'))
        result = {'X': X, 'feature_names': np.asarray(names), 'indices': ix}
    require(result['X'].shape == (len(ix) * 10, 37) and result['X'].dtype == np.float32
            and len(result['feature_names']) == 37 and np.isfinite(result['X']).all(), '37-feature schema differs')
    return result


def fit_teacher(X, names, labels, parameters=None):
    from catboost import CatBoostClassifier, Pool
    parameters = COMBINER if parameters is None else parameters
    require(len(np.unique(labels)) == 2, 'Teacher requires both classes')
    model = CatBoostClassifier(**parameters)
    model.fit(Pool(X, label=labels, feature_names=list(names)), verbose=False)
    return model


def predict_teacher(path, data):
    from catboost import CatBoostClassifier, Pool
    model = CatBoostClassifier(); model.load_model(str(path))
    require(model.feature_names_ == data['feature_names'].tolist(), 'Teacher feature order differs')
    p = model.predict_proba(Pool(data['X'], feature_names=model.feature_names_))[:, 1].reshape(-1, 10)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid teacher prediction')
    return p


def oof(data, raw, directory, parameters=None):
    """Hold out complete users. One fold assignment is reused across all views."""
    ix = data['indices']; group = folds(raw['u'][ix]); y = raw['y'][ix]
    require(set(group.tolist()) == set(range(5)), 'All five user folds must be represented')
    output = np.full(y.shape, np.nan, dtype=np.float64); receipts = []
    X = data['X'].reshape(len(ix), 10, 37)
    for f in range(5):
        train, held = group != f, group == f
        require(not np.intersect1d(raw['u'][ix[train]], raw['u'][ix[held]]).size, 'Teacher user leakage')
        teacher = fit_teacher(X[train].reshape(-1, 37), data['feature_names'], y[train].ravel(), parameters)
        path = directory / f'fold-{f}.cbm'; path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.exists(), 'Never overwrite a teacher attempt'); teacher.save_model(str(path))
        output[held] = predict_teacher(path, {'X': X[held].reshape(-1, 37), 'feature_names': data['feature_names']})
        receipts.append({'fold': f, 'teacher_sha256': digest(path), 'training_users': arrays_identity({'users': np.unique(raw['u'][ix[train]])}),
                         'held_users': arrays_identity({'users': np.unique(raw['u'][ix[held]])}), 'training_events': int(train.sum()), 'held_events': int(held.sum())})
    require(np.isfinite(output).all(), 'OOF coverage incomplete')
    save_arrays(directory / 'oof.npz', probabilities=output, indices=ix, folds=group)
    write_new(directory / 'oof.json', {'folds': receipts, 'parameters': COMBINER if parameters is None else parameters,
                                      'labels': arrays_identity({'y': y})})
    return output


def cosine(root, manifest, raw, cutoff, ix):
    import torch
    vectors = []
    for member in MEMBERS:
        state = torch.load(foundation(root, manifest, cutoff, member) / 'event.pt', map_location='cpu', weights_only=True)
        # Portable event checkpoint stores the original preference item space.
        vectors.append(state['model_state']['preference.item.weight'].numpy()[raw['i'][ix]])
    return slate.geometry(vectors)


def make_view(root, manifest, fitting, target, ix, cutoff, members):
    before = compose(dict(zip(MEMBERS, members)))
    anchor = calibrate_v7_probabilities(before, 1.12)
    event, row = slate.master(members, anchor, slate.user_context(fitting, target, ix, cutoff),
                             cosine(root, manifest, target, cutoff, ix))
    return {'indices': ix, 'event': event, 'row': row, 'anchor': anchor, 'baseline': before,
            'members': members, 'rowids': target['rowids'][ix]}


def pool(datasets, raw):
    result = {k: np.concatenate([d[k] for d in datasets]) for k in ('indices', 'event', 'row', 'anchor', 'strata')}
    result['weights'] = slate.balanced_weights(result['indices'], result['strata'])
    result['labels'] = raw['y'][result['indices']]
    result['unique_users'] = len(np.unique(raw['u'][result['indices']]))
    result['coverage'] = [{'stratum': s.tolist(), 'views': int((code == k).sum()),
                           'weight': float(result['weights'][code == k].sum())}
                          for k, s in enumerate(np.unique(result['strata'], axis=0))
                          for code in [np.unique(result['strata'], axis=0, return_inverse=True)[1]]]
    return result
