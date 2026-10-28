"""Schema-v4 chronological graph correction; shared runtime owns all gates."""
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import sys
import time

import numpy as np

from .contracts import (DAY, JAN20, canonical, compose, context, digest,
                        first_indices, inside, require)
from .cache import arrays_identity
from .storage import read, save_arrays, write_new
from .policy import V7_SHA


def preflight(root, spec):
    """Read-only; never initialize a missing authoritative research ledger."""
    root = Path(root).resolve()
    absent = [v['path'] for v in spec['inputs'].values() if not inside(root, v['path']).is_file()]
    ledger_path = inside(root, 'artifacts/experiments/compute-ledger.json')
    if not ledger_path.is_file():
        absent.append('artifacts/experiments/compute-ledger.json')
    require(not absent, 'Restore verified assets and authoritative ledger before research; missing: ' + ', '.join(absent))
    ledger = read(ledger_path)
    carry = spec['inputs']['carry']
    prior_asset = spec['inputs']['continuation']
    require(digest(inside(root, carry['path'])) == carry['sha256']
            and digest(inside(root, prior_asset['path'])) == prior_asset['sha256'], 'Continuation evidence changed')
    prior = read(inside(root, prior_asset['path']))
    require(ledger['total_seconds'] == spec['budget']['total_seconds'] == 14400
            and ledger['carry_identity'] == carry['sha256'], 'Preserve original budget lineage')
    require(all(a['status'] == 'complete' for a in ledger['attempts']), 'Reconcile unfinished or failed research attempts')
    require(np.isfinite(ledger['used_seconds'])
            and prior['total_compute_seconds'] <= ledger['used_seconds'] < 14400, 'Reconcile authoritative cumulative compute')
    require(ledger['attempts'] and abs(ledger['used_seconds'] -
            (read(inside(root, carry['path']))['total_compute_seconds'] +
             sum(a['seconds'] for a in ledger['attempts']))) < 1e-6, 'Research accounting does not reconcile')
    return {'inputs': len(spec['inputs']), 'used_seconds': ledger['used_seconds'],
            'remaining_seconds': 14400 - ledger['used_seconds'], 'carry_identity': ledger['carry_identity']}


def graph(root, work, m, module, raw, cutoff, name=None, use_cache=True):
    from .postprocessing import cached_scores
    prefix = context(raw, np.flatnonzero(raw['t'] < cutoff), True)
    identity = {
        'kind': 'history_graph_v1', 'cutoff': cutoff, 'prefix': arrays_identity(prefix),
        'source': {p: h for p, h in m['source']['files'].items()
                   if p.startswith('tools/ml_workflow/') or p == m['spec']['adapter']},
        'parameters': deepcopy(module.CONFIG),
        'runtime': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
        'python': sys.version.split()[0], 'precision': 'float64',
    }
    path = work / 'graphs' / ((str(cutoff) if name is None else name) + '.npz')
    receipt = path.with_suffix('.json')
    if path.exists() or receipt.exists():
        require(use_cache and path.is_file() and receipt.is_file(), 'Preserve graph evidence; no independent replay reuse')
        record = read(receipt)
        require(record['identity'] == identity and record['sha256'] == digest(path), 'Graph identity or bytes changed')
    else:
        compute = lambda: module.build_graph(deepcopy(prefix), int(cutoff))
        if use_cache:
            arrays, cache_receipt = cached_scores(root, work, m, identity, lambda folder: compute())
        else:
            arrays, cache_receipt = compute(), None
        require(int(arrays['cutoff']) == cutoff, 'Adapter changed graph cutoff')
        module.validate_graph(arrays)
        save_arrays(path, **arrays)
        write_new(receipt, {'identity': identity, 'sha256': digest(path),
                           'arrays': arrays_identity(arrays), 'cache_used': bool(use_cache),
                           'cache_receipt': cache_receipt.relative_to(work).as_posix() if cache_receipt else None})
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    require(arrays_identity(arrays) == read(receipt)['arrays'] and int(arrays['cutoff']) == cutoff,
            'Graph schema or cutoff changed')
    module.validate_graph(arrays)
    files = {p.relative_to(work).as_posix(): digest(p) for p in (path, receipt)}
    cache = read(receipt)['cache_receipt']
    if cache:
        files[cache] = digest(inside(work, cache))
    return arrays, files


def _reference(root, work, m, module, raw, fit, valid, case):
    from . import engine, postprocessing
    if m['spec']['purpose'] == 'synthetic_validation':
        _, scores = engine.baseline(root, m, case.baseline, raw['rowids'][valid])
        _, fitting = engine.baseline(root, m, 'baseline_fit_' + case.name, raw['rowids'][fit])
        receipt = work / 'synthetic-baselines' / (case.name + '.json')
        if not receipt.exists():
            write_new(receipt, {'synthetic': True, 'fitting': arrays_identity({'scores': fitting}),
                                'target': arrays_identity({'scores': scores})})
        return fitting, scores, {receipt.relative_to(work).as_posix(): digest(receipt)}
    inventory = postprocessing.inventory(root, m)
    parameter_case = case.name if case.name in ('primary', 'secondary', 'five_day') else 'five_day' if case.meta_cutoff == JAN20 + DAY else 'development'
    fitting, receipt = postprocessing.crossfit(root, work, m, module, inventory, raw, fit, case.meta_cutoff, parameter_case)
    scores, files = postprocessing.reference_scores(root, work, m, inventory, raw, fit, valid, case)
    files[receipt.relative_to(work).as_posix()] = digest(receipt)
    return fitting, scores, files


def score(root, work, m, module, raw, candidate, case):
    from . import engine
    from predict_v7 import calibrate_v7_probabilities
    start = time.monotonic()
    fit = first_indices(raw['u'], raw['t'], case.meta_cutoff, case.meta_cutoff + DAY)
    valid = first_indices(raw['u'], raw['t'], case.target_cutoff, case.target_end)
    require(len(fit) and len(valid) and raw['t'][fit].max() < case.target_cutoff, 'Invalid history fitting timeline')
    if m['spec']['purpose'] == 'research' and case.name in ('primary', 'secondary', 'five_day'):
        require(len(valid) == {'primary': 43964, 'secondary': 35138, 'five_day': 51488}[case.name], 'Official first-user cohort differs')
    fitting, before, files = _reference(root, work, m, module, raw, fit, valid, case)
    params = deepcopy(m['spec']['candidates'][candidate])
    fit_graph, graph_files = graph(root, work, m, module, raw, case.meta_cutoff)
    files.update(graph_files)
    fit_features = module.affinities(fit_graph, context(raw, fit, False), params['half_life_hours'])
    anchor_fit = calibrate_v7_probabilities(fitting, 1.12)
    metadata = {'spec_sha256': m['spec_sha256'], 'candidate': candidate,
                'parameters_sha256': canonical(params), 'meta_cutoff': case.meta_cutoff,
                'maximum_label_timestamp': int(raw['t'][fit].max()), 'fitting_events': len(fit),
                'fitting_identity': canonical(arrays_identity({'anchor': anchor_fit, 'features': fit_features,
                                                              'labels': raw['y'][fit], 'users': raw['u'][fit]})),
                'graph_identity': canonical(graph_files)}
    path = work / 'calibrators' / candidate / (canonical(metadata) + '.json')
    if path.exists():
        saved = module.load_calibrator(path, deepcopy(metadata), deepcopy(params))
    else:
        saved = module.fit_calibrator(anchor_fit.copy(), fit_features.copy(), raw['y'][fit].copy(),
                                      deepcopy(metadata), path, deepcopy(params))
    require(saved['metadata'] == metadata and saved['parameters'] == params, 'Adapter modified expected calibrator state')
    centered = fit_features - fit_features.mean(1, keepdims=True)
    norm = {'rms': max(float(np.sqrt(np.mean(centered ** 2))), params['config']['scale_floor'])}
    require(saved['normalization'] == norm, 'Graph normalization differs from fitting features')
    target_graph, graph_files = graph(root, work, m, module, raw, case.target_cutoff)
    files.update(graph_files)
    features = module.affinities(target_graph, context(raw, valid, False), params['half_life_hours'])
    anchor = calibrate_v7_probabilities(before, 1.12)
    after = module.predict_calibrator(anchor.copy(), features.copy(), deepcopy(saved))
    require(after.shape == before.shape and np.isfinite(after).all() and ((after >= 0) & (after <= 1)).all()
            and np.allclose(after.sum(1), anchor.sum(1), rtol=0, atol=1e-14), 'Invalid history predictions or event mass')
    prediction = work / 'predictions' / (candidate + '-' + case.name + '.npz')
    y = raw['y'][valid]
    save_arrays(prediction, indices=valid, rowids=raw['rowids'][valid], labels=y, baseline=before, v7=anchor, candidate=after)
    for p in (path, prediction):
        files[p.relative_to(work).as_posix()] = digest(p)
    row = {'case': asdict(case), 'events': len(valid), 'fit_events': len(fit),
           'maximum_meta_label_timestamp': metadata['maximum_label_timestamp'],
           **engine.point(y, before, after), 'v7_comparison': {**engine.point(y, anchor, after), 'anchor_sha256': V7_SHA},
           'models': {'calibrator': path.relative_to(work).as_posix()}, 'calibrator_metadata': metadata,
           'files': files, 'alpha': saved['alpha'], 'normalization': norm,
           'maximum_event_mass_change': float(np.abs(after.sum(1) - anchor.sum(1)).max()),
           'timings': {'total': time.monotonic() - start}}
    if case.name in ('primary', 'secondary', 'five_day'):
        for result, comparator in ((row, before), (row['v7_comparison'], anchor)):
            result.update(asdict(engine.PairedBootstrapEvaluator().evaluate(after, comparator, y, raw['u'][valid])))
            require(result['resamples'] >= 990, 'Insufficient paired bootstrap draws')
    age = (raw['t'][valid] - case.target_cutoff) / 3600
    late = (age >= 48) & (age < 120)
    if late.any():
        row['late'] = {'events': int(late.sum()), **engine.point(y[late], before[late], after[late])}
        row['v7_comparison']['late'] = {'events': int(late.sum()), **engine.point(y[late], anchor[late], after[late])}
    if case.name == 'five_day':
        require(late.any() and (int(late.sum()) == 7596 if m['spec']['purpose'] == 'research' else True), 'Missing official late cohort')
    warm = np.isin(raw['u'][valid], raw['u'][raw['t'] < case.target_cutoff])
    masks = {'warm': warm, 'cold': ~warm, 'history_empty': (raw['h'][valid] < 0).all(1), 'late': late}
    row['diagnostics'] = {k: {'events': int(v.sum()), **(engine.point(y[v], before[v], after[v])
                         if v.any() and len(np.unique(y[v])) == 2 else {})} for k, v in masks.items()}
    print('history', candidate, case.name, 'vs_V6', row['auc_diff'], 'vs_V7', row['v7_comparison']['auc_diff'], saved['alpha'], flush=True)
    return row


def final_anchor(root, work, m, number):
    from . import engine
    from predict_v7 import calibrate_v7_probabilities, load_aligned_probabilities
    evaluation = engine.events(root, m, 'evaluation')
    ids = evaluation['rowids'].ravel()
    if m['spec']['purpose'] == 'synthetic_validation':
        _, before = engine.baseline(root, m, 'baseline_evaluation', evaluation['rowids'])
    else:
        engine._private_replay(root, work, m, number)
        require(len(evaluation['u']) == len(np.unique(evaluation['u'])) == 52939
                and np.array_equal(ids, np.load(root / 'data/cache/evaluation_rowids.npy', allow_pickle=False))
                and (evaluation['t'] >= JAN20 + 7 * DAY).all(), 'Official evaluation identity differs')
        before = load_aligned_probabilities(work / 'v6-replays' / str(number) / 'artifacts/reproduced_v6.csv', ids).reshape(-1, 10)
        members = {}
        from .contracts import MEMBERS
        for member in MEMBERS:
            with np.load(root / ('artifacts/v6/' + member + '/reference_predictions.npz'), allow_pickle=False) as saved:
                members[member] = saved['combiner_predictions'].reshape(-1, 10)
        require(np.allclose(before, compose(members), rtol=0, atol=1e-15), 'Fresh V6 differs from final references')
    anchor = calibrate_v7_probabilities(before, 1.12)
    if m['spec']['purpose'] == 'research':
        from submission_export import export_submission, validate_csv
        from row_auc_final_utils import verify_csv_probabilities
        output = work / 'v6-replays' / str(number) / 'v7_anchor.csv'
        example = inside(root, m['spec']['inputs']['example']['path'])
        export_submission(ids, anchor, example, output, 529390)
        validate_csv(output, example, 529390)
        verify_csv_probabilities(output, ids, anchor)
        require(digest(output) == m['spec']['anchor_sha256'] == V7_SHA, 'Fresh V7 anchor fingerprint differs')
    return ids, anchor, evaluation


def final_probabilities(root, work, m, module, calibrator, number):
    from . import engine
    screen = engine.checked(work, 'screen')
    secondary = screen['results']['secondary']
    require(calibrator == inside(work, secondary['models']['calibrator']), 'Use the validated secondary calibrator')
    expected = deepcopy(secondary['calibrator_metadata'])
    params = deepcopy(m['spec']['candidates'][expected['candidate']])
    require(expected['spec_sha256'] == m['spec_sha256'] and expected['meta_cutoff'] == JAN20 + 5 * DAY
            and expected['maximum_label_timestamp'] < JAN20 + 6 * DAY
            and expected['parameters_sha256'] == canonical(params), 'Final calibrator lineage differs')
    saved = module.load_calibrator(calibrator, expected, params)
    ids, anchor, evaluation = final_anchor(root, work, m, number)
    raw = engine.events(root, m, 'train')
    final_graph, _ = graph(root, work, m, module, raw, JAN20 + 7 * DAY, 'final-' + str(number), use_cache=False)
    features = module.affinities(final_graph, evaluation, params['half_life_hours'])
    p = module.predict_calibrator(anchor.copy(), features, deepcopy(saved))
    require(np.allclose(p.sum(1), anchor.sum(1), rtol=0, atol=1e-14), 'Final history correction changed mass')
    return ids, p
