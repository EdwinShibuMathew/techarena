"""Read-only descriptive metrics. None of these results can promote a model."""
import numpy as np

from experiment_runner import FastAuc
from ml_workflow.contracts import require, compose
from predict_v7 import calibrate_v7_probabilities


def auc(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p, dtype=np.float64).ravel()
    if len(y) == 0 or len(np.unique(y)) < 2: return None
    ids = np.arange(len(y), dtype=np.int64)
    return FastAuc(y, p, ids).value(np.ones(len(y), dtype=np.int64))


def describe(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    require(y.ndim == 2 and y.shape[1] == 10 and p.shape == y.shape
            and np.isin(y, [0, 1]).all() and (y.sum(1) <= 1).all()
            and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid descriptive metric input')
    clicked = y.sum(1) > 0; positives = p[y == 1]; zero_negatives = p[~clicked].ravel()
    within = None
    if clicked.any():
        selected = p[clicked][np.arange(clicked.sum()), y[clicked].argmax(1)]
        credit = ((selected[:, None] > p[clicked]) + .5 * (selected[:, None] == p[clicked])).sum(1) - .5
        within = float((credit / 9).mean())
    ac = auc(y[clicked], p[clicked])
    an = auc(np.r_[np.ones(len(positives)), np.zeros(len(zero_negatives))], np.r_[positives, zero_negatives])
    n_clicked_negative, n_zero_negative = int(clicked.sum()) * 9, int((~clicked).sum()) * 10
    denominator = n_clicked_negative + n_zero_negative
    reconstructed = None
    if clicked.any() and denominator:
        reconstructed = float(((ac or 0.) * n_clicked_negative + (an or 0.) * n_zero_negative) / denominator)
    # Loss stabilization is for diagnostics only; exported probabilities are never clipped.
    safe = np.clip(p, 1e-15, 1 - 1e-15)
    return {'auc': auc(y, p), 'clicked_negative_auc': ac, 'no_click_negative_auc': an,
            'negative_pair_weights': {'clicked': n_clicked_negative / denominator if denominator else 0.,
                                      'no_click': n_zero_negative / denominator if denominator else 0.},
            'reconstructed_global_auc': reconstructed, 'within_event_auc': within,
            'mean_event_score_sum': float(p.sum(1).mean()), 'event_score_auc': auc(clicked.astype(np.int8), p.sum(1)),
            'event_score_sum_gt1_fraction': float((p.sum(1) > 1).mean()),
            'row_logloss': float(-(y * np.log(safe) + (1 - y) * np.log1p(-safe)).mean()),
            'row_brier': float(((p - y) ** 2).mean())}


def component_swap(allocation_source, mass_source):
    """Score-only diagnostic, not an exportable or selectable candidate."""
    a, b = np.asarray(allocation_source, dtype=np.float64), np.asarray(mass_source, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 10
            and np.isfinite(a).all() and np.isfinite(b).all() and (a >= 0).all() and (b >= 0).all(), 'Invalid component input')
    sa, sb = a.sum(1, keepdims=True), b.sum(1, keepdims=True)
    require(not ((sa[:, 0] == 0) & (sb[:, 0] > 0)).any(), 'Undefined allocation for a zero-mass event')
    return np.divide(a, sa, out=np.zeros_like(a), where=sa > 0) * sb


def component_report(y, a, v7, v6, masks):
    output = {'diagnostic_only': True, 'promotion_supported': False, 'variants': {}}
    for key, p in [('A_allocation_V7_mass', component_swap(a, v7)), ('V7_allocation_A_mass', component_swap(v7, a))]:
        row = {'out_of_range_rows': int(((p < 0) | (p > 1)).sum())}
        for cohort, mask in masks.items():
            if mask.any() and len(np.unique(y[mask])) == 2:
                row[cohort] = {'delta_vs_V6': auc(y[mask], p[mask]) - auc(y[mask], v6[mask]),
                               'delta_vs_V7': auc(y[mask], p[mask]) - auc(y[mask], v7[mask])}
        output['variants'][key] = row
    return output


def subgroup_masks(raw, ix, target):
    age = (raw['t'][ix] - target) / 3600
    warm = np.isin(raw['u'][ix], raw['u'][raw['t'] < target])
    empty = (raw['h'][ix] < 0).all(1)
    repeat = (raw['i'][ix, :, None] == raw['h'][ix, None, :]).any(axis=(1, 2))
    masks = {'all': np.ones(len(ix), dtype=bool), 'warm': warm, 'cold': ~warm,
             'history_empty': empty, 'history_present': ~empty, 'has_repeat_candidate': repeat,
             'no_repeat_candidate': ~repeat}
    for begin, end in ((0, 6), (6, 12), (12, 18), (18, 24), (24, 48), (48, 72)):
        masks[f'age_{begin}_{end}'] = (age >= begin) & (age < end)
    return masks


def group_importance(values, names):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == len(names) == 47, 'Importance feature schema differs')
    average = values.mean(0); top = np.argsort(-average, kind='stable')[:8]
    return {'method': 'PredictionValuesChange', 'interpretation': 'descriptive, not causal attribution',
            'groups': {'original37': float(average[:37].sum()), 'graph7': float(average[37:44].sum()),
                       'age': float(average[44]), 'prefix_size': float(average[45:].sum())},
            'top8': {names[i]: float(average[i]) for i in top},
            'feature_names': list(names), 'mean_importance': average.tolist()}


def range_coverage(fitting_min, fitting_max, target, names):
    require(target.shape[1] == len(names) and np.isfinite(target).all(), 'Invalid feature-range input')
    outside = ((target < fitting_min) | (target > fitting_max)).mean(0)
    return {name: {'fit_min': float(fitting_min[i]), 'fit_max': float(fitting_max[i]),
                   'target_min': float(target[:, i].min()), 'target_max': float(target[:, i].max()),
                   'outside_fitting_range_fraction': float(outside[i])} for i, name in enumerate(names)}


def member_ablation(y, original, current, v7, masks):
    """Post-hoc fixed-weight component swaps; no member/weight is selected."""
    result = {}
    for member in original:
        solo = dict(original); solo[member] = current[member]
        leave_one = dict(current); leave_one[member] = original[member]
        rows = {}
        for key, members in [('single_replacement', solo), ('restore_one_original', leave_one)]:
            try:
                p = calibrate_v7_probabilities(compose(members), 1.12)
            except ValueError as error:
                rows[key] = {'invalid_probability_diagnostic': str(error)}
                continue
            rows[key] = {cohort: {'delta_vs_V7': auc(y[mask], p[mask]) - auc(y[mask], v7[mask])}
                         for cohort, mask in masks.items() if mask.any() and len(np.unique(y[mask])) == 2}
        result[member] = rows
    return {'diagnostic_only': True, 'promotion_supported': False, 'members': result}
