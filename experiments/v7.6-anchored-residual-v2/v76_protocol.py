"""Frozen focused successor; no outer-result tuning or neural retraining."""
import hashlib
import numpy as np
from ml_workflow.contracts import JAN20, DAY, require
from recipe import MEMBERS, COMBINER, CASES, FULL

HOUR = 3600
PREFIXES = [JAN20 + h * HOUR for h in (6, 12, 24, 48, 72, 96, 120, 144)]
BLOCKS = list(range(JAN20 + 12 * HOUR, JAN20 + 6 * DAY, 6 * HOUR))
BOUNDARIES = [JAN20 + d * DAY for d in (2, 3, 5, 6)]
INNER = [JAN20 + h * HOUR for h in (36, 42)]
PENALTIES = [.1, 1., 10.]
FEATURES = ['graph_affinity_24h', 'graph_affinity_72h', 'graph_centered_24h', 'graph_centered_72h',
            'log_graph_item_support', 'graph_history_coverage', 'graph_edge_coverage',
            'anchor_row_logit', 'log1p_anchor_event_score_sum', 'anchor_conditional_probability',
            'anchor_conditional_entropy', 'history_length_over_20', 'unseen_user']
SPEC = {'schema_version': 1, 'design_id': 'v7.6-v7-anchored-residual-v2', 'purpose': 'research',
        'hypothesis': 'Bounded V7-anchored graph residuals trained on causal stale views improve global row AUC',
        'baseline_lineage': 'rebuilt_v6_v7_recipe', 'release_policy': 'v6_v7_four_gate',
        'candidate_order': ['residual'], 'features': FEATURES, 'prefixes': PREFIXES,
        'blocks': BLOCKS, 'boundaries': BOUNDARIES, 'inner_blocks': INNER, 'penalties': PENALTIES,
        'hidden': 16, 'radius': .25, 'gamma': 1.12, 'epochs': 32, 'learning_rate': .001,
        'weight_decay': .01, 'batch_events': 2048, 'seed': 2026, 'pair_seed': 5026, 'view_seed': 6026,
        'bce_weight': .5, 'pair_weight': .5, 'negative_rows_per_positive': 16,
        'dtype': 'float64', 'device': 'cpu', 'threads': 4, 'early_stopping': False,
        'initialization': 'Xavier_uniform_hidden_gain1_bias0_output0', 'folds': 5,
        'normalization': {'scale_floor': 1e-6, 'standardized_bound': 5., 'feature_logit_epsilon': 1e-6},
        'history_hours': 48, 'pair_gap_hours': [6, 48], 'maximum_prefix_age_hours': 168,
        'fitting_end': 'min(c+24h,d)', 'event_weight': 'one_uniform_view_per_event_per_epoch',
        'inner_regimes': ['freshest', 'oldest'], 'inner_selection': 'maximin_ci_lower_nonnegative_all_positive_mean_each',
        'bootstrap': {'resamples': 1000, 'seed': 2026, 'minimum_valid': 990},
        'development': [{'name': n, 'meta_cutoff': CASES[n][0], 'target_cutoff': CASES[n][1],
                         'target_end': CASES[n][2]} for n in ('jan22_long', 'jan24_forward')],
        'comparators': ['V6', 'V7'], 'total_seconds': 43200, 'finalization_reserve_seconds': 3600,
        'training_job_seconds': 900, 'other_job_seconds': 1200,
        'validation_status': 'retrospective reused dates; not independent holdout evidence',
        'upload': 'manual', 'retrain_neural': False, 'event_mass_constraint': 'none'}


def pairs(block):
    require(block in BLOCKS, 'Undeclared forecast block')
    return [(c, d, min(c + DAY, d)) for c in PREFIXES for d in PREFIXES
            if 6 * HOUR <= d - c <= 48 * HOUR and d <= block and block - d < 168 * HOUR]


def validation_pairs(block):
    available = pairs(block); result = {}
    for name, choose in [('freshest', max), ('oldest', min)]:
        d = choose(v[1] for v in available)
        result[name] = max((v for v in available if v[1] == d), key=lambda v: v[0])
    return result


def user_folds(users):
    users = np.asarray(users)
    unique, inverse = np.unique(users, return_inverse=True)
    codes = np.array([int.from_bytes(hashlib.blake2b(('2026/' + str(int(u))).encode(), digest_size=8).digest(), 'little') % 5
                      for u in unique], dtype=np.int8)
    return codes[inverse]


def fitting_indices(raw, boundary, held_fold=None):
    """Forecast-eligible first event/user in the fixed historical window."""
    require(boundary in BOUNDARIES + INNER, 'Undeclared residual fitting boundary')
    from ml_workflow.contracts import first_indices
    ix = first_indices(raw['u'], raw['t'], max(BLOCKS[0], boundary - 2 * DAY), boundary)
    if held_fold is not None:
        require(type(held_fold) is int and 0 <= held_fold < 5, 'Invalid user fold')
        ix = ix[user_folds(raw['u'][ix]) != held_fold]
    require(len(ix) and (raw['t'][ix] < boundary).all(), 'Empty or noncausal residual fitting records')
    return ix
