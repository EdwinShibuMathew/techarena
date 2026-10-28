"""Four preregistered ablations; original eight gates plus historical floors."""
import hashlib
import numpy as np
from ml_workflow.contracts import DAY, JAN20, require
from recipe import MEMBERS, COMBINER, CASES, FULL

MATURE = [JAN20 + d * DAY for d in (1, 2, 3, 4, 5, 6)]
METAS = [JAN20 + d * DAY for d in (1, 2, 4, 5)]
CANDIDATES = {'A': {'event': 4, 'row': False}, 'B': {'event': 11, 'row': False},
              'C': {'event': 4, 'row': True}, 'D': {'event': 11, 'row': True}}
WEIGHTS = np.array([2/9, 2/9, 2/9, 1/6, 1/6], dtype=np.float64)
EVENT_NAMES = ['log1p_anchor_sum', 'conditional_entropy', 'maximum_conditional', 'member_logit_rms',
               'log1p_user_events', 'smoothed_user_ctr', 'log1p_user_clicks', 'log1p_activity_gap',
               'history_length_over20', 'log1p_history_gap', 'unseen_user']
ROW_NAMES = [f'member_logit_deviation_{m}' for m in MEMBERS] + [f'member_rank_deviation_{m}' for m in MEMBERS] + [
    'slate_mean_cosine', 'slate_max_cosine', 'slate_weighted_cosine']
FLOOR_PROFILE = 'dual_anchor_plus_historical_floors_v1'
FLOOR_SOURCES = {
    'primary': ('docs/results/v7.3-v7-anchor-v1/screen.json', ['results', 'primary'], 43964, .8187833364842867),
    'secondary': ('docs/results/v7.3-v7-anchor-v1/screen.json', ['results', 'secondary'], 35138, .8275321389476477),
    'five_day': ('docs/results/v7.3-v7-anchor-v1/stress.json', ['results', 'five_day'], 51488, .7822476123338514),
    'late': ('docs/results/v8-seed44-row-auc/stress.json', ['results', 'late_48_120h'], 7596, .7395732277865564)}
SPEC = {'schema_version': 1, 'purpose': 'research', 'design_id': 'v7.7-reliability-slate-v1',
        'hypothesis': 'First-return user reliability and candidate slate evidence improve global row ranking',
        'baseline_lineage': 'rebuilt_v6_v7_recipe', 'release_policy': 'v6_v7_four_gate', 'release_profile': FLOOR_PROFILE,
        'candidate_order': list(CANDIDATES), 'candidates': CANDIDATES, 'mature_prefixes': MATURE, 'metadata_days': METAS,
        'event_features': EVENT_NAMES, 'row_features': ROW_NAMES, 'normalization_scale_floor': 1e-6,
        'standardized_bound': 5., 'feature_probability_epsilon': 1e-6,
        'event_radius': .25, 'row_raw_radius': .125, 'initialization': 'zero',
        'objective': {'bce': .5, 'global_pair': .5, 'l2_sum': .01, 'negatives_per_positive': 16},
        'optimizer': {'method': 'L-BFGS-B', 'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
        'dtype': 'float64', 'device': 'cpu', 'threads': 4, 'folds': 5, 'fold_seed': 2026, 'pair_seed': 5026,
        'training_population': 'first event per user since each prefix; query in metadata day',
        'strata': 'prefix lag by day and query six-hour bin; equal nonempty stratum weight; per-event sum<=1',
        'teacher_population': 'metadata-day first users; held users excluded across every view',
        'bootstrap': {'resamples': 1000, 'seed': 2026, 'minimum_valid': 990},
        'development': [{'name': n, 'meta_cutoff': CASES[n][0], 'target_cutoff': CASES[n][1], 'target_end': CASES[n][2]}
                        for n in ('jan22_long', 'jan24_forward')],
        'phase_limits': {'prepare': 5000, 'develop': 8000, 'promotion': 10000},
        'total_seconds': 43200, 'finalization_reserve_seconds': 3600, 'training_job_seconds': 900,
        'other_job_seconds': 1200, 'retrain_neural': False, 'upload': 'manual',
        'validation_status': 'retrospective reused dates; historical floors are point thresholds, not paired historical tests'}


def prefixes(meta):
    require(meta in METAS, 'Undeclared metadata role')
    return [p for p in MATURE if 0 <= meta - p <= 2 * DAY]


def folds(users):
    unique, inverse = np.unique(users, return_inverse=True)
    values = [int.from_bytes(hashlib.blake2b(('2026/' + str(int(u))).encode(), digest_size=8).digest(), 'little') % 5 for u in unique]
    return np.asarray(values, dtype=np.int8)[inverse]
