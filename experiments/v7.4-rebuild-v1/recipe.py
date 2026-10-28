"""Predeclared from-raw reconstruction; historical baseline identities stay intact."""
MEMBERS = {'dim8_seed42': (8, 42, 260, 'mixed'),
           'dim8_seed43': (8, 43, 260, 'mixed'),
           'dim8_seed44': (8, 44, 260, 'ce'),
           'dim16_seed42': (16, 42, 196, 'ce'),
           'dim16_seed43': (16, 43, 196, 'ce')}
DAY = 86400
JAN20 = 1737331200
CUTOFFS = [JAN20 + d * DAY for d in (1, 2, 4, 5, 6)]
FULL = JAN20 + 7 * DAY
TOTAL_SECONDS = 43200
COMBINER = {'iterations': 545, 'depth': 6, 'learning_rate': .05, 'l2_leaf_reg': 20,
            'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': 42,
            'thread_count': 4, 'allow_writing_files': False, 'use_best_model': False,
            'boosting_type': 'Plain', 'bootstrap_type': 'MVS', 'border_count': 254,
            'feature_border_type': 'GreedyLogSum', 'grow_policy': 'SymmetricTree',
            'leaf_estimation_backtracking': 'AnyImprovement', 'leaf_estimation_iterations': 10,
            'leaf_estimation_method': 'Newton', 'min_data_in_leaf': 1, 'model_size_reg': .5,
            'nan_mode': 'Min', 'random_score_type': 'NormalWithModelSizeDecrease',
            'random_strength': 1., 'rsm': 1., 'sampling_frequency': 'PerTree',
            'score_function': 'Cosine', 'subsample': .8}
CASES = {
    'jan22_long': (JAN20 + DAY, JAN20 + 2 * DAY, JAN20 + 5 * DAY),
    'jan24_forward': (JAN20 + 2 * DAY, JAN20 + 4 * DAY, JAN20 + 5 * DAY),
    'primary': (JAN20 + 4 * DAY, JAN20 + 5 * DAY, FULL),
    'secondary': (JAN20 + 5 * DAY, JAN20 + 6 * DAY, FULL),
    'five_day': (JAN20 + DAY, JAN20 + 2 * DAY, FULL),
}
RECIPE = {'schema_version': 5, 'baseline_lineage': 'rebuilt_v6_v7_recipe',
          'members': {k: list(v) for k, v in MEMBERS.items()},
          'preference': {'initial_epochs': 196, 'initial_lr': .012, 'fine_tune_lr': .003,
                         'fine_tune_seed_offset': 1000, 'batch_size': 4096,
                         'negative_items': 31, 'regularization': .02},
          'event': {'epochs': 8, 'lr': .0015, 'batch_size': 2048,
                    'anchor_regularization': .1, 'effect_regularization': .05,
                    'pair_sampling_seed_offset': 3000},
          'combiner': COMBINER, 'gamma': 1.12, 'total_seconds': TOTAL_SECONDS,
          'bootstrap': {'resamples': 1000, 'seed': 2026, 'minimum_valid': 990},
          'release_policy': 'v6_v7_four_gate', 'candidate_order': ['history_24h', 'history_72h'],
          'hypothesis': 'Causal co-click neighbors improve rebuilt fixed V7 without late-window regression',
          'old_ledger_lost': True, 'original_fingerprints_required': False}
