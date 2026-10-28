"""A frozen design artifact only. This module contains no fitting entry point."""
from ml_workflow.contracts import require, canonical, JAN20, DAY

FEATURES = ['graph_affinity_24h', 'graph_affinity_72h', 'graph_centered_24h', 'graph_centered_72h',
            'log_graph_item_support', 'graph_history_coverage', 'graph_edge_coverage',
            'anchor_row_logit', 'log1p_anchor_event_score_sum', 'anchor_conditional_probability',
            'anchor_conditional_entropy', 'history_length_over_20', 'unseen_user']


def design():
    return {'schema_version': 1, 'design_id': 'v7.6-v7-anchored-residual-v1', 'status': 'design_only_not_executed',
            'training_authorized_by_this_audit': False, 'promotion_supported': False,
            'hypothesis': 'A bounded V7-anchored graph correction limits cross-event distortion while retaining useful graph evidence',
            'anchor': {'lineage': 'rebuilt_v6_v7_recipe', 'version': 'V7', 'gamma': 1.12,
                       'replace_original_combiners': False, 'retrain_neural_models': False},
            'features': FEATURES, 'excluded_features': ['raw_prefix_age', 'global_prefix_event_count', 'global_prefix_user_count', 'calendar_date'],
            'network': {'hidden_layers': [16], 'activation': 'tanh', 'output_activation': 'tanh',
                        'log_odds_radius': .25, 'output_initialization': 'zero_weights_and_bias',
                        'formula': 'sigmoid(logit(anchor) + 0.25*tanh(network(features)))',
                        'zero_output_returns_exact_anchor': True, 'preserve_zero_one_endpoints': True,
                        'event_mass_constraint': 'none',
                        'identity_event_mask': 'history_length==0 OR max(graph_edge_coverage across ten candidates)==0'},
            'normalization': {'fit_on_training_records_only': True, 'scale_floor': 1e-6,
                              'standardized_feature_bound': 5., 'feature_logit_epsilon': 1e-6,
                              'entropy_normalizer': 'log(10)', 'event_score_sum_is_probability': False},
            'training': {'epochs': 32, 'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .01,
                         'batch_events': 2048, 'seed': 2026, 'pair_seed': 5026, 'early_stopping': False,
                         'bce_weight': .5, 'global_row_pair_weight': .5, 'negative_rows_per_positive': 16,
                         'negative_pool': 'all unclicked candidate rows in the event minibatch, including other users',
                         'anchor_divergence': 'mean Bernoulli KL(anchor || corrected) over rows',
                         'divergence_penalties': [.1, 1., 10.]},
            'historical_forecasts': {'block_hours': 6, 'first_block_start': JAN20 + 12 * 3600,
                 'combiner_neural_prefix': 'latest available cutoff c <= block_start - 6h',
                 'combiner_fitting_interval': '[c, block_start)',
                 'prediction_neural_prefix': 'latest available cutoff d <= block_start',
                 'graph_prefix': 'd', 'target_interval': '[block_start, block_start + 6h)',
                 'first_user_per_block': True, 'require_all_fitting_labels_before_target': True},
            'inner_selection': {'validation_blocks': [[JAN20 + DAY + h * 3600, JAN20 + DAY + (h + 6) * 3600] for h in (12, 18)],
                 'user_folds': 5, 'group_seed': 2026, 'exclude_validation_users_from_all_prior_residual_fitting_records': True,
                 'fitting_records': 'only historical forecasts whose labels precede validation block start',
                 'eligibility': 'nonnegative global-row AUC differences vs both anchors in both blocks; positive mean vs each',
                 'selection': 'maximum minimum paired 95% lower bound over blocks and anchors',
                 'tie_break': 'largest divergence penalty within 1e-12 of best score',
                 'if_none_eligible': 'stop before outer evaluation', 'freeze_one_penalty_before_Jan22_outer_results': True},
            'outer_fitting': {'record_window_hours': 48, 'label_end': 'metadata_cutoff + 24h',
                 'first_record_per_user_in_window': True, 'normalization_fitting_records_only': True,
                 'chronology': 'same development and promotion cutoffs as V7.5',
                 'outer_hyperparameter_updates': False, 'runner_up_after_failure': False},
            'acceptance': {'release_policy': 'v6_v7_four_gate', 'comparators': ['rebuilt V6', 'rebuilt V7'],
                 'bootstrap': {'resamples': 1000, 'seed': 2026, 'minimum_valid': 990},
                 'primary_and_five_day_ci_lower': '>0', 'secondary_and_late_auc_difference': '>=0',
                 'validation_status': 'retrospective reused dates; not independent holdout evidence',
                 'promotion_verifier': 'ml_workflow.promotion.verify_promotion',
                 'identity_alone_qualifies': False},
            'budget': {'ledger': 'artifacts/rebuild/compute-ledger.json', 'total_seconds': 43200,
                       'extend_or_reset': False, 'finalization_reserve_seconds': 3600,
                       'training_job_seconds': 900, 'other_job_seconds': 1200},
            'delivery': {'retain_existing_verified_baseline': True, 'competition_upload': 'manual'}}


def forecast_schedule(block_start, prefixes):
    c = max((p for p in prefixes if p <= block_start - 6 * 3600), default=None)
    d = max((p for p in prefixes if p <= block_start), default=None)
    require(c is not None and d is not None and c < block_start and c <= d <= block_start, 'Insufficient causal forecast prefixes')
    return {'combiner_prefix': c, 'fit_start': c, 'fit_end': block_start, 'prediction_prefix': d,
            'target_start': block_start, 'target_end': block_start + 6 * 3600}


def validate_design(value):
    require(value == design(), 'Successor design differs from the frozen proposal')
    require(len(value['features']) == len(set(value['features'])) == 13, 'Incorrect successor feature schema')
    require(value['training_authorized_by_this_audit'] is False and value['promotion_supported'] is False,
            'A design document cannot authorize fitting or claim promotion')
    end = JAN20 + 2 * DAY
    require(all(b[1] <= end for b in value['inner_selection']['validation_blocks']), 'Inner selection crosses earliest outer target')
    return {'valid': True, 'design_sha256': canonical(value), 'training_executed': False}
