"""Predeclared V7.5 family. Changing it creates a different experiment."""
from recipe import MEMBERS, COMBINER, DAY, JAN20, FULL, CASES

EXTRA_CUTOFFS = [JAN20 + 6 * 3600, JAN20 + 12 * 3600, JAN20 + 3 * DAY]
PREFIXES = sorted(EXTRA_CUTOFFS + [JAN20 + d * DAY for d in (1, 2, 4, 5, 6)])
ADDED_NAMES = ['graph_affinity_24h', 'graph_affinity_72h', 'graph_centered_24h',
               'graph_centered_72h', 'log_graph_item_support', 'graph_history_coverage',
               'graph_edge_coverage', 'log_prefix_age_hours', 'log_prefix_events', 'log_prefix_users']
CANDIDATES = {'A': {'pool': False, 'gamma': 1.12},
              'B': {'pool': True, 'gamma': 1.},
              'C': {'pool': True, 'gamma': 1.12}}
PROTOCOL = {'schema_version': 6, 'mode': 'history_age_combiner',
            'baseline_lineage': 'rebuilt_v6_v7_recipe', 'release_policy': 'v6_v7_four_gate',
            'hypothesis': 'Graph evidence and causal stale-prefix training improve aged transfer without losing near-window gains',
            'members': {k: list(v) for k, v in MEMBERS.items()}, 'combiner': COMBINER,
            'extra_prefixes': EXTRA_CUTOFFS, 'prefixes': PREFIXES, 'added_features': ADDED_NAMES,
            'candidates': CANDIDATES, 'candidate_order': list(CANDIDATES),
            'pool_maximum_prefix_gap_hours': 96, 'age_feature_cap_hours': 168,
            'replica_weights': 'one_over_number_of_prefixes', 'finalization_reserve_seconds': 3600,
            'bootstrap': {'resamples': 1000, 'seed': 2026, 'minimum_valid': 990},
            'development': [{'name': name, 'meta_cutoff': CASES[name][0], 'target_cutoff': CASES[name][1],
                             'target_end': CASES[name][2]} for name in ('jan22_long', 'jan24_forward')]}


def fitting_prefixes(meta, pooled):
    return [c for c in PREFIXES if 0 <= meta - c <= 4 * DAY] if pooled else [meta]
