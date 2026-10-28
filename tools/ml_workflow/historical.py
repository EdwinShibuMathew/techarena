"""Frozen, retrospective point thresholds; never paired historical claims."""
from .contracts import require

PROFILE = 'dual_anchor_plus_historical_floors_v1'
FLOORS = {
    'primary': ('docs/results/v7.3-v7-anchor-v1/screen.json', ['results', 'primary'], 43964, .8187833364842867),
    'secondary': ('docs/results/v7.3-v7-anchor-v1/screen.json', ['results', 'secondary'], 35138, .8275321389476477),
    'five_day': ('docs/results/v7.3-v7-anchor-v1/stress.json', ['results', 'five_day'], 51488, .7822476123338514),
    'late': ('docs/results/v8-seed44-row-auc/stress.json', ['results', 'late_48_120h'], 7596, .7395732277865564)}


def verify_floors(manifest, artifact, read, source_record):
    require(manifest['schema_version'] == 1 and manifest['profile'] == PROFILE
            and set(manifest['floors']) == set(FLOORS), 'Historical floor profile differs')
    for name, (path, keys, count, value) in FLOORS.items():
        row = manifest['floors'][name]
        require(row == {'path': path, 'keys': keys, 'events': count, 'auc': value,
                        'sha256': source_record['files'][path]}, 'Historical floor definition differs')
        source = read(artifact(path))
        for key in keys: source = source[key]
        require(source['candidate_auc'] == value, 'Historical point threshold differs')
    return {name: row['auc'] for name, row in manifest['floors'].items()}


def floor_gates(comparisons, floors):
    return {name: bool(comparisons[name]['V6']['candidate_auc'] > value) for name, value in floors.items()}
