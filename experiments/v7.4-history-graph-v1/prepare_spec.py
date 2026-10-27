"""Check restored assets read-only, then freeze V7.4 only in the authoritative workspace."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ml_workflow.contracts import ExperimentSpec, digest, inside, require
from ml_workflow.history_postprocessing import preflight
from ml_workflow.storage import read, write_new
from adapter import CANDIDATES


def check(root=ROOT):
    root = Path(root).resolve()
    spec = ExperimentSpec.parse(read(root / 'experiments/v7.4-history-graph-v1/experiment.template.json')).value
    require(spec['candidates'] == CANDIDATES, 'Template and frozen adapter differ')
    missing = {name: asset['path'] for name, asset in spec['inputs'].items()
               if not inside(root, asset['path']).is_file()}
    for name, path in [('v6_bundle', 'artifacts/v6/manifest.json'),
                       ('authoritative_ledger', 'artifacts/experiments/compute-ledger.json')]:
        if not inside(root, path).is_file(): missing[name] = path
    result = {'ready_for_prepare': False, 'missing': missing,
              'scientific_run_performed': False, 'promotion_supported': False}
    if not missing:
        try:
            result['accounting'] = preflight(root, spec)
            for asset in spec['inputs'].values():
                require(digest(inside(root, asset['path'])) == asset['sha256'], 'Restored input fingerprint differs')
            result['ready_for_prepare'] = True
        except (ValueError, OSError) as error:
            result['error'] = str(error)
    return spec, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--freeze', action='store_true', help='Write a new committed-input specification after successful checks')
    args = parser.parse_args()
    spec, result = check(args.root)
    if args.freeze:
        require(result['ready_for_prepare'], 'Restore and reconcile the private assets/ledger before freezing research')
        destination = args.root.resolve() / 'experiments/v7.4-history-graph-v1/experiment.json'
        write_new(destination, spec)
        result['specification'] = str(destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['ready_for_prepare'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
