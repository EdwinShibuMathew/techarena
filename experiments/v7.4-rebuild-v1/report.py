"""Publish compact terminal evidence; never turn an unfinished run into a release."""
from pathlib import Path
import json
import sys

import run
import data as dataset
import study
from recipe import CUTOFFS, FULL, MEMBERS
from ml_workflow.contracts import digest, require
from ml_workflow.storage import read, write_new


def report(root):
    work = run.workspace(root)
    genesis = read(work / 'genesis.json')
    prior_report = root / 'docs/results/v7.4-rebuild-v1/results.json'
    old_hash = read(prior_report)['compute']['ledger_sha256'] if prior_report.exists() else None
    sealed = work / 'accounting_snapshots' / (old_hash + '.json') if old_hash else None
    if sealed is not None and sealed.exists():
        require(digest(sealed) == old_hash, 'Historical accounting seal changed')
        ledger = read(sealed)
        require(ledger['total_seconds'] == 43200 and all(a['status'] == 'complete' for a in ledger['attempts'])
                and abs(sum(a['seconds'] for a in ledger['attempts']) - ledger['used_seconds']) < 1e-6,
                'Historical accounting does not reconcile')
    else:
        ledger = study.accounting(work)
    require(run.completed(work, 'prepare'), 'Require the accounted official reconstruction checks')
    data = dataset.verify(work / 'data')
    stages = {}
    terminal = None
    for name in ('develop', 'screen', 'stress'):
        if run.completed(work, name):
            stages[name] = study.checked(work, name)
            if stages[name]['passed'] is False:
                terminal = 'rejected_at_' + name
                break
    ready = (work / 'readiness.json').exists()
    if ready:
        study.verify(root, work)
        terminal = 'promoted_against_rebuilt_comparators'
    require(terminal is not None, 'Do not issue terminal evidence for an ongoing campaign')
    inventory = {}
    for cutoff in CUTOFFS + [FULL]:
        for member in MEMBERS:
            if run.completed(work, f'train-{cutoff}-{member}'):
                folder = work / 'baselines' / str(cutoff) / member
                inventory[f'{cutoff}/{member}'] = {name: digest(folder / name)
                                                    for name in ('event.pt', 'preference.pt', 'positive.npz')}
    compact = {}
    for stage, value in stages.items():
        rows = {}
        for key, row in value['results'].items():
            fields = ('events', 'baseline_auc', 'candidate_auc', 'auc_diff', 'ci_lower', 'ci_upper', 'resamples', 'late')
            rows[key] = {k: row[k] for k in fields if k in row}
            rows[key]['v7_comparison'] = {k: row['v7_comparison'][k] for k in fields if k in row['v7_comparison']}
            rows[key].update(candidate=row['candidate'], case=row['case'], alpha=row['alpha'],
                             fitting_events=row['fitting_events'], prediction_sha256=row['prediction_sha256'])
        compact[stage] = {'passed': value['passed'], 'winner': value['winner'], 'results': rows,
                          'report_sha256': digest(work / 'stages' / (stage + '.json'))}
    result = {'schema_version': 5, 'status': terminal, 'baseline_lineage': 'rebuilt_v6_v7_recipe',
              'historical_v6_v7_reproduced': False, 'submission_ready': ready, 'promotion_supported': ready,
              'validation_status': 'retrospective reused dates; not independent holdout evidence',
              'genesis_sha256': digest(work / 'genesis.json'), 'generation': genesis['generation'],
              'runtime': genesis['runtime'], 'historical_accounting': genesis['historical_accounting'],
              'raw_sha256': genesis['raw_sha256'], 'data_metadata_sha256': digest(work / 'data/metadata.json'),
              'data': {'n_users': data['n_users'], 'n_items': data['n_items'],
                       'train_events': data['arrays']['train-u.npy']['shape'][0],
                       'evaluation_events': data['arrays']['evaluation-u.npy']['shape'][0],
                       'full_preference_examples_verified': 1009707},
              'compute': {'total_seconds': ledger['total_seconds'], 'used_seconds': ledger['used_seconds'],
                          'attempts': len(ledger['attempts']), 'ledger_sha256': digest(sealed if sealed is not None and sealed.exists() else work / 'compute-ledger.json'),
                          'status_counts': {name: sum(a['status'] == name for a in ledger['attempts'])
                                            for name in sorted({a['status'] for a in ledger['attempts']})}},
              'sources': {name: read(work / (name + '-source.json'))['source']['commit']
                          for name in ('baseline', 'scientific')},
              'prefix_models_trained': sum(not key.startswith(str(FULL) + '/') for key in inventory),
              'full_prefix_models_trained': sum(key.startswith(str(FULL) + '/') for key in inventory),
              'models': inventory, 'stages': compact,
              'stages_not_run': [name for name in ('screen', 'stress', 'finalize')
                                 if name not in stages and not (name == 'finalize' and ready)]}
    output = root / 'docs/results/v7.4-rebuild-v1/results.json'
    if output.exists(): require(read(output) == result, 'Preserve different terminal evidence')
    else: write_new(output, result)
    return {key: result[key] for key in ('status', 'prefix_models_trained', 'full_prefix_models_trained', 'submission_ready', 'compute')}


if __name__ == '__main__':
    print(json.dumps(report(Path(__file__).resolve().parents[2]), indent=2))
