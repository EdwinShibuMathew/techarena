"""Verify and record a terminal V7.5 outcome and the actual delivered CSV."""
from pathlib import Path
import json

import campaign
import pipeline
from protocol import MEMBERS, FULL, PROTOCOL, EXTRA_CUTOFFS
from ml_workflow.contracts import canonical, digest, inside, require
from ml_workflow.storage import read, write_new


def report(root):
    work = campaign.workdir(root); parent = root / 'artifacts/rebuild'
    pipeline.verify(root, work)
    destination = root / 'docs/results/v7.5-history-age-v1/results.json'
    if destination.exists():
        old_hash = read(destination)['compute']['ledger_sha256']
        accounting_path = work / 'accounting_snapshots' / (old_hash + '.json')
        require(digest(accounting_path) == old_hash, 'Terminal accounting seal changed')
        ledger = read(accounting_path)
    else:
        accounting_path = parent / 'compute-ledger.json'
        ledger = read(accounting_path)
    require(not any(a['status'] == 'running' for a in ledger['attempts']), 'Wait for final accounting')
    require(ledger['total_seconds'] == 43200 and ledger['carry_identity'] == digest(parent / 'genesis.json')
            and abs(sum(a['seconds'] for a in ledger['attempts']) - ledger['used_seconds']) < 1e-6
            and ledger['used_seconds'] <= 43200, 'Accounting does not reconcile')
    manifest = read(work / 'manifest.json'); compact = {}; status = None
    for name, expected in manifest['inherited_models'].items():
        require(digest(inside(root, name)) == expected, 'Inherited asset changed')
    for pattern in ('baseline-export-*.json', 'export-*.json'):
        for path in (work / 'tasks').glob(pattern): require(campaign.completed(work, path.stem), 'Replay evidence changed')
    for stage in ('develop', 'screen', 'stress'):
        if not campaign.completed(work, stage): break
        record = pipeline.checked(work, stage); rows = {}
        for key, row in record['results'].items():
            fields = ('status', 'case', 'candidate', 'events', 'baseline_auc', 'candidate_auc', 'auc_diff', 'ci_lower', 'ci_upper', 'resamples', 'late', 'reason')
            rows[key] = {k: row[k] for k in fields if k in row}
            if 'v7_comparison' in row: rows[key]['v7_comparison'] = row['v7_comparison']
        compact[stage] = {'passed': record['passed'], 'winner': record['winner'], 'results': rows,
                          'report_sha256': digest(work / 'stages' / (stage + '.json'))}
        if record['passed'] is False:
            status = 'rejected_at_' + stage; break
    promoted = (work / 'readiness.json').exists()
    if promoted:
        pipeline.verify_gates(work); status = 'promoted_against_rebuilt_comparators'
    if status is None and 43200 - ledger['used_seconds'] < 4800: status = 'stopped_at_budget_reserve'
    require(status is not None, 'Do not issue terminal evidence for an ongoing campaign')
    selected = read(work / ('readiness.json' if promoted else 'baseline-delivery.json'))
    require(digest(inside(root, selected['submission'])) == selected['sha256'], 'Delivered CSV changed')
    models = {}
    for cutoff in EXTRA_CUTOFFS + [FULL]:
        for member in MEMBERS:
            task = f'train-{cutoff}-{member}'
            if campaign.completed(work, task):
                folder = work / 'models' / str(cutoff) / member
                models[f'{cutoff}/{member}'] = {name: digest(folder / name) for name in ('event.pt', 'preference.pt', 'positive.npz')}
    new_attempts = [a for a in ledger['attempts'] if a['owner'].split('/', 1)[-1].startswith('v75-')]
    source = {role: read(work / (role + '-source.json'))['source']['commit']
              for role in ('baseline', 'scientific') if (work / (role + '-source.json')).exists()}
    result = {'schema_version': 6, 'status': status, 'v7_5_promoted': promoted,
              'competition_csv_ready': True, 'recommended_submission': selected['submission'],
              'recommended_sha256': selected['sha256'], 'model_role': selected['model_role'],
              'baseline_lineage': 'rebuilt_v6_v7_recipe', 'historical_v6_v7_reproduced': False,
              'validation_status': 'retrospective reused dates; not independent holdout evidence',
              'manifest_sha256': digest(work / 'manifest.json'), 'protocol_sha256': canonical(PROTOCOL),
              'sources': source, 'baseline_delivery': read(work / 'baseline-delivery.json'),
              'compute': {'total_seconds': 43200, 'starting_used_seconds': manifest['starting_used_seconds'],
                          'new_seconds': sum(a['seconds'] for a in new_attempts), 'used_seconds': ledger['used_seconds'],
                          'ledger_sha256': digest(accounting_path), 'v75_attempts': len(new_attempts),
                          'status_counts': {s: sum(a['status'] == s for a in new_attempts) for s in sorted({a['status'] for a in new_attempts})}},
              'inherited_models': manifest['inherited_models'], 'new_models': models, 'stages': compact,
              'stage_not_run': [s for s in ('screen', 'stress', 'finalize') if s not in compact and not (s == 'finalize' and promoted)],
              'competition_upload_performed': False}
    sealed = work / 'accounting_snapshots' / (result['compute']['ledger_sha256'] + '.json')
    if not sealed.exists(): write_new(sealed, ledger)
    require(digest(sealed) == result['compute']['ledger_sha256'], 'Accounting snapshot serialization differs')
    if destination.exists(): require(read(destination) == result, 'Preserve a different terminal report')
    else: write_new(destination, result)
    delivery = {'schema_version': 6, 'csv_valid': True, 'replay_verified': True,
                'model_role': selected['model_role'], 'v7_5_promoted': promoted,
                'submission': selected['submission'], 'sha256': selected['sha256'],
                'receipt': selected['submission'] + '.receipt.json', 'checksum': selected['submission'] + '.sha256'}
    path = destination.parent / 'delivery.json'
    if path.exists(): require(read(path) == delivery, 'Preserve different delivery evidence')
    else: write_new(path, delivery)
    return {k: result[k] for k in ('status', 'v7_5_promoted', 'competition_csv_ready', 'recommended_submission', 'recommended_sha256', 'compute')}


if __name__ == '__main__': print(json.dumps(report(Path(__file__).resolve().parents[2]), indent=2))
