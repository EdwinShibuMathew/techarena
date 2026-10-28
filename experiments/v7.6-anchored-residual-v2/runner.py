"""Accounted V7.6 campaign. Scientific failures are terminal; uploads are manual."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'legacy'), str(ROOT / 'experiments/v7.5-history-age-v1'),
                str(ROOT / 'experiments/v7.4-rebuild-v1'), str(ROOT / 'experiments/v7.5-audit')]
from ml_workflow.contracts import canonical, digest, inside, require
from ml_workflow.storage import read, write_new, verify_snapshot
from ml_workflow import engine
import run as rebuild
import campaign as old_campaign
import audit as old_audit
import data as dataset
from v76_protocol import SPEC, PREFIXES, MEMBERS, FULL

RELATIVE = 'experiments/v7.6-anchored-residual-v2'
WORK = 'artifacts/v7_6'


def initialize(root):
    work = root / WORK
    if (work / 'manifest.json').exists():
        m = read(work / 'manifest.json')
        require(m['protocol'] == SPEC and m['genesis_sha256'] == digest(root / 'artifacts/rebuild/genesis.json'), 'V7.6 generation changed')
        return work, m
    # Called only after acquiring/reserving the initial preflight lease.
    report, _ = old_audit.verify(root)
    old = root / 'artifacts/v7_5'; folders, files = {}, {}
    from pipeline import model_paths
    for cutoff in PREFIXES + [FULL]:
        for member in MEMBERS:
            folder = model_paths(root, old, cutoff, member)
            folders[str(cutoff) + '/' + member] = folder.relative_to(root).as_posix()
            for name in ('event.pt', 'preference.pt', 'positive.npz'):
                path = folder / name; files[path.relative_to(root).as_posix()] = digest(path)
    parent = root / 'artifacts/rebuild'; ledger = read(parent / 'compute-ledger.json')
    engineering = root / 'artifacts/v7_6-engineering/source-readiness.json'
    require(engineering.is_file(), 'Provide successful frozen-source engineering receipt before preflight')
    m = {'schema_version': 1, 'protocol': SPEC, 'genesis_sha256': digest(parent / 'genesis.json'),
         'data_sha256': digest(parent / 'data/metadata.json'), 'foundation_files': files,
         'foundation_folders': folders, 'protected_inputs': report['protected_inputs'],
         'audit_report_sha256': digest(root / 'docs/results/v7.5-audit/audit.json'),
         'prior_attempts_sha256': canonical(ledger['attempts'][:-1]), 'prior_attempts': len(ledger['attempts']) - 1,
         'engineering_receipt': engineering.relative_to(root).as_posix(), 'engineering_sha256': digest(engineering)}
    write_new(work / 'manifest.json', m)
    return work, m


def completed(root, work, task):
    path = work / 'tasks' / (task + '.json')
    if not path.exists(): return False
    record = read(path); token = record['token']
    require(record['accounted'] is True and record['manifest_sha256'] == digest(work / 'manifest.json'), 'V7.6 task generation differs')
    validation = work / 'attempts' / (token + '.validated.json')
    require(record['validation_sha256'] == digest(validation), 'V7.6 task validation changed')
    claim = read(work / 'attempts' / (token + '.claim.json')); valid = read(validation)
    require(claim['task'] == task and claim['manifest_sha256'] == valid['manifest_sha256'] == record['manifest_sha256']
            and valid['source_sha256'] == claim['source']['sha256']
            and valid['claim_sha256'] == digest(work / 'attempts' / (token + '.claim.json'))
            and claim['source'] == read(work / 'scientific-source.json')['source'], 'V7.6 task source differs')
    genesis = read(root / 'artifacts/rebuild/genesis.json')
    attempts = read(root / 'artifacts/rebuild/compute-ledger.json')['attempts']
    require(any(a['owner'] == genesis['generation'] + '/v76-' + token and a['status'] == 'complete' for a in attempts), 'Task lacks settled compute ownership')
    for name, expected in valid['files'].items(): require(digest(inside(work, name)) == expected, 'V7.6 task output changed: ' + name)
    return True


def launch(root, action, task, arguments=None, limit=1200, final=False):
    work = root / WORK
    if (work / 'manifest.json').exists() and completed(root, work, task): return
    for path in [work / 'selection.json', *[work / 'stages' / (s + '.json') for s in ('develop', 'screen', 'stress')]]:
        require(not path.exists() or read(path)['passed'], 'Scientific failure is terminal; no new claims')
    require(action == 'preflight' or completed(root, work, 'preflight'), 'Preflight must complete first')
    parent = root / 'artifacts/rebuild'; genesis = read(parent / 'genesis.json')
    remaining = 43200 - read(parent / 'compute-ledger.json')['used_seconds']
    require(final or remaining - limit >= 3600, 'Protect finalization reserve')
    token = task + '-' + str(time.time_ns())
    with rebuild.ReconstructionBudget(parent / 'compute-ledger.json', genesis['generation'] + '/v76-' + token,
                                      limit, 43200, carry_identity=digest(parent / 'genesis.json')) as budget:
        require(final or budget.state['used_seconds'] <= 43200 - 3600,
                'Protect finalization reserve after acquiring the shared ledger lock')
        work, _ = initialize(root)
        pointer = work / 'scientific-source.json'
        if pointer.exists():
            locked = read(pointer); source = locked['source']; folder = inside(work, locked['snapshot_relative']); verify_snapshot(folder, source)
        else:
            folder, source = rebuild.freeze(root, work)
            write_new(pointer, {'source': source, 'snapshot_relative': folder.relative_to(work).as_posix()})
        claim = {'task': task, 'action': action, 'arguments': arguments or {}, 'token': token,
                 'manifest_sha256': digest(work / 'manifest.json'), 'source': source,
                 'snapshot_relative': folder.relative_to(work).as_posix()}
        write_new(work / 'attempts' / (token + '.claim.json'), claim)
        validation = work / 'attempts' / (token + '.validated.json')
        # Fully validated identical outputs can survive a failed final accounting write.
        recovered = None
        for prior in reversed(budget.state['attempts'][:-1]):
            owner = prior['owner'].split('/', 1)[-1]
            if not owner.startswith('v76-'): continue
            old_token = owner[4:]; old_path = work / 'attempts' / (old_token + '.validated.json')
            if not old_path.exists(): continue
            old_claim = read(work / 'attempts' / (old_token + '.claim.json'))
            if action not in ('export', 'verify-release') and all(old_claim[k] == claim[k] for k in ('task', 'action', 'arguments', 'manifest_sha256', 'source')):
                recovered = read(old_path)
                require(recovered['claim_sha256'] == digest(work / 'attempts' / (old_token + '.claim.json')), 'Recovery claim changed')
                for name, expected in recovered['files'].items(): require(digest(inside(work, name)) == expected, 'Recovery output changed')
                recovered = {**recovered, 'technical_recovery_from': old_token,
                             'claim_sha256': digest(work / 'attempts' / (token + '.claim.json'))}; break
        if recovered: write_new(validation, recovered)
        else:
            command = [sys.executable, str(folder / 'source' / RELATIVE / 'runner.py'), action,
                       '--root', str(root), '--worker', '--token', token]
            require(budget.run(command, folder / 'source') == 0, 'V7.6 worker failed; preserve charges and evidence')
        valid = read(validation)
        require(valid['manifest_sha256'] == claim['manifest_sha256'] and valid['source_sha256'] == source['sha256']
                and valid['claim_sha256'] == digest(work / 'attempts' / (token + '.claim.json')), 'Worker source differs')
        for name, expected in valid['files'].items(): require(digest(inside(work, name)) == expected, 'Worker output changed')
    write_new(work / 'tasks' / (task + '.json'), {'token': token, 'accounted': True,
              'manifest_sha256': claim['manifest_sha256'], 'validation_sha256': digest(validation)})
    print('completed', task, flush=True)


def worker(root, action, token):
    work = root / WORK; parent = root / 'artifacts/rebuild'
    claim = read(work / 'attempts' / (token + '.claim.json')); m = read(work / 'manifest.json')
    genesis = read(parent / 'genesis.json'); latest = read(parent / 'compute-ledger.json')['attempts'][-1]
    require(latest['status'] == 'running' and latest['owner'] == genesis['generation'] + '/v76-' + token
            and latest['controller_pid'] == os.getppid() and claim['action'] == action
            and claim['manifest_sha256'] == digest(work / 'manifest.json') and m['protocol'] == SPEC,
            'V7.6 worker requires its current owning lease and frozen protocol')
    folder = inside(work, claim['snapshot_relative']); verify_snapshot(folder, claim['source'])
    require(ROOT == folder / 'source', 'Execute only frozen V7.6 source')
    require(genesis['runtime'] == {k: importlib.metadata.version(k) for k in genesis['runtime']}
            and genesis['python'] == sys.version, 'Frozen numerical runtime differs')
    require(digest(parent / 'data/metadata.json') == m['data_sha256'], 'Data metadata changed')
    if action != 'preflight':
        require(read(work / 'preflight.json')['source_sha256'] == claim['source']['sha256'], 'Scientific source changed after preflight')
    engine._start_watchdog(latest['controller_pid'])
    import v76_workflow as workflow
    files = workflow.worker(root, work, claim)
    write_new(work / 'attempts' / (token + '.validated.json'), {'manifest_sha256': digest(work / 'manifest.json'),
              'source_sha256': claim['source']['sha256'], 'claim_sha256': digest(work / 'attempts' / (token + '.claim.json')), 'files': files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'prepare-inner', 'select', 'prepare-outer', 'freeze', 'develop',
        'screen', 'stress', 'finalize', 'verify', 'status', 'replay', 'run', 'feature', 'combiner', 'view',
        'innerfit', 'selection', 'outerfit', 'reference', 'score', 'summarize', 'export', 'verify-release'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--token', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.root.resolve(); work = root / WORK
    if args.worker:
        require(args.token, 'Worker claim required'); worker(root, args.action, args.token); return
    require(args.action not in ('feature', 'combiner', 'view', 'innerfit', 'selection', 'outerfit', 'reference', 'score',
                               'summarize', 'export', 'verify-release'), 'Internal workers require a controller lease')
    if args.action == 'status':
        ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
        print(json.dumps({'used_seconds': ledger['used_seconds'], 'remaining_seconds': 43200 - ledger['used_seconds'],
             'attempts': len(ledger['attempts']), 'running': [a['owner'] for a in ledger['attempts'] if a['status'] == 'running'],
             'completed_tasks': len(list((work / 'tasks').glob('*.json'))),
             'selection': {k: read(work / 'selection.json')[k] for k in ('passed', 'penalty')} if (work / 'selection.json').exists() else None})); return
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import v76_workflow as workflow
        workflow.controller(root, work, args.action)


if __name__ == '__main__': main()
