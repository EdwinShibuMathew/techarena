"""Accounted V7.5 controller; baseline delivery precedes optional promotion."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'legacy'), str(ROOT / 'experiments/v7.4-rebuild-v1')]
from ml_workflow.contracts import canonical, digest, inside, require
from ml_workflow.storage import read, write_new, verify_snapshot
from ml_workflow import engine
import run as rebuild
import data as dataset
from protocol import PROTOCOL, MEMBERS, FULL, EXTRA_CUTOFFS

RELATIVE = 'experiments/v7.5-history-age-v1'


def workdir(root): return inside(root, 'artifacts/v7_5')


def initialize(root):
    work = workdir(root); parent = root / 'artifacts/rebuild'
    if (work / 'manifest.json').exists():
        m = read(work / 'manifest.json')
        require(m['protocol'] == PROTOCOL and m['parent_genesis_sha256'] == digest(parent / 'genesis.json'), 'Frozen V7.5 generation changed')
        require(digest(parent / 'accounting_snapshots' / (m['parent_ledger_sha256'] + '.json')) == m['parent_ledger_sha256'], 'Historical accounting seal changed')
        return work, m
    with (parent / 'compute-ledger.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = read(parent / 'compute-ledger.json')
        require(ledger['total_seconds'] == 43200 and all(a['status'] == 'complete' for a in ledger['attempts']), 'Reconcile parent accounting first')
        require(rebuild.completed(parent, 'develop') and read(parent / 'stages/develop.json')['passed'] is False, 'Preserve terminal V7.4 decision')
        seal = digest(parent / 'compute-ledger.json')
        sealed = parent / 'accounting_snapshots' / (seal + '.json')
        if not sealed.exists(): write_new(sealed, ledger)
        require(digest(sealed) == seal, 'Historical accounting serialization differs')
        dependencies = {}
        from recipe import CUTOFFS
        for cutoff in CUTOFFS:
            for member in MEMBERS:
                require(rebuild.completed(parent, f'train-{cutoff}-{member}'), 'Verify all inherited checkpoints')
                folder = parent / 'baselines' / str(cutoff) / member
                for name in ('event.pt', 'preference.pt', 'positive.npz'):
                    path = folder / name; dependencies[path.relative_to(root).as_posix()] = digest(path)
        m = {'schema_version': 6, 'protocol': PROTOCOL, 'protocol_sha256': canonical(PROTOCOL),
             'parent_genesis_sha256': digest(parent / 'genesis.json'), 'parent_ledger_sha256': seal,
             'starting_used_seconds': ledger['used_seconds'], 'inherited_models': dependencies,
             'data_sha256': digest(parent / 'data/metadata.json')}
        write_new(work / 'manifest.json', m)
    return work, m


def completed(work, task):
    path = work / 'tasks' / (task + '.json')
    if not path.exists(): return False
    record = read(path)
    require(record['manifest_sha256'] == digest(work / 'manifest.json'), 'Task generation changed')
    validation = work / 'attempts' / (record['token'] + '.validated.json')
    require(record['validation_sha256'] == digest(validation), 'Task evidence changed')
    for name, expected in read(validation)['files'].items():
        require(digest(inside(work, name)) == expected, 'Task output changed: ' + name)
    return True


def launch(root, action, task, limit=1200, arguments=None, research=False):
    work, m = initialize(root)
    if completed(work, task): return
    parent = root / 'artifacts/rebuild'
    ledger_path = parent / 'compute-ledger.json'
    remaining = 43200 - read(ledger_path)['used_seconds']
    if research: require(remaining - limit >= PROTOCOL['finalization_reserve_seconds'], 'Research stopped to protect finalization reserve')
    token = task + '-' + str(time.time_ns()); args = arguments or {}
    genesis = read(parent / 'genesis.json')
    with rebuild.ReconstructionBudget(ledger_path, genesis['generation'] + '/v75-' + token, limit, 43200,
                                      carry_identity=digest(parent / 'genesis.json')) as budget:
        source_lock = work / ('baseline-source.json' if action.startswith('baseline') or action == 'train' else 'scientific-source.json')
        if source_lock.exists():
            locked = read(source_lock); folder = work / locked['snapshot_relative']; source = locked['source']
            verify_snapshot(folder, source)
        else:
            folder, source = rebuild.freeze(root, work)
            write_new(source_lock, {'source': source, 'snapshot_relative': folder.relative_to(work).as_posix()})
        claim = {'task': task, 'action': action, 'arguments': args, 'token': token, 'source': source,
                 'snapshot_relative': folder.relative_to(work).as_posix(), 'manifest_sha256': digest(work / 'manifest.json')}
        write_new(work / 'attempts' / (token + '.claim.json'), claim)
        recovered = None
        for prior in reversed(budget.state['attempts'][:-1]):
            owner = prior['owner'].split('/', 1)[-1]
            if not owner.startswith('v75-'): continue
            old_token = owner[4:]; old_path = work / 'attempts' / (old_token + '.validated.json')
            if not old_path.exists(): continue
            old_claim = read(work / 'attempts' / (old_token + '.claim.json'))
            if all(old_claim[k] == claim[k] for k in ('task', 'action', 'arguments', 'source', 'manifest_sha256')):
                recovered = read(old_path)
                for name, value in recovered['files'].items(): require(digest(inside(work, name)) == value, 'Recovery output changed')
                recovered = {**recovered, 'technical_recovery_from': old_token}; break
        validation = work / 'attempts' / (token + '.validated.json')
        if recovered: write_new(validation, recovered)
        else:
            command = [sys.executable, str(folder / 'source' / RELATIVE / 'campaign.py'), action,
                       '--root', str(root), '--worker', '--token', token]
            require(budget.run(command, folder / 'source') == 0, 'Worker failed; preserve charges and technical recovery evidence')
        record = read(validation)
        require(record['manifest_sha256'] == claim['manifest_sha256'] and record['source_sha256'] == source['sha256'], 'Worker source changed')
        for name, value in record['files'].items(): require(digest(inside(work, name)) == value, 'Worker output changed')
    write_new(work / 'tasks' / (task + '.json'), {'token': token, 'manifest_sha256': claim['manifest_sha256'],
              'validation_sha256': digest(validation), 'accounted': True})


def worker(root, action, token):
    work = workdir(root); parent = root / 'artifacts/rebuild'
    m = read(work / 'manifest.json'); claim = read(work / 'attempts' / (token + '.claim.json'))
    latest = read(parent / 'compute-ledger.json')['attempts'][-1]; genesis = read(parent / 'genesis.json')
    require(latest['owner'] == genesis['generation'] + '/v75-' + token and latest['status'] == 'running'
            and latest['controller_pid'] == os.getppid() and claim['action'] == action
            and claim['manifest_sha256'] == digest(work / 'manifest.json') and m['protocol'] == PROTOCOL,
            'Worker requires the current owning controller and frozen protocol')
    require(genesis['runtime'] == {n: importlib.metadata.version(n) for n in genesis['runtime']}
            and genesis['python'] == sys.version, 'Frozen runtime changed')
    verify_snapshot(work / claim['snapshot_relative'], claim['source'])
    require(ROOT == work / claim['snapshot_relative'] / 'source', 'Use the frozen source')
    engine._start_watchdog(latest['controller_pid'])
    require(digest(parent / 'data/metadata.json') == m['data_sha256'], 'Input metadata changed')
    for name, expected in genesis['raw_sha256'].items(): require(digest(root / name) == expected, 'Raw input changed')
    import pipeline
    outputs = pipeline.worker(root, work, claim)
    write_new(work / 'attempts' / (token + '.validated.json'), {'manifest_sha256': digest(work / 'manifest.json'),
              'source_sha256': claim['source']['sha256'], 'files': outputs})


def baseline(root):
    work, _ = initialize(root)
    for member in MEMBERS:
        launch(root, 'train', f'train-{FULL}-{member}', 900, {'cutoff': FULL, 'member': member})
        launch(root, 'baseline-feature', 'baseline-fit-feature-' + member, arguments={'member': member})
        launch(root, 'baseline-combiner', 'baseline-combiner-' + member, arguments={'member': member})
    for number in (1, 2): launch(root, 'baseline-export', f'baseline-export-{number}', arguments={'number': number})
    import pipeline
    pipeline.publish_baseline(root, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['baseline', 'prepare', 'develop', 'screen', 'stress', 'finalize', 'replay', 'replay-baseline', 'verify', 'status',
                                          'train', 'baseline-feature', 'baseline-combiner', 'baseline-export',
                                          'feature', 'combiner', 'score', 'summarize', 'export'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--token', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.root.resolve()
    try:
        if args.worker: worker(root, args.action, args.token); return 0
        work = workdir(root); work.mkdir(parents=True, exist_ok=True)
        if args.action == 'status':
            ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
            print(json.dumps({'used_seconds_including_reservation': ledger['used_seconds'], 'latest': ledger['attempts'][-1],
                              'completed_tasks': len(list((work / 'tasks').glob('*.json')))}, indent=2))
            return 0
        with (work / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'baseline': baseline(root)
            elif args.action in ('prepare', 'develop', 'screen', 'stress', 'finalize', 'replay', 'replay-baseline', 'verify'):
                import pipeline
                pipeline.controller(root, work, args.action, launch)
            else: raise ValueError('Internal action requires an owning controller')
        return 0
    except (ValueError, OSError) as error: parser.exit(1, str(error) + '\n')


if __name__ == '__main__': raise SystemExit(main())
