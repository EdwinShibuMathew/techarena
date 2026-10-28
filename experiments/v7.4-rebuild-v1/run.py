"""Budgeted from-raw campaign controller. Rebuilt and historical assets never alias."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'legacy'))
from ml_workflow.contracts import canonical, digest, inside, require
from ml_workflow.storage import read, write_new, snapshot, verify_snapshot
from ml_workflow.budget import Budget
from ml_workflow import engine
from consolidate import write_json
from recipe import RECIPE, MEMBERS, CUTOFFS, FULL, TOTAL_SECONDS

RELATIVE = 'experiments/v7.4-rebuild-v1'


class ReconstructionBudget(Budget):
    def __enter__(self):
        super().__enter__()
        try:
            # Exclusive ownership proves any previous reservation has lost its
            # lock. Keep its entire charge; never reconstruct an elapsed refund.
            changed = False
            for attempt in self.state['attempts'][:-1]:
                if attempt['status'] == 'running':
                    attempt.update(status='abandoned_reserved', seconds=attempt['reserved_seconds'],
                                   error='Prior ownership ended without final accounting; full reservation retained')
                    changed = True
            if changed: write_json(self.path, self.state)
            return self
        except BaseException:
            self.lock.close()
            raise


def workspace(root):
    return inside(root, 'artifacts/rebuild')


def initialize(root):
    work = workspace(root); genesis_path = work / 'genesis.json'
    if genesis_path.exists():
        genesis = read(genesis_path)
        require(genesis['recipe'] == RECIPE and genesis['total_seconds'] == TOTAL_SECONDS,
                'Preserve the approved reconstruction generation and budget')
        return work, genesis
    require(not (work / 'compute-ledger.json').exists(), 'Do not replace existing reconstruction accounting')
    template = read(root / 'experiments/v7.4-history-graph-v1/experiment.template.json')
    raw = {v['path']: v['sha256'] for v in template['inputs'].values() if v['path'].startswith('data/raw/')}
    require(len(raw) == 3, 'Declare all raw inputs')
    genesis = {'schema_version': 5, 'generation': 'v74-from-raw-' + str(time.time_ns()),
               'recipe': RECIPE, 'recipe_sha256': canonical(RECIPE), 'total_seconds': TOTAL_SECONDS,
               'authorization': 'User approved scratch reconstruction, rebuilt comparators and a new 12-hour budget',
               'historical_accounting': {'original_ledger_lost': True,
                  'last_reported_seconds_unverified': 4459.699446460001, 'restored': False},
               'raw_sha256': raw,
               'runtime': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch', 'catboost', 'scikit-learn')},
               'python': sys.version, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_new(genesis_path, genesis)
    return work, genesis


def freeze(root, work):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    folder = work / 'snapshots' / commit
    if not folder.exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=folder.parent) as temporary:
            stage = Path(temporary) / 'bundle'; stage.mkdir()
            record = snapshot(root, stage)
            write_new(stage / 'identity.json', record)
            stage.rename(folder)
    record = read(folder / 'identity.json'); verify_snapshot(folder, record)
    return folder, record


def completed(work, task):
    marker = work / 'tasks' / (task + '.success.json')
    if not marker.exists(): return False
    record = read(marker)
    require(record['genesis_sha256'] == digest(work / 'genesis.json'), 'Completed task belongs to another generation')
    require(record['validation_sha256'] == digest(work / 'attempts' / (record['token'] + '.validated.json')), 'Task validation changed')
    validation = read(work / 'attempts' / (record['token'] + '.validated.json'))
    for relative, value in validation['files'].items(): require(digest(inside(work, relative)) == value, 'Completed artifact changed: ' + relative)
    return True


def recover_validation(work, claim, state):
    """Adopt only a fully validated identical task; partial outputs never qualify."""
    for prior in reversed(state['attempts'][:-1]):
        token = prior['owner'].split('/', 1)[-1]
        path = work / 'attempts' / (token + '.validated.json')
        if not path.exists(): continue
        old_claim = read(work / 'attempts' / (token + '.claim.json'))
        if any(old_claim[k] != claim[k] for k in ('task', 'action', 'arguments', 'source', 'genesis_sha256')): continue
        validation = read(path)
        require(validation['genesis_sha256'] == claim['genesis_sha256']
                and validation['source_sha256'] == claim['source']['sha256'], 'Recovery identity changed')
        for relative, value in validation['files'].items(): require(digest(inside(work, relative)) == value, 'Recovery artifact changed')
        return {**validation, 'technical_recovery_from': token}
    return None


def launch(root, action, task, limit, arguments=None):
    work, genesis = initialize(root)
    if completed(work, task):
        print('completed', task, flush=True); return
    token = task + '-' + str(time.time_ns())
    ledger = work / 'compute-ledger.json'
    with ReconstructionBudget(ledger, genesis['generation'] + '/' + token, limit, TOTAL_SECONDS,
                carry_identity=digest(work / 'genesis.json')) as budget:
        lock = work / ('baseline-source.json' if action == 'train' else 'scientific-source.json')
        if action != 'prepare' and lock.exists():
            locked = read(lock); folder = work / locked['snapshot_relative']; source = locked['source']
            verify_snapshot(folder, source)
        else:
            folder, source = freeze(root, work)
            if action != 'prepare':
                write_new(lock, {'snapshot_relative': folder.relative_to(work).as_posix(), 'source': source})
        claim = {'schema_version': 5, 'task': task, 'action': action, 'token': token,
                 'arguments': arguments or {}, 'source': source, 'snapshot_relative': folder.relative_to(work).as_posix(),
                 'genesis_sha256': digest(work / 'genesis.json')}
        write_new(work / 'attempts' / (token + '.claim.json'), claim)
        recovered = recover_validation(work, claim, budget.state)
        if recovered is not None:
            write_new(work / 'attempts' / (token + '.validated.json'), recovered)
        else:
            command = [sys.executable, str(folder / 'source' / RELATIVE / 'run.py'), action,
                       '--root', str(root), '--worker', '--token', token]
            require(budget.run(command, folder / 'source') == 0, 'Reconstruction worker failed; preserve reservation/checkpoint evidence')
        validation = read(work / 'attempts' / (token + '.validated.json'))
        require(validation['genesis_sha256'] == claim['genesis_sha256'] and validation['source_sha256'] == source['sha256'], 'Worker validation identity differs')
        for relative, value in validation['files'].items(): require(digest(inside(work, relative)) == value, 'Worker artifact changed')
    # Ready task markers are controller-owned and follow durable accounting.
    write_new(work / 'tasks' / (task + '.success.json'), {
        'schema_version': 5, 'token': token, 'genesis_sha256': digest(work / 'genesis.json'),
        'validation_sha256': digest(work / 'attempts' / (token + '.validated.json')),
        'compute': budget.state})


def worker(root, action, token):
    work = workspace(root); genesis = read(work / 'genesis.json')
    claim = read(work / 'attempts' / (token + '.claim.json'))
    state = read(work / 'compute-ledger.json'); attempt = state['attempts'][-1]
    require(attempt['owner'] == genesis['generation'] + '/' + token and attempt['status'] == 'running'
            and attempt['controller_pid'] == os.getppid() and claim['action'] == action
            and claim['genesis_sha256'] == digest(work / 'genesis.json'), 'Worker requires an active generation-owning controller')
    verify_snapshot(work / claim['snapshot_relative'], claim['source'])
    require(ROOT == work / claim['snapshot_relative'] / 'source', 'Execute the frozen reconstruction source')
    require(genesis['runtime'] == {n: importlib.metadata.version(n) for n in genesis['runtime']}
            and genesis['python'] == sys.version, 'Reconstruction runtime changed')
    engine._start_watchdog(attempt['controller_pid'])
    for relative, value in genesis['raw_sha256'].items():
        require(digest(root / relative) == value, 'Declared raw archive changed')
    if action == 'prepare':
        import data
        if not (work / 'data').exists(): data.prepare(root, work / 'data')
        metadata = data.verify(work / 'data')
        require(metadata['n_users'] == 54456 and metadata['n_items'] == 50000, 'Official vocabulary differs')
        arrays = data.load(work / 'data', 'train')
        positive = data.prefix_examples(arrays, FULL, metadata['n_users'], metadata['n_items'])
        require(len(positive['users']) == 1009707, 'Reconstructed click multiset differs from the documented recipe')
        files = {'data/metadata.json': digest(work / 'data/metadata.json')}
    elif action == 'train':
        import train
        cutoff, member = claim['arguments']['cutoff'], claim['arguments']['member']
        folder = work / 'baselines' / str(cutoff) / member
        identity = canonical({p: value for p, value in claim['source']['files'].items()
                              if p.startswith(RELATIVE + '/') and Path(p).name in ('train.py', 'data.py', 'recipe.py')
                              or p.startswith('legacy/')})
        train.fit(folder, work / 'data', cutoff, member, identity)
        files = {p.relative_to(work).as_posix(): digest(p) for p in
                 (folder / 'event.pt', folder / 'preference.pt', folder / 'positive.npz')}
    else:
        import study
        files = study.worker(root, work, claim)
    write_new(work / 'attempts' / (token + '.validated.json'), {
        'schema_version': 5, 'genesis_sha256': digest(work / 'genesis.json'),
        'source_sha256': claim['source']['sha256'], 'files': files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'baselines', 'train', 'develop', 'screen', 'stress', 'finalize', 'verify', 'replay', 'status',
                                         'feature', 'combiner', 'crossfit', 'score', 'summarize', 'export'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--token', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.root.resolve()
    try:
        if args.worker: worker(root, args.action, args.token); return 0
        work = workspace(root)
        if args.action == 'status':
            if (work / 'compute-ledger.json').exists():
                ledger = read(work / 'compute-ledger.json')
                print(json.dumps({'used_seconds_including_reservations': ledger['used_seconds'],
                                  'total_seconds': ledger['total_seconds'], 'latest': ledger['attempts'][-1]}, indent=2))
                for p in sorted((work / 'baselines').glob('*/*/progress.json')):
                    print(json.dumps(read(p), sort_keys=True))
            else: print(json.dumps({'started': False}))
        elif args.action == 'prepare': launch(root, 'prepare', 'prepare', 1800)
        elif args.action == 'baselines':
            require(completed(work, 'prepare'), 'Prepare and verify raw data first')
            for cutoff in CUTOFFS:
                for member in MEMBERS:
                    launch(root, 'train', 'train-' + str(cutoff) + '-' + member, 3600, {'cutoff': cutoff, 'member': member})
        elif args.action in ('develop', 'screen', 'stress', 'finalize', 'replay'):
            import study
            study.controller(root, args.action, launch)
        elif args.action == 'verify':
            import study
            print(json.dumps(study.verify(root, work), indent=2))
        else: parser.error('Direct training requires the controller')
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__': raise SystemExit(main())
