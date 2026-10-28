"""V7.7 controller: immutable attempts, settled ownership and bounded compute."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELATIVE = 'experiments/v7.7-reliability-slate-v1'
WORK = 'artifacts/v7_7'
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'legacy'), str(ROOT / 'experiments/v7.4-rebuild-v1')]
from ml_workflow.contracts import canonical, digest, inside, require
from ml_workflow.storage import read, write_new, snapshot, verify_snapshot
from ml_workflow.budget import Budget
from ml_workflow import engine
from consolidate import write_json
from v77_protocol import SPEC


class CampaignBudget(Budget):
    """Continue the original ledger, retaining abandoned reservations in full."""
    def __enter__(self):
        super().__enter__()
        try:
            for attempt in self.state['attempts'][:-1]:
                if attempt['status'] == 'running':
                    attempt.update(status='abandoned_reserved', seconds=attempt['reserved_seconds'],
                                   error='Lost prior ownership; full reservation retained')
            write_json(self.path, self.state)
            return self
        except BaseException:
            self.lock.close(); raise


def same(path, value):
    if path.exists(): require(read(path) == value, 'Immutable record differs: ' + str(path))
    else: write_new(path, value)


def key(action, arguments=None): return action + '-' + canonical(arguments or {})[:24]


def initialize(root, work, ledger):
    path = work / 'manifest.json'
    parent = root / 'artifacts/rebuild'
    if path.exists():
        manifest = read(path)
        require(manifest['protocol'] == SPEC and manifest['genesis_sha256'] == digest(parent / 'genesis.json'), 'Campaign generation changed')
        return manifest
    prior = root / 'artifacts/v7_6/manifest.json'
    validation = read(root / 'docs/results/v7.6-anchored-residual-v2/validation.json')
    require(digest(prior) == validation['manifest_sha256'], 'Prior asset inventory changed')
    inherited = read(prior)
    engineering = root / 'artifacts/v7_7-engineering/source-readiness.json'
    require(engineering.is_file(), 'Frozen source engineering receipt required')
    manifest = {k: inherited[k] for k in ('foundation_files', 'foundation_folders', 'protected_inputs', 'data_sha256', 'genesis_sha256')}
    manifest.update(schema_version=1, protocol=SPEC, parent_manifest_sha256=digest(prior),
                    prior_attempts=len(ledger['attempts']) - 1, prior_attempts_sha256=canonical(ledger['attempts'][:-1]),
                    engineering_receipt=engineering.relative_to(root).as_posix(), engineering_sha256=digest(engineering))
    write_new(path, manifest)
    return manifest


def validation(root, work, token, task):
    claim_path = work / 'attempts' / (token + '.claim.json')
    valid_path = work / 'attempts' / (token + '.validated.json')
    claim, valid = read(claim_path), read(valid_path)
    require(claim['task'] == task and claim['manifest_sha256'] == digest(work / 'manifest.json')
            and valid['claim_sha256'] == digest(claim_path) and valid['source_sha256'] == claim['source']['sha256']
            and claim['source'] == read(work / 'scientific-source.json')['source'], 'Task identity changed')
    ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
    require(any(a['owner'] == claim['owner'] and a['status'] == 'complete' for a in ledger['attempts']), 'Task accounting is not complete')
    for name, value in valid['files'].items():
        require(digest(inside(work, name)) == value, 'Task output changed: ' + name)
    require(valid['result'] in valid['files'], 'Missing task result')
    return claim, valid


def register(root, work, token, task):
    """Controller only: called AFTER successful ledger settlement."""
    _, valid = validation(root, work, token, task)
    marker = {'token': token, 'accounted': True, 'validation_sha256': digest(work / 'attempts' / (token + '.validated.json'))}
    same(work / 'tasks' / (task + '.json'), marker)
    return read(inside(work, valid['result']))


def completed(root, work, action, arguments=None, adopt=False):
    task = key(action, arguments); marker = work / 'tasks' / (task + '.json')
    if marker.exists():
        record = read(marker)
        require(record['accounted'] is True and record['validation_sha256'] == digest(work / 'attempts' / (record['token'] + '.validated.json')), 'Task marker changed')
        _, valid = validation(root, work, record['token'], task)
        return read(inside(work, valid['result']))
    if adopt:
        # A settled, validated attempt can be adopted with its ORIGINAL owner.
        # Failed/unsettled attempts cannot be relabeled as fresh inference.
        ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
        for attempt in reversed(ledger['attempts']):
            suffix = attempt['owner'].split('/')[-1]
            if attempt['status'] != 'complete' or not suffix.startswith('v77-'): continue
            token = suffix[4:]; cp = work / 'attempts' / (token + '.claim.json')
            if cp.exists() and read(cp)['task'] == task:
                return register(root, work, token, task)
    return None


def scientific_open(root, work):
    for name in ('selection', 'screen', 'stress'):
        result = completed(root, work, name)
        require(result is None or result['passed'], 'Scientific failure is terminal: ' + name)


def launch(root, work, action, arguments=None, phase='prepare', final=False):
    arguments = arguments or {}; task = key(action, arguments)
    prior = completed(root, work, action, arguments, adopt=True)
    if prior is not None: return prior
    scientific_open(root, work)
    require(action == 'preflight' or completed(root, work, 'preflight') is not None, 'Preflight must settle first')
    parent = root / 'artifacts/rebuild'; genesis = read(parent / 'genesis.json')
    limit = SPEC['training_job_seconds'] if action in ('teacher', 'deploy', 'head') else SPEC['other_job_seconds']
    token = phase + '-' + task + '-' + str(time.time_ns()); owner = genesis['generation'] + '/v77-' + token
    with CampaignBudget(parent / 'compute-ledger.json', owner, limit, 43200, carry_identity=digest(parent / 'genesis.json')) as budget:
        require(final or budget.state['used_seconds'] <= 43200 - 3600, 'Protect finalization reserve under lock')
        charged = sum(a.get('seconds', a['reserved_seconds']) for a in budget.state['attempts']
                      if '/v77-' + phase + '-' in a['owner'])
        require(charged <= SPEC['phase_limits'][phase], 'Aggregate phase ceiling exhausted: ' + phase)
        initialize(root, work, budget.state)
        pointer = work / 'scientific-source.json'
        if pointer.exists(): locked = read(pointer); folder = inside(work, locked['snapshot_relative']); source = locked['source']
        else:
            folder = work / 'snapshot'; folder.mkdir(parents=True)
            source = snapshot(root, folder); write_new(folder / 'identity.json', source)
            write_new(work / 'scientific-source-record.json', source)
            write_new(pointer, {'snapshot_relative': 'snapshot', 'source': source})
        verify_snapshot(folder, source)
        claim = {'task': task, 'action': action, 'arguments': arguments, 'token': token, 'owner': owner,
                 'manifest_sha256': digest(work / 'manifest.json'), 'source': source, 'phase': phase}
        write_new(work / 'attempts' / (token + '.claim.json'), claim)
        command = [sys.executable, str(folder / 'source' / RELATIVE / 'v77_runner.py'), action,
                   '--root', str(root), '--worker', '--token', token]
        require(budget.run(command, folder / 'source') == 0, 'Worker failed; preserve all charges and attempt evidence')
        valid = read(work / 'attempts' / (token + '.validated.json'))
        require(valid['claim_sha256'] == digest(work / 'attempts' / (token + '.claim.json'))
                and valid['source_sha256'] == source['sha256'], 'Worker source differs')
        for name, expected in valid['files'].items(): require(digest(inside(work, name)) == expected, 'Worker output changed')
    result = register(root, work, token, task)
    print(json.dumps({'completed': task, 'passed': result.get('passed'), 'used_seconds': read(parent / 'compute-ledger.json')['used_seconds']}), flush=True)
    return result


def worker(root, action, token):
    work = root / WORK; parent = root / 'artifacts/rebuild'
    claim = read(work / 'attempts' / (token + '.claim.json')); manifest = read(work / 'manifest.json')
    ledger = read(parent / 'compute-ledger.json'); current = ledger['attempts'][-1]; genesis = read(parent / 'genesis.json')
    require(current['status'] == 'running' and current['owner'] == claim['owner']
            and current['controller_pid'] == os.getppid() and claim['action'] == action
            and manifest['protocol'] == SPEC and claim['manifest_sha256'] == digest(work / 'manifest.json'), 'Worker requires current owned lease')
    require(ROOT == work / 'snapshot/source', 'Worker must execute frozen source')
    verify_snapshot(work / 'snapshot', claim['source'])
    require(genesis['runtime'] == {k: importlib.metadata.version(k) for k in genesis['runtime']}
            and genesis['python'] == sys.version, 'Production numerical runtime differs')
    require(digest(parent / 'data/metadata.json') == manifest['data_sha256'], 'Data identity changed')
    engine._start_watchdog(current['controller_pid'])
    import v77_workflow as workflow
    directory = work / 'jobs' / token; directory.mkdir(parents=True)
    result = workflow.worker(root, work, directory, claim)
    write_new(directory / 'result.json', result)
    # Only this attempt directory is inventoried; workers never publish markers.
    files = {p.relative_to(work).as_posix(): digest(p) for p in directory.rglob('*') if p.is_file()}
    write_new(work / 'attempts' / (token + '.validated.json'), {'claim_sha256': digest(work / 'attempts' / (token + '.claim.json')),
              'source_sha256': claim['source']['sha256'], 'result': (directory / 'result.json').relative_to(work).as_posix(), 'files': files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action'); parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS); parser.add_argument('--token', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.root.resolve(); work = root / WORK
    if args.worker: worker(root, args.action, args.token); return
    require(args.action in ('run', 'preflight', 'develop', 'screen', 'stress', 'finalize', 'replay', 'verify', 'status'), 'Unsupported controller action')
    if args.action == 'status':
        ledger = read(root / 'artifacts/rebuild/compute-ledger.json')
        print(json.dumps({'used_seconds': ledger['used_seconds'], 'remaining_seconds': 43200 - ledger['used_seconds'],
                          'attempts': len(ledger['attempts']), 'running': sum(a['status'] == 'running' for a in ledger['attempts'])})); return
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import v77_workflow as workflow
        workflow.controller(root, work, args.action)


if __name__ == '__main__': main()
