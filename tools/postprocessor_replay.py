"""Maintained replay lifecycle around immutable postprocessor inference code.

Only standard-library imports belong at module scope: bridge subprocesses must
import ml_workflow from the verified scientific snapshot, not today's runtime.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BRIDGE = Path(__file__).resolve()
BOOTSTRAP = (
    'import runpy,sys; source=sys.argv.pop(1); script=sys.argv.pop(1); '
    'sys.path.insert(0,source); runpy.run_path(script,run_name="__main__")'
)


def _context(root, release):
    """Called only in a fresh interpreter with the frozen tools directory first."""
    from ml_workflow import engine
    from ml_workflow.contracts import digest, identifier, inside, require
    from ml_workflow.policy import receipt_ready, release_flags
    from ml_workflow.storage import read

    identifier(release)
    config_path = inside(root, 'configs/' + release + '.json')
    config = read(config_path)
    work = engine.locate(root, config['run_id'])
    require(engine.ROOT.resolve() == (work / 'source').resolve(), 'Bridge requires the frozen runtime')
    m = engine.load(root, work)
    require(m['spec'].get('release_name') == release and m['spec']['schema_version'] in (2, 3, 4)
            and m['spec']['purpose'] == 'research', 'Wrong release recipe')
    require(receipt_ready(m['spec'], config), 'Release is not ready for its declared policy')
    verified = engine.verify(root, m['run_id'])
    require(verified['upload_ready'] and verified['csv_sha256'] == config['csv_sha256'],
            'Release evidence is incomplete')
    screen = engine.checked(work, 'screen')
    calibrator = inside(work, screen['results']['secondary']['models']['calibrator'])
    protected = [inside(root, v['path']) for v in m['spec']['inputs'].values()]
    protected += [calibrator, config_path, work / 'candidate.csv', work / 'manifest.json']
    for stage in ('develop', 'screen', 'stress', 'finalize'):
        protected += [inside(work, p) for p in read(work / (stage + '.json'))['files']]
    record = {
        'run_id': m['run_id'], 'release_name': release,
        'manifest_sha256': digest(work / 'manifest.json'), 'config_sha256': digest(config_path),
        'source_snapshot_sha256': m['source']['sha256'], 'source_commit': m['source']['commit'],
        'rows': verified['rows'], 'csv_sha256': config['csv_sha256'],
        'flags': release_flags(m['spec']), 'carried_seconds': m['carried_seconds'],
        'carry_identity': m['carry_identity'],
        'protected_paths': [p.relative_to(root).as_posix() for p in protected],
    }
    if m['spec']['schema_version']==4:
        record['context_schema']=4
    return work, m, calibrator, record


def _command(source, root, release, *args):
    return [sys.executable, str(BRIDGE), '--bootstrap', '--source', str(source),
            '--root', str(root), '--release', release, *map(str, args)]


def _checked_snapshot(root, release):
    """Use maintained validation before a child imports any snapshot modules."""
    from ml_workflow import engine
    from ml_workflow.contracts import digest, identifier, inside, require
    from ml_workflow.storage import read, verify_snapshot

    identifier(release)
    config = read(inside(root, 'configs/' + release + '.json'))
    work = engine.locate(root, config['run_id'])
    m = read(work / 'manifest.json')
    require(read(work / 'prepared.json')['manifest_sha256'] == digest(work / 'manifest.json'),
            'Frozen run manifest changed')
    verify_snapshot(work, m['source'])
    return work


def _inspect(root, work, release):
    from ml_workflow.contracts import require

    result = subprocess.run(_command(work / 'source', root, release, '--inspect'),
                            cwd=work / 'source', capture_output=True, text=True, timeout=120)
    require(result.returncode == 0, 'Frozen release verification failed: ' + result.stderr.strip())
    return json.loads(result.stdout)


def _validate_output(output, root, record):
    from ml_workflow.contracts import digest, inside, require
    from predict_v7 import validate_output_path

    output = validate_output_path(output, root, [inside(root, p) for p in record['protected_paths']])
    if output.exists():
        require(output.is_file() and digest(output) == record['csv_sha256'], 'Preserve different existing output')
    return output


def _validation(record, token, stage, root, bridge_sha):
    result = {
        'schema_version': 1, 'token': token,
        **{k: record[k] for k in ('run_id', 'release_name', 'manifest_sha256', 'config_sha256',
                                 'source_snapshot_sha256', 'source_commit', 'rows', 'csv_sha256')},
        'stage_relative': stage.relative_to(root).as_posix(), 'replay_code_sha256': bridge_sha,
    }
    if record.get('context_schema')==4:
        from ml_workflow import engine
        from ml_workflow.contracts import digest
        work=engine.locate(root,record['run_id'])
        path=work/'graphs'/('final-'+token+'.npz')
        result['graph_files']={p.relative_to(root).as_posix():digest(p) for p in (path,path.with_suffix('.json'))}
    return result


def staged_worker(root, release, token, stage):
    from ml_workflow import engine, postprocessing
    from ml_workflow.contracts import digest, identifier, inside, require
    from ml_workflow.storage import read, write_new

    identifier(token)
    work, m, calibrator, record = _context(root, release)
    engine._start_watchdog(engine._worker_lease(root, work, m, token))
    stage = inside(root, Path(stage).relative_to(root))
    claim = read(work / (token + '.attempt.json'))
    require(claim['stage_relative'] == stage.relative_to(root).as_posix()
            and claim['replay_code_sha256'] == digest(BRIDGE)
            and claim['release_identity'] == record, 'Replay claim changed')
    require(stage.is_dir() and not any(stage.iterdir()), 'Replay stage must be empty')
    # Old snapshot replay_worker() is intentionally never invoked: it publishes.
    sys.path.insert(0, str(work / 'source/legacy'))
    ids, probabilities = postprocessing.final_probabilities(
        root, work, m, engine.adapter(work, m), calibrator, token)
    from submission_export import export_submission, validate_csv
    from row_auc_final_utils import verify_csv_probabilities

    candidate = stage / 'candidate.csv'
    example = inside(root, m['spec']['inputs']['example']['path'])
    require(len(ids) == record['rows'], 'Replay row count differs')
    export_submission(ids, probabilities, example, candidate, record['rows'])
    validate_csv(candidate, example, record['rows'])
    verify_csv_probabilities(candidate, ids, probabilities)
    require(digest(candidate) == record['csv_sha256'], 'Release replay fingerprint differs')
    require(_context(root, release)[3] == record, 'Release changed during replay')
    write_new(stage / 'validated.json', _validation(record, token, stage, root, digest(BRIDGE)))
    return 0


def replay_release(root, output, release_name='v7_1'):
    from ml_workflow import engine
    from ml_workflow.budget import Budget
    from ml_workflow.contracts import digest, identifier, inside, require
    from ml_workflow.storage import read, write_new

    root = Path(root).resolve()
    identifier(release_name)
    config = read(inside(root, 'configs/' + release_name + '.json'))
    work = engine.locate(root, config['run_id'])
    record = _inspect(root, work, release_name)
    output = _validate_output(output, root, record)
    token = 'replay-' + str(time.time_ns())
    bridge_sha = digest(BRIDGE)
    ledger = inside(root, 'artifacts/experiments/compute-ledger.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + token + '-', dir=output.parent) as temporary:
        stage = Path(temporary)
        with Budget(ledger, record['run_id'] + '/' + token, 1200, 14400,
                    record['carried_seconds'], record['carry_identity']) as budget:
            write_new(work / (token + '.attempt.json'), {
                'spec_sha256': read(work / 'manifest.json')['spec_sha256'], 'state': 'claimed',
                'stage_relative': stage.relative_to(root).as_posix(),
                'replay_code_sha256': bridge_sha, 'release_identity': record,
            })
            command = _command(work / 'source', root, release_name,
                               '--worker', '--token', token, '--stage-dir', stage)
            require(budget.run(command, work / 'source') == 0, 'Independent release replay failed')
            validation = _validation(record, token, stage, root, bridge_sha)
            require(read(stage / 'validated.json') == validation, 'Replay validation record differs')
            require(digest(stage / 'candidate.csv') == record['csv_sha256'], 'Staged replay changed')
        # Nothing below runs if the durable final accounting write fails.
        require(digest(BRIDGE) == bridge_sha and _inspect(root, work, release_name) == record,
                'Release or replay implementation changed')
        require(read(stage / 'validated.json') == validation
                and digest(stage / 'candidate.csv') == record['csv_sha256'], 'Staged replay changed')
        validation_path = work / (token + '.validated.json')
        write_new(validation_path, validation)
        ledger_path = work / (token + '.ledger.json')
        write_new(ledger_path, {
            'schema_version': 2, 'exit_code': 0, 'compute': deepcopy(budget.state),
            'validation_sha256': digest(validation_path), 'csv_sha256': record['csv_sha256'],
            'replay_code_sha256': bridge_sha,
        })
        output = _validate_output(output, root, record)
        if not output.exists():
            try:
                os.link(stage / 'candidate.csv', output)
            except FileExistsError:
                # Another publisher may have won; it must satisfy the same rules.
                _validate_output(output, root, record)
        _validate_output(output, root, record)
        result = {
            'schema_version': 2, **record['flags'], 'rows': record['rows'],
            'csv_sha256': record['csv_sha256'], 'output_relative': output.relative_to(root).as_posix(),
            'run_id': record['run_id'], 'token': token, 'manifest_sha256': record['manifest_sha256'],
            'source_commit': record['source_commit'], 'source_snapshot_sha256': record['source_snapshot_sha256'],
            'replay_code_sha256': bridge_sha, 'ledger_sha256': digest(ledger_path),
            'validation_sha256': digest(validation_path),
        }
        write_new(work / (token + '.result.json'), result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--token')
    parser.add_argument('--stage-dir', type=Path)
    parser.add_argument('--bootstrap', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--source', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.bootstrap:
            from ml_workflow.contracts import require
            work = _checked_snapshot(root, args.release)
            require(args.source == work / 'source', 'Wrong scientific snapshot')
            flags = ['--root', str(root), '--release', args.release]
            if args.inspect: flags.append('--inspect')
            if args.worker: flags.append('--worker')
            if args.token: flags += ['--token', args.token]
            if args.stage_dir: flags += ['--stage-dir', str(args.stage_dir)]
            # exec keeps the Budget-owned PID/parent and discards all maintained
            # ml_workflow imports before the scientific runtime is imported.
            os.execv(sys.executable, [sys.executable, '-c', BOOTSTRAP,
                                     str(work / 'source/tools'), str(BRIDGE), *flags])
        if args.inspect and not args.worker:
            print(json.dumps(_context(root, args.release)[3], sort_keys=True))
            return 0
        if args.worker and not args.inspect and args.token and args.stage_dir:
            return staged_worker(root, args.release, args.token, args.stage_dir)
        parser.error('Use a supported replay CLI; bridge workers require an active controller lease')
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
