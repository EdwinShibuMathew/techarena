"""Required contracts for schema-v2 calibration and user-grouped cross-fitting."""
import copy
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from ml_workflow import engine,postprocessing
from ml_workflow.contracts import ExperimentSpec,MEMBERS
from ml_workflow.storage import read, write_new
from ml_workflow.contracts import digest, canonical
from ml_workflow import budget as budget_module, policy
from ml_workflow.budget import Budget
import postprocessor_replay as replay
spec=importlib.util.spec_from_file_location('entropy_adapter',ROOT/'experiments/v7.1-entropy-sharpening-v1/adapter.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
ML=all(importlib.util.find_spec(x) is not None for x in ('torch','catboost','scipy'))


class ReplayPublicationTests(unittest.TestCase):
    """Real accounting and filesystem behavior; numerical work is substituted."""

    def fixture(self, root, release='v7_3_trial'):
        work = root / 'artifacts/experiments/run'
        (work / 'source').mkdir(parents=True)
        write_new(work / 'manifest.json', {'spec_sha256': 'f' * 64})
        write_new(root / ('configs/' + release + '.json'), {'run_id': 'run'})
        (work / 'candidate.csv').write_bytes(b'verified synthetic CSV')
        published = root / ('artifacts/submissions/' + release + '.csv')
        published.parent.mkdir(parents=True)
        os.link(work / 'candidate.csv', published)
        record = {
            'run_id': 'run', 'release_name': release, 'manifest_sha256': digest(work / 'manifest.json'),
            'config_sha256': digest(root / ('configs/' + release + '.json')),
            'source_snapshot_sha256': 'a' * 64, 'source_commit': 'synthetic-frozen-source',
            'rows': 20, 'csv_sha256': digest(published), 'carried_seconds': 0., 'carry_identity': None,
            'protected_paths': [(work / 'candidate.csv').relative_to(root).as_posix()],
            'flags': policy.release_flags({'release_policy': 'v7_upload_trial'} if release != 'v7_1' else {}),
        }
        return work, published, record

    def launch(self, root, record, command, cwd):
        token = command[command.index('--token') + 1]
        stage = Path(command[command.index('--stage-dir') + 1])
        (stage / 'candidate.csv').write_bytes(b'verified synthetic CSV')
        write_new(stage / 'validated.json', replay._validation(record, token, stage, root, digest(replay.BRIDGE)))
        self.assertFalse(list(cwd.parent.glob('replay-*.result.json')))
        return 0

    def mocked(self, stack, root, record, run=None):
        stack.enter_context(patch.object(replay, '_inspect', return_value=record))
        return stack.enter_context(patch.object(Budget, 'run', side_effect=run or (
            lambda command, cwd: self.launch(root, record, command, cwd))))

    def test_default_clis_after_publication_use_separate_replay_files(self):
        for release, script in (('v7_3_trial', 'predict_postprocessor.py'), ('v7_1', 'predict_v7_1.py')):
            with self.subTest(release=release), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp); work, published, record = self.fixture(root, release)
                self.mocked(stack, root, record)
                spec = importlib.util.spec_from_file_location('replay_cli', ROOT / 'tools' / script)
                cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
                args = [script, '--root', str(root)]
                if release != 'v7_1': args += ['--release', release]
                stack.enter_context(patch.object(sys, 'argv', args))
                stack.enter_context(patch('builtins.print'))
                cli.main()
                output = root / ('artifacts/replays/' + release + '.csv')
                self.assertEqual(output.read_bytes(), published.read_bytes())
                self.assertFalse(os.path.samefile(output, published))
                result = read(next(work.glob('replay-*.result.json')))
                for key, value in record['flags'].items(): self.assertEqual(result[key], value)
                self.assertEqual(result['ledger_sha256'], digest(next(work.glob('replay-*.ledger.json'))))
                self.assertEqual(result['validation_sha256'], digest(next(work.glob('replay-*.validated.json'))))

    def test_protected_hardlink_and_symlink_aliases_fail_before_budget(self):
        for alias in ('hardlink', 'symlink'):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp); work, published, record = self.fixture(root)
                output = root / ('artifacts/' + alias + '.csv')
                if alias == 'hardlink': os.link(published, output)
                else: output.symlink_to(published)
                run = self.mocked(stack, root, record)
                with self.assertRaisesRegex(ValueError, 'aliases a protected input'):
                    replay.replay_release(root, output, 'v7_3_trial')
                run.assert_not_called()
                self.assertFalse((root / 'artifacts/experiments/compute-ledger.json').exists())
                self.assertEqual(published.read_bytes(), b'verified synthetic CSV')

    def test_accounting_failure_cannot_publish_and_preserves_existing_output(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp); work, published, record = self.fixture(root)
                output = root / 'artifacts/replay.csv'
                if existing: output.write_bytes(b'verified synthetic CSV')
                self.mocked(stack, root, record)
                original = budget_module.write_json
                def fail(path, value):
                    if value['attempts'][-1]['status'] == 'complete': raise OSError('injected accounting failure')
                    original(path, value)
                stack.enter_context(patch.object(budget_module, 'write_json', side_effect=fail))
                with self.assertRaisesRegex(OSError, 'accounting failure'):
                    replay.replay_release(root, output, 'v7_3_trial')
                self.assertEqual(output.exists(), existing)
                if existing: self.assertEqual(output.read_bytes(), b'verified synthetic CSV')
                self.assertFalse(list(work.glob('replay-*.result.json')))
                self.assertFalse(list(work.glob('replay-*.ledger.json')))
                ledger = read(root / 'artifacts/experiments/compute-ledger.json')
                self.assertEqual(ledger['attempts'][-1]['status'], 'running')
                self.assertEqual(ledger['used_seconds'], 1200)
                self.assertEqual(published.read_bytes(), b'verified synthetic CSV')

    def test_publication_order_and_ledger_write_failure(self):
        from ml_workflow import storage
        for failing in (False, True):
            with self.subTest(failing=failing), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp); work, _, record = self.fixture(root)
                output = root / 'artifacts/replay.csv'; events = []
                self.mocked(stack, root, record)
                actual_account = budget_module.write_json; actual_record = storage.write_new; actual_link = os.link
                def account(path, value):
                    actual_account(path, value)
                    events.append(value['attempts'][-1]['status'])
                def save(path, value):
                    if path.name.endswith('.ledger.json'):
                        self.assertEqual(events[-1], 'complete'); self.assertFalse(output.exists())
                        if failing: raise OSError('injected replay ledger failure')
                        events.append('ledger')
                    if path.name.endswith('.result.json'):
                        self.assertTrue(output.exists()); self.assertEqual(events[-1], 'csv'); events.append('ready')
                    actual_record(path, value)
                def link(source, destination, *args, **kwargs):
                    if destination == output:
                        self.assertEqual(events[-1], 'ledger'); events.append('csv')
                    return actual_link(source, destination, *args, **kwargs)
                stack.enter_context(patch.object(budget_module, 'write_json', side_effect=account))
                stack.enter_context(patch.object(storage, 'write_new', side_effect=save))
                stack.enter_context(patch.object(replay.os, 'link', side_effect=link))
                if failing:
                    with self.assertRaisesRegex(OSError, 'replay ledger failure'):
                        replay.replay_release(root, output, 'v7_3_trial')
                    self.assertFalse(output.exists()); self.assertFalse(list(work.glob('replay-*.result.json')))
                else:
                    replay.replay_release(root, output, 'v7_3_trial')
                    self.assertEqual(events, ['running', 'complete', 'ledger', 'csv', 'ready'])

    def test_worker_failures_timeout_and_staged_tampering_cannot_publish(self):
        for failure in ('exit', 'timeout', 'csv', 'record'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp); work, _, record = self.fixture(root)
                output = root / 'artifacts/replay.csv'
                def run(command, cwd):
                    if failure == 'exit': return 1
                    if failure == 'timeout': raise subprocess.TimeoutExpired(command, 1)
                    self.launch(root, record, command, cwd)
                    stage = Path(command[command.index('--stage-dir') + 1])
                    if failure == 'csv': (stage / 'candidate.csv').write_bytes(b'corrupt')
                    else: (stage / 'validated.json').write_text('{}')
                    return 0
                self.mocked(stack, root, record, run)
                with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                    replay.replay_release(root, output, 'v7_3_trial')
                self.assertFalse(output.exists()); self.assertFalse(list(work.glob('replay-*.result.json')))
                self.assertFalse(list(work.glob('replay-*.ledger.json')))

    def test_existing_outputs_races_and_result_write_failure(self):
        from ml_workflow import storage
        for failure in ('matching', 'different', 'race_matching', 'race_different', 'result'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp); work, _, record = self.fixture(root)
                output = root / 'artifacts/replay.csv'
                if failure in ('matching', 'different'):
                    output.write_bytes(b'verified synthetic CSV' if failure == 'matching' else b'preserve me')
                run = self.mocked(stack, root, record)
                original_link = os.link; original_record = storage.write_new
                def link(source, destination, *args, **kwargs):
                    if destination == output and failure.startswith('race_'):
                        output.write_bytes(b'verified synthetic CSV' if failure == 'race_matching' else b'preserve me')
                    return original_link(source, destination, *args, **kwargs)
                def save(path, value):
                    if failure == 'result' and path.name.endswith('.result.json'): raise OSError('result failure')
                    original_record(path, value)
                stack.enter_context(patch.object(replay.os, 'link', side_effect=link))
                stack.enter_context(patch.object(storage, 'write_new', side_effect=save))
                if failure in ('different', 'race_different', 'result'):
                    with self.assertRaises((ValueError, OSError)): replay.replay_release(root, output, 'v7_3_trial')
                    self.assertFalse(list(work.glob('replay-*.result.json')))
                    if failure != 'result': self.assertEqual(output.read_bytes(), b'preserve me')
                    if failure == 'different': run.assert_not_called()
                else:
                    replay.replay_release(root, output, 'v7_3_trial')
                    self.assertEqual(output.read_bytes(), b'verified synthetic CSV')


class FrozenReplayBridgeTests(unittest.TestCase):
    """Sealed synthetic science, real frozen engine/lease/CSV and subprocesses."""

    def fixture(self, root, trial=False, delay=False):
        from ml_workflow.storage import save_arrays
        from submission_export import export_submission
        release = 'v7_3_trial' if trial else 'v7_1'
        (root / 'configs').mkdir()
        experiment = 'v7.3-v7-anchor-v1' if trial else 'v7.1-entropy-sharpening-v1'
        value = read(ROOT / 'experiments' / experiment / 'experiment.json')
        work = root / 'artifacts/experiments/frozen'
        source = work / 'source'; source.mkdir(parents=True)
        paths = list((ROOT / 'tools').rglob('*.py')) + list((ROOT / 'legacy').glob('*.py'))
        paths += [ROOT / value['adapter']]
        for path in paths:
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        # Declare the numerical substitute BEFORE sealing. Runtime validation,
        # gates, snapshot verification, export and lease checks remain real.
        frozen_postprocessing = source / 'tools/ml_workflow/postprocessing.py'
        with frozen_postprocessing.open('a') as stream:
            stream.write('''
def final_probabilities(root, work, m, module, calibrator, number):
    marker = root / 'artifacts/entered'
    marker.touch()
''')
            if delay:
                stream.write("    time.sleep(2)\n    (root / 'artifacts/escaped').touch()\n")
            stream.write('''    with np.load(inside(root, m['spec']['inputs']['replay_values']['path']), allow_pickle=False) as saved:
        return saved['rowids'], saved['probabilities']

def replay_worker(*args, **kwargs):
    raise AssertionError('Historical worker publication must never be invoked')
''')
        inputs = root / 'artifacts/validation-inputs'; inputs.mkdir()
        ids = np.array(['synthetic-' + str(i) for i in range(20)])
        probabilities = np.linspace(.01, .2, 20).reshape(2, 10)
        values = inputs / 'values.npz'; save_arrays(values, rowids=ids, probabilities=probabilities)
        example = inputs / 'example.csv'
        example.write_text('rowid,predicted_probability\n' + ''.join(row + ',0\n' for row in ids))
        inventory = inputs / 'inventory.json'; write_new(inventory, {})
        carry = inputs / 'carry.json'; write_new(carry, {'total_compute_seconds': 0., 'compute_limit_seconds': 14400})
        value['device'] = 'cpu'
        assets = {'train': values, 'evaluation': values, 'replay_values': values,
                  'example': example, 'inventory': inventory, 'carry': carry}
        if trial:
            validation = inputs / 'v7-validation.json'; write_new(validation, {'synthetic': True})
            anchor = inputs / 'v7.json'
            write_new(anchor, {'submission_sha256': policy.V7_SHA, 'sharpening_gamma': 1.12,
                               'validation_report_sha256': digest(validation)})
            assets.update(v7_config=anchor, v7_validation=validation)
        value['inputs'] = {name: {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}
                           for name, path in assets.items()}
        files = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*.py')}
        archive = work / 'source.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name in files: tar.add(source / name, arcname='source/' + name)
        m = {'run_id': 'frozen', 'spec': value, 'spec_sha256': canonical(value),
             'source': {'commit': 'synthetic-scientific-snapshot', 'sha256': digest(archive), 'files': files},
             'carried_seconds': 0., 'carry_identity': digest(carry), 'validation_status': 'synthetic control-flow fixture'}
        write_new(work / 'manifest.json', m)
        write_new(work / 'prepared.json', {'manifest_sha256': digest(work / 'manifest.json')})
        write_new(work / 'prepare.ledger.json', {'exit_code': 0})
        write_new(work / 'calibrator.json', {'synthetic': True})
        metric = {'baseline_auc': .8, 'candidate_auc': .81, 'auc_diff': .01,
                  'ci_lower': .005, 'ci_upper': .015, 'resamples': 1000}
        if trial: metric['v7_comparison'] = {**metric, 'anchor_sha256': policy.V7_SHA}
        def report(stage, results, **extra):
            path = work / (stage + '.json')
            write_new(path, {'complete': True, 'passed': True, 'files': {}, 'results': results,
                             'manifest_sha256': digest(work / 'manifest.json'), **extra})
            write_new(work / (stage + '.seal.json'), {'sha256': digest(path)})
            write_new(work / (stage + '.ledger.json'), {'exit_code': 0})
        development = {candidate + '/' + case['name']: metric
                       for candidate in value['candidate_order'] for case in value['development']}
        winner = value['candidate_order'][0]
        report('develop', development, winner=winner)
        report('screen', {'primary': metric, 'secondary': {**metric, 'models': {'calibrator': 'calibrator.json'}}})
        report('stress', {'five_day': {**metric, 'late': metric}})
        export_submission(ids, probabilities, example, work / 'candidate.csv', 20)
        write_new(work / 'finalize.json', {'complete': True, 'independent_replay': True, 'rows': 20,
                  'manifest_sha256': digest(work / 'manifest.json'), 'csv_sha256': digest(work / 'candidate.csv'),
                  'files': {'candidate.csv': digest(work / 'candidate.csv')}})
        write_new(work / 'finalize.seal.json', {'sha256': digest(work / 'finalize.json')})
        write_new(work / 'finalize.ledger.json', {'exit_code': 0})
        # Actual publication creates the protected hard link. Only initial fixture
        # loading is substituted; subsequent bridge loads are unmodified.
        with patch.object(engine, 'load', return_value=m): engine.publish(root, work, m)
        return work, release, m

    def test_both_clis_replay_sealed_snapshot_despite_changed_live_runtime(self):
        for trial in (False, True):
            with self.subTest(trial=trial), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); work, release, m = self.fixture(root, trial)
                with self.assertRaisesRegex(ValueError, 'Runtime changed'): engine.load(root, work)
                source_hash = digest(work / 'source.tar.gz')
                script = 'predict_postprocessor.py' if trial else 'predict_v7_1.py'
                command = [sys.executable, str(ROOT / 'tools' / script), '--root', str(root)]
                if trial: command += ['--release', release]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                result = json.loads(completed.stdout)
                output = root / result['output_relative']
                self.assertEqual(output.read_bytes(), (work / 'candidate.csv').read_bytes())
                self.assertFalse(os.path.samefile(output, work / 'candidate.csv'))
                self.assertEqual(source_hash, digest(work / 'source.tar.gz'))
                self.assertEqual(result['source_snapshot_sha256'], m['source']['sha256'])
                for key, value in policy.release_flags(m['spec']).items(): self.assertEqual(result[key], value)
                ledger = read(next(work.glob('replay-*.ledger.json')))
                self.assertEqual(ledger['compute']['attempts'][-1]['status'], 'complete')
                internal = read(next(work.glob('replay-*.validated.json')))
                self.assertFalse({'upload_ready', 'submission_ready'} & internal.keys())

    def test_snapshot_tampering_and_unleased_worker_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work, release, _ = self.fixture(root)
            stage = root / 'artifacts/stage'; stage.mkdir()
            command = replay._command(work / 'source', root, release, '--worker', '--token', 'unleased', '--stage-dir', stage)
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(list(stage.iterdir()), [])
            with (work / 'source/tools/ml_workflow/postprocessing.py').open('a') as stream:
                stream.write('\n# tampered after sealing\n')
            with self.assertRaisesRegex(ValueError, 'Snapshot source changed'):
                replay.replay_release(root, root / 'artifacts/replay.csv', release)
            self.assertFalse((root / 'artifacts/experiments/compute-ledger.json').exists())

    def test_worker_rechecks_snapshot_before_importing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work, release, _ = self.fixture(root)
            marker = root / 'artifacts/tampered-module-executed'
            original_run = Budget.run
            def run(budget, command, cwd):
                path = work / 'source/tools/ml_workflow/engine.py'
                path.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n' + path.read_text())
                return original_run(budget, command, cwd)
            with patch.object(Budget, 'run', autospec=True, side_effect=run):
                with self.assertRaisesRegex(ValueError, 'Independent release replay failed'):
                    replay.replay_release(root, root / 'artifacts/replay.csv', release)
            self.assertFalse(marker.exists())
            self.assertFalse((root / 'artifacts/replay.csv').exists())
            self.assertFalse(list(work.glob('replay-*.result.json')))

    def test_worker_stops_on_controller_death_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work, release, _ = self.fixture(root, delay=True)
            process = subprocess.Popen([sys.executable, str(ROOT / 'tools/predict_v7_1.py'), '--root', str(root)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 30
                while not (root / 'artifacts/entered').exists():
                    self.assertIsNone(process.poll(), 'Controller exited before inference')
                    self.assertLess(time.monotonic(), deadline); time.sleep(.05)
                process.kill(); process.wait(timeout=5); time.sleep(2.2)
                self.assertFalse((root / 'artifacts/escaped').exists())
                self.assertFalse((root / ('artifacts/replays/' + release + '.csv')).exists())
                self.assertFalse(list(work.glob('replay-*.result.json')))
                state = read(root / 'artifacts/experiments/compute-ledger.json')
                self.assertEqual(state['used_seconds'], 1200)
                self.assertEqual(state['attempts'][-1]['status'], 'running')
            finally:
                if process.poll() is None: process.kill(); process.wait(timeout=5)


class AllocationContractTests(unittest.TestCase):
    def test_mass_bounds_permutation_ties_and_zero_identity(self):
        p=np.array([[0]*10,[.2]*10,[1,.5]+[0]*8,[.9,.05,.05]+[0]*7,[1e-300]*10],dtype=np.float64)
        for theta in ((0,0),(.25,.25),(-.25,-.25)):
            q=adapter.predict_calibrator(p,theta)
            np.testing.assert_allclose(q.sum(1),p.sum(1),rtol=0,atol=1e-14)
            self.assertTrue(np.isfinite(q).all() and ((q>=0)&(q<=1)).all())
            np.testing.assert_array_equal(q[p==0],0)
            for i in range(10):
                for j in range(i):np.testing.assert_array_equal(np.sign(q[:,i]-q[:,j]),np.sign(p[:,i]-p[:,j]))
            order=np.arange(9,-1,-1)
            np.testing.assert_allclose(adapter.predict_calibrator(p[:,order],theta),q[:,order],rtol=0,atol=1e-14)
        np.testing.assert_array_equal(adapter.predict_calibrator(p,(0,0)),p)
        for bad in ([1,0],[np.nan,0]):
            with self.assertRaises(ValueError):adapter.predict_calibrator(p,bad)

    def test_folds_keep_users_and_all_candidates_together(self):
        users=np.repeat(np.arange(25),3);folds=adapter.user_folds(users)
        self.assertEqual(set(folds),set(range(5)))
        for user in np.unique(users):self.assertEqual(len(np.unique(folds[users==user])),1)
        np.testing.assert_array_equal(folds,adapter.user_folds(users))

    def test_unready_release_refuses_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(postprocessing,'final_probabilities') as compute:
                with self.assertRaises(FileNotFoundError):postprocessing.replay_release(Path(tmp),Path(tmp)/'out.csv')
                compute.assert_not_called()

    def test_schema_v1_compatibility_and_v2_validation(self):
        fixture_spec=importlib.util.spec_from_file_location('fixture',ROOT/'experiments/workflow-smoke/prepare_fixture.py')
        fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            v1=read(fixture.create(Path(tmp)));ExperimentSpec.parse(v1)
            v2=copy.deepcopy(v1);v2.update(schema_version=2,mode='ensemble_postprocessor')
            v2['inputs']['inventory']={'path':'inventory.json','sha256':'f'*64}
            v2['candidates']={'fixture':{'kind':'ensemble_postprocessor','config':adapter.CONFIG}}
            ExperimentSpec.parse(v2)
            v2['mode']='neural'
            with self.assertRaises(ValueError):ExperimentSpec.parse(v2)


@unittest.skipUnless(ML,'CPU ML dependencies required; enforced by model-contracts CI')
class CalibratorModelTests(unittest.TestCase):
    def test_actual_optimization_serialized_reload_and_zero_scores(self):
        rng=np.random.default_rng(42);p=rng.uniform(.001,.1,(50,10));p[0]=0;p[1,0]=0
        y=np.zeros_like(p,dtype=np.int8);y[np.arange(1,50),p[1:].argmax(1)]=1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'calibrator.json';metadata={'cutoff':123}
            saved=adapter.fit_calibrator(p,y,metadata,path)
            self.assertEqual(saved,adapter.load_calibrator(path,metadata))
            self.assertLessEqual(saved['objective'],saved['identity_objective'])
            first=adapter.predict_calibrator(p,saved['coefficients'])
            np.testing.assert_array_equal(first,adapter.predict_calibrator(p,adapter.load_calibrator(path)['coefficients']))
            with self.assertRaises(ValueError):adapter.load_calibrator(path,{'cutoff':999})
            bad=read(path);bad['coefficients']=[1,0]
            import json
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):adapter.load_calibrator(path)

    def test_real_crossfit_and_verified_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';work.mkdir()
            n=40;raw={'u':np.repeat(np.arange(20),2),'y':np.zeros((n,10),dtype=np.int8)}
            raw['y'][::2,0]=1;fit=np.arange(n)
            inv={'parameters':{member:{'primary':{'iterations':2,'depth':2,'random_seed':42,'thread_count':1,'allow_writing_files':False}} for member in MEMBERS}}
            m={'source':{'files':{'tools/ml_workflow/postprocessing.py':'fixture'}},'spec':{'adapter':'fixture'}}
            data={'X':np.random.default_rng(42).normal(size=(n*10,2)).astype(np.float32),'feature_names':np.array(['x','y'])}
            with patch.object(postprocessing,'features',return_value=data):
                first,receipt=postprocessing.crossfit(root,work,m,adapter,inv,raw,fit,123,'primary')
                second,receipt2=postprocessing.crossfit(root,work,m,adapter,inv,raw,fit,123,'primary')
            np.testing.assert_array_equal(first,second)
            self.assertFalse(read(receipt)['hit']);self.assertTrue(read(receipt2)['hit'])
            folder=root/'artifacts/cache/ml-workflow/postprocess-v1'/read(receipt)['key']
            next(folder.glob('*.cbm')).write_bytes(b'corrupt')
            with patch.object(postprocessing,'features',return_value=data),self.assertRaises(ValueError):
                postprocessing.crossfit(root,work,m,adapter,inv,raw,fit,123,'primary')

    def test_budgeted_schema_v2_preparation_seals_inputs_and_refuses_reentry(self):
        sys.path.insert(0,str(ROOT/'tests'))
        from test_workflow_integration import WorkflowIntegrationTests
        from ml_workflow.storage import write_new
        from ml_workflow.contracts import digest
        fixture_spec=importlib.util.spec_from_file_location('fixture_v2',ROOT/'experiments/workflow-smoke/prepare_fixture.py')
        fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);helper=WorkflowIntegrationTests()
            helper.make_repository(root)
            path=fixture.create(root,'post-prepare');value=read(path)
            inv=path.parent/'inventory.json';write_new(inv,{})
            value.update(schema_version=2,mode='ensemble_postprocessor',adapter='experiments/v7.1-entropy-sharpening-v1/adapter.py')
            value['inputs']['inventory']={'path':inv.relative_to(root).as_posix(),'sha256':digest(inv)}
            value['candidates']={'fixture':{'kind':'ensemble_postprocessor','config':adapter.CONFIG}}
            path.write_text(__import__('json').dumps(value))
            helper.cli(root,'prepare','--spec',str(path),'--run-id','post-prepare')
            work=root/'artifacts/validation/post-prepare'
            self.assertEqual(read(work/'prepare.ledger.json')['exit_code'],0)
            m=engine.load(root,work)
            self.assertEqual(len(engine.events(root,m,'train')['u']),168)
            ledger=(work/'compute.json').read_bytes()
            helper.cli(root,'prepare','--spec',str(path),'--run-id','post-prepare',code=1)
            helper.cli(root,'prepare','--run-id','post-prepare','--worker',code=1)
            self.assertEqual((work/'compute.json').read_bytes(),ledger)
            with (root/m['generated_inputs']['train']['path']).open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Prepared input changed'):engine.load(root,work)

    def test_unleased_preparation_cannot_write_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):postprocessing.prepare_worker(root,'missing')
            self.assertFalse((root/'artifacts').exists())
