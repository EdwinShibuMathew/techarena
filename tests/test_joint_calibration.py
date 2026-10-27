"""Required schema-v3 fitting, eligibility, saved-state and CSV contracts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from ml_workflow import engine, postprocessing
from ml_workflow.contracts import ExperimentSpec, JAN20, DAY, MEMBERS
from ml_workflow.storage import read, save_arrays
spec = importlib.util.spec_from_file_location('joint_adapter', ROOT / 'experiments/v7.2-joint-calibration-v1/adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
ML = all(importlib.util.find_spec(x) is not None for x in ('torch', 'catboost', 'scipy'))


class JointContractTests(unittest.TestCase):
    def test_schema_contract_and_aged_selection(self):
        value = read(ROOT / 'experiments/v7.2-joint-calibration-v1/experiment.json')
        ExperimentSpec.parse(value)
        for key, bad in [('schema_version', 4), ('selection', 'maximin_nonnegative_origins'),
                         ('postprocessing_contract', {'event_mass': 'preserve'})]:
            changed = deepcopy(value);changed[key] = bad
            with self.assertRaises(ValueError):ExperimentSpec.parse(changed)
        rows = {}
        for name in value['candidate_order']:
            for case in value['development']:
                rows[name + '/' + case['name']] = {'auc_diff': .002, 'diagnostics': {'late': {'events': 5, 'auc_diff': .001}}}
        self.assertEqual(engine._winner(value, rows), 'event')
        rows['event/jan22_long']['diagnostics']['late']['auc_diff'] = -.001
        self.assertEqual(engine._winner(value, rows), 'joint')
        rows['joint/jan24_forward']['auc_diff'] = -.001
        self.assertEqual(engine._winner(value, rows), 'entropy_joint')
        rows['entropy_joint/jan22_long']['diagnostics']['late']['auc_diff'] = -.001
        self.assertIsNone(engine._winner(value, rows))
        rows['event/jan22_long']['diagnostics']['late']['events'] = 0
        with self.assertRaises(ValueError):engine._winner(value, rows)

    def test_general_release_rejects_absence_and_unsafe_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(postprocessing, 'final_probabilities') as compute:
                for name in ('v7_2', 'v7_1'):
                    with self.assertRaises(FileNotFoundError):postprocessing.replay_release(Path(tmp), Path(tmp)/'out.csv', name)
                with self.assertRaises(ValueError):postprocessing.replay_release(Path(tmp), Path(tmp)/'out.csv', '../v6')
                compute.assert_not_called()


@unittest.skipUnless(ML, 'CPU ML dependencies required; enforced by model-contracts CI')
class JointModelTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(2026)
        p = rng.uniform(.001, .15, (80, 10))
        p[0] = 0;p[1, 0] = 0;p[2, 0] = 1;p[3, 1] = p[3, 0]
        y = np.zeros_like(p, dtype=np.int8)
        y[np.arange(1, 80), p[1:].argmax(1)] = 1
        return p, y

    def test_real_fitting_reload_identity_order_and_fit_only_normalization(self):
        p, y = self.data()
        with tempfile.TemporaryDirectory() as tmp:
            for name, params in adapter.CANDIDATES.items():
                path = Path(tmp)/(name+'.json');metadata = {'candidate': name, 'cutoff': 1}
                record = adapter.fit_calibrator(p, y, deepcopy(metadata), path, deepcopy(params))
                self.assertEqual(record, adapter.load_calibrator(path, metadata, params))
                self.assertLessEqual(record['objective'], record['identity_objective'] + 1e-12)
                logmass = np.log(np.maximum(p.sum(1), 1e-12))
                self.assertEqual(record['normalization']['log_mass_mean'], float(logmass.mean()))
                out = adapter.predict_calibrator(p, record)
                np.testing.assert_array_equal(out, adapter.predict_calibrator(p, adapter.load_calibrator(path)))
                np.testing.assert_array_equal(out[p == 0], 0);np.testing.assert_array_equal(out[p == 1], 1)
                for i in range(10):
                    for j in range(i):np.testing.assert_array_equal(np.sign(out[:,i]-out[:,j]), np.sign(p[:,i]-p[:,j]))
                permutation = np.arange(9, -1, -1)
                np.testing.assert_allclose(adapter.predict_calibrator(p[:,permutation], record), out[:,permutation], atol=1e-15, rtol=0)
                identity = deepcopy(record);identity['coefficients'] = [0]*4
                np.testing.assert_array_equal(adapter.predict_calibrator(p, identity), p)
                before = deepcopy(record['normalization'])
                adapter.predict_calibrator(p * .1, record)
                self.assertEqual(record['normalization'], before)
                for field, bad in [('coefficients', [1]*4), ('normalization', {'log_mass_mean': 0, 'log_mass_scale': 0})]:
                    changed = deepcopy(record);changed[field] = bad
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):adapter.load_calibrator(path)
                with self.assertRaises(ValueError):adapter.validate_state(record, {'cutoff':999})
                with self.assertRaises(ValueError):adapter.validate_state(record, parameters={'active':[0]})

    def test_event_correction_changes_mass_and_inactive_parameters_are_rejected(self):
        p, y = self.data()
        with tempfile.TemporaryDirectory() as tmp:
            record = adapter.fit_calibrator(p,y,{},Path(tmp)/'fit.json',adapter.CANDIDATES['event'])
            self.assertGreater(np.abs(adapter.predict_calibrator(p,record).sum(1)-p.sum(1)).max(), 1e-6)
            record['coefficients'][0] = .1
            with self.assertRaises(ValueError):adapter.predict_calibrator(p,record)

    def test_shared_reference_is_reused_and_tampering_refused(self):
        from types import SimpleNamespace
        rng=np.random.default_rng(2026)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';work.mkdir()
            fit=np.arange(30);valid=np.arange(30,40)
            raw={'y':np.zeros((40,10),dtype=np.int8)};raw['y'][::2,0]=1
            data={'X':rng.normal(size=(400,2)).astype(np.float32),'feature_names':np.array(['a','b'])}
            def features(*args):
                idx=args[5]
                return {'X':data['X'].reshape(40,10,2)[idx].reshape(-1,2),'feature_names':data['feature_names']}
            inv={'parameters':{member:{'development':{'iterations':2,'depth':2,'thread_count':1,'allow_writing_files':False}} for member in MEMBERS}}
            case=SimpleNamespace(name='forward',meta_cutoff=JAN20+2*DAY,target_cutoff=JAN20+4*DAY)
            with patch.object(postprocessing,'features',side_effect=features):
                a,files=postprocessing.reference_scores(root,work,{'spec':{'schema_version':3}},inv,raw,fit,valid,case)
                b,again=postprocessing.reference_scores(root,work,{'spec':{'schema_version':3}},inv,raw,fit,valid,case)
                np.testing.assert_array_equal(a,b);self.assertEqual(files,again)
                next((work/'reference-combiners').glob('*.cbm')).write_bytes(b'corrupt')
                with self.assertRaises(ValueError):postprocessing.reference_scores(root,work,{'spec':{'schema_version':3}},inv,raw,fit,valid,case)

    def test_real_saved_calibrator_synthetic_final_csv_and_independent_replay(self):
        p, y = self.data()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';work.mkdir()
            state=work/'state.json';adapter.fit_calibrator(p,y,{},state,adapter.CANDIDATES['joint'])
            # Real synthetic source data, saved state, baseline loader, exporter and CSV validator.
            # The private neural V6 replay is intentionally inapplicable to synthetic validation.
            rowids=np.array([[f'e-{i}-{j}' for j in range(10)] for i in range(len(p))])
            baseline=root/'baseline.npz'
            save_arrays(baseline,version=np.array('synthetic'),member_names=np.array(list(MEMBERS)),members=np.stack([p]*5),scores=p,rowids=rowids)
            evaluation={'rowids':rowids}
            example=root/'example.csv'
            import csv
            with example.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['rowid','predicted_probability']);writer.writerows((v,0) for v in rowids.ravel())
            m={'spec':{'schema_version':3,'purpose':'synthetic_validation','inputs':{'baseline_evaluation':{'path':'baseline.npz'},'example':{'path':'example.csv'}}}}
            sys.path.insert(0,str(ROOT/'legacy'))
            from submission_export import export_submission,validate_csv
            from row_auc_final_utils import verify_csv_probabilities
            with patch.object(engine,'events',return_value=evaluation),patch.object(engine,'_private_replay') as private:
                one,a=postprocessing.final_probabilities(root,work,m,adapter,state,1)
                two,b=postprocessing.final_probabilities(root,work,m,adapter,state,2)
            private.assert_not_called();np.testing.assert_array_equal(a,b)
            for number,ids,scores in [(1,one,a),(2,two,b)]:
                output=work/f'{number}.csv';export_submission(ids,scores,example,output,len(ids))
                validate_csv(output,example,len(ids));verify_csv_probabilities(output,ids,scores)
            self.assertEqual((work/'1.csv').read_bytes(),(work/'2.csv').read_bytes())
            with self.assertRaises(FileExistsError):export_submission(one,a,example,work/'1.csv',len(one))


if __name__ == '__main__':unittest.main()
