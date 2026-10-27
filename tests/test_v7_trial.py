"""V7-relative acceptance is explicit, reproducible and never full promotion."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from ml_workflow import engine, policy
from ml_workflow.contracts import ExperimentSpec,digest
from ml_workflow.storage import read,write_new
spec=importlib.util.spec_from_file_location('v7_anchor_adapter',ROOT/'experiments/v7.3-v7-anchor-v1/adapter.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
ML=all(importlib.util.find_spec(x) is not None for x in ('torch','catboost','scipy'))


def metric(gain=.002,low=.001):
    return {'baseline_auc':.8,'candidate_auc':.8+gain,'auc_diff':gain,'ci_lower':low,'ci_upper':.01,'resamples':1000,'anchor_sha256':policy.V7_SHA}


class TrialContractTests(unittest.TestCase):
    def test_policy_requires_explicit_identity_and_legacy_default_unchanged(self):
        value=read(ROOT/'experiments/v7.3-v7-anchor-v1/experiment.json');ExperimentSpec.parse(value)
        for key,bad in [('release_policy','unknown'),('upload_comparator','V6'),('anchor_gamma',1.2),('anchor_sha256','0'*64),('selection','maximin_nonnegative_origins_and_late')]:
            changed=deepcopy(value);changed[key]=bad
            with self.assertRaises(ValueError):ExperimentSpec.parse(changed)
        previous=read(ROOT/'experiments/v7.2-joint-calibration-v1/experiment.json')
        ExperimentSpec.parse(previous);self.assertFalse(policy.is_trial(previous))
        with self.assertRaises(ValueError):
            previous['upload_comparator']='V7';ExperimentSpec.parse(previous)
        policy.validate_anchor(ROOT,value)

    def test_selection_and_trial_gates_use_v7_without_silent_late_gate(self):
        value=read(ROOT/'experiments/v7.3-v7-anchor-v1/experiment.json')
        rows={candidate+'/'+case['name']:{'auc_diff':-.01,'v7_comparison':metric(),'diagnostics':{'late':{'events':3,'auc_diff':-.01}}}
              for candidate in value['candidate_order'] for case in value['development']}
        self.assertEqual(engine._winner(value,rows),'mass')
        rows['mass/jan24_forward']['v7_comparison']=metric(gain=-.001)
        self.assertEqual(engine._winner(value,rows),'entropy_mass')
        screen={name:{**metric(gain=-.001,low=-.002),'v7_comparison':metric()} for name in ('primary','secondary')}
        five={**metric(gain=-.001,low=-.002),'late':metric(gain=-.001),'v7_comparison':{**metric(),'late':metric(gain=-.003)}}
        self.assertTrue(policy.stage_passed(value,'screen',screen));self.assertTrue(policy.stage_passed(value,'stress',{'five_day':five}))
        self.assertFalse(policy.stage_passed({},'screen',screen));self.assertFalse(policy.stage_passed({},'stress',{'five_day':five}))
        screen['primary']['v7_comparison']['ci_lower']=0
        self.assertFalse(policy.stage_passed(value,'screen',screen))
        screen['primary']['v7_comparison']['ci_lower']=.001
        screen['secondary']['v7_comparison']=metric(gain=-.001)
        self.assertFalse(policy.stage_passed(value,'screen',screen))
        five['v7_comparison']['ci_lower']=0
        self.assertFalse(policy.stage_passed(value,'stress',{'five_day':five}))

    def test_checked_recomputes_trial_evidence_and_refuses_missing_or_changed_comparison(self):
        value=read(ROOT/'experiments/v7.3-v7-anchor-v1/experiment.json')
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);write_new(work/'manifest.json',{'spec':value})
            results={name:{**metric(),'v7_comparison':metric()} for name in ('primary','secondary')}
            def report():
                (work/'screen.json').write_text(json.dumps({'complete':True,'passed':True,'files':{},'manifest_sha256':digest(work/'manifest.json'),'results':results}))
                (work/'screen.seal.json').write_text(json.dumps({'sha256':digest(work/'screen.json')}))
            report();engine.checked(work,'screen')
            results['primary']['v7_comparison']['ci_lower']=-.001;report()
            with self.assertRaises(ValueError):engine.checked(work,'screen')
            results['primary']['v7_comparison']=metric();results['primary']['v7_comparison']['anchor_sha256']='0'*64;report()
            with self.assertRaises(ValueError):engine.checked(work,'screen')
            with self.assertRaises(ValueError):policy.validate_metric({'auc_diff':.1},True)

    def test_trial_receipt_cannot_claim_production_promotion(self):
        trial={'release_policy':'v7_upload_trial'}
        flags=policy.release_flags(trial)
        self.assertTrue(flags['upload_ready']);self.assertFalse(flags['submission_ready']);self.assertFalse(flags['promotion_supported'])
        self.assertTrue(policy.receipt_ready(trial,flags));self.assertFalse(policy.receipt_ready({},flags))
        bad=deepcopy(flags);bad['promotion_supported']=True
        self.assertFalse(policy.receipt_ready(trial,bad))
        self.assertFalse(policy.receipt_ready({'release_policy':'unknown'},{'submission_ready':True}))
        self.assertTrue(policy.receipt_ready({}, {'submission_ready':True}))

    def test_trial_publication_and_verification_preserve_separate_readiness(self):
        value=read(ROOT/'experiments/v7.3-v7-anchor-v1/experiment.json')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'artifacts/experiments/run';work.mkdir(parents=True);(root/'configs').mkdir()
            m={'run_id':'run','spec':value,'source':{'commit':'synthetic-publication','sha256':'f'*64}}
            write_new(work/'manifest.json',m)
            def report(stage,results,**extra):
                path=work/(stage+'.json')
                write_new(path,{'complete':True,'passed':True,'files':{},'manifest_sha256':digest(work/'manifest.json'),'results':results,**extra})
                write_new(work/(stage+'.seal.json'),{'sha256':digest(path)})
            development={candidate+'/'+case['name']:{**metric(),'v7_comparison':metric()}
                         for candidate in value['candidate_order'] for case in value['development']}
            report('develop',development,winner='mass')
            report('screen',{name:{**metric(),'v7_comparison':metric()} for name in ('primary','secondary')})
            report('stress',{'five_day':{**metric(),'late':metric(gain=-.001),
                                        'v7_comparison':{**metric(),'late':metric(gain=-.002)}}})
            # Publication control fixture; numerical CSV validation is exercised separately.
            (work/'candidate.csv').write_bytes(b'synthetic-verified-csv')
            write_new(work/'finalize.json',{'complete':True,'independent_replay':True,'manifest_sha256':digest(work/'manifest.json'),
                      'rows':529390,'csv_sha256':digest(work/'candidate.csv'),'files':{'candidate.csv':digest(work/'candidate.csv')}})
            write_new(work/'finalize.seal.json',{'sha256':digest(work/'finalize.json')})
            for stage in ('develop','screen','stress','finalize'):write_new(work/(stage+'.ledger.json'),{'exit_code':0})
            with patch.object(engine,'load',return_value=m):
                engine.publish(root,work,m)
                actual=engine.verify(root,'run')
                self.assertTrue(actual['upload_ready']);self.assertFalse(actual['submission_ready'])
                config=root/'configs/v7_3_trial.json';receipt=read(config)
                self.assertFalse(receipt['promotion_supported']);self.assertFalse(receipt['v6_gate_results']['late'])
                receipt['promotion_supported']=True
                config.write_text(json.dumps(receipt));(work/'published.json').write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):engine.verify(root,'run')

    def test_trial_accounting_failure_prevents_any_publication(self):
        from ml_workflow import budget as budget_module
        from ml_workflow.budget import Budget
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'artifacts/experiments/run';work.mkdir(parents=True)
            m={'spec_sha256':'f'*64,'spec':{'purpose':'research','release_policy':'v7_upload_trial','release_name':'v7_3_trial','budget':{'phase_seconds':1,'total_seconds':10}},'carried_seconds':0.,'carry_identity':None,'source':{'commit':'fixture'}}
            ledger=root/'artifacts/experiments/compute-ledger.json';original=budget_module.write_json
            def fail(path,value):
                if path==ledger and value['attempts'][-1]['status']=='complete':raise OSError('accounting failure')
                original(path,value)
            with patch.object(engine,'locate',return_value=work),patch.object(engine,'load',return_value=m),patch.object(engine,'checked',return_value={}),patch.object(Budget,'run',return_value=0),patch.object(budget_module,'write_json',side_effect=fail),patch.object(engine,'publish') as publish:
                with self.assertRaises(OSError):engine.run(root,'run','finalize')
                publish.assert_not_called()
            self.assertFalse((root/'configs/v7_3_trial.json').exists())
            self.assertFalse((work/'published.json').exists())


@unittest.skipUnless(ML,'CPU ML dependencies required; enforced by model-contracts CI')
class TrialModelTests(unittest.TestCase):
    def test_real_v7_anchor_fitting_normalization_reload_and_csv_replay(self):
        from predict_v7 import calibrate_v7_probabilities
        rng=np.random.default_rng(2026);p=rng.uniform(.001,.12,(80,10));p[0]=0;p[1,0]=0;p[2]=0;p[2,0]=1;p[3,0]=p[3,1]
        anchor=calibrate_v7_probabilities(p,1.12)
        y=np.zeros_like(p,dtype=np.int8);y[np.arange(1,80),p[1:].argmax(1)]=1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,params in adapter.CANDIDATES.items():
                path=root/(name+'.json');metadata={'candidate':name,'cutoff':123}
                state=adapter.fit_calibrator(p,y,deepcopy(metadata),path,deepcopy(params))
                self.assertEqual(state,adapter.load_calibrator(path,metadata,params))
                self.assertLessEqual(state['objective'],state['identity_objective']+1e-12)
                lm=np.log(np.maximum(anchor.sum(1),1e-12))
                self.assertEqual(state['normalization'],{'log_mass_mean':float(lm.mean()),'log_mass_scale':max(float(lm.std()),1e-6)})
                identity=deepcopy(state);identity['coefficients']=[0]*4
                np.testing.assert_array_equal(adapter.predict_calibrator(p,identity),anchor)
                first=adapter.predict_calibrator(p,state);second=adapter.predict_calibrator(p,adapter.load_calibrator(path))
                np.testing.assert_array_equal(first,second)
                np.testing.assert_array_equal(first[anchor==0],0);np.testing.assert_array_equal(first[anchor==1],1)
                for i in range(10):
                    for j in range(i):np.testing.assert_array_equal(np.sign(first[:,i]-first[:,j]),np.sign(p[:,i]-p[:,j]))
                permutation=np.arange(9,-1,-1)
                np.testing.assert_allclose(adapter.predict_calibrator(p[:,permutation],state),first[:,permutation],atol=1e-15,rtol=0)
                with self.assertRaises(ValueError):adapter.load_calibrator(path,{'cutoff':999})
                state['coefficients'][0]=.1
                with self.assertRaises(ValueError):adapter.predict_calibrator(p,state)
            import csv
            sys.path.insert(0,str(ROOT/'legacy'))
            from submission_export import export_submission,validate_csv
            from row_auc_final_utils import verify_csv_probabilities
            ids=np.array([f'e-{i}' for i in range(p.size)]);example=root/'example.csv'
            with example.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['rowid','predicted_probability']);writer.writerows((i,0) for i in ids)
            for number,scores in [(1,first),(2,second)]:
                output=root/f'{number}.csv';export_submission(ids,scores,example,output,len(ids));validate_csv(output,example,len(ids));verify_csv_probabilities(output,ids,scores)
            self.assertEqual((root/'1.csv').read_bytes(),(root/'2.csv').read_bytes())


if __name__=='__main__':unittest.main()
