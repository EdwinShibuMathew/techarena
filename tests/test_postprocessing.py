"""Required contracts for schema-v2 calibration and user-grouped cross-fitting."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from ml_workflow import engine,postprocessing
from ml_workflow.contracts import ExperimentSpec,MEMBERS
from ml_workflow.storage import read
spec=importlib.util.spec_from_file_location('entropy_adapter',ROOT/'experiments/v7.1-entropy-sharpening-v1/adapter.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
ML=all(importlib.util.find_spec(x) is not None for x in ('torch','catboost','scipy'))


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
