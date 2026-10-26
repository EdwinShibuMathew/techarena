"""Cross-campaign reuse and corruption boundaries for feature caching."""
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from ml_workflow.cache import arrays_identity, get_or_compute
from ml_workflow.contracts import canonical


class SharedFeatureCacheTests(unittest.TestCase):
    def test_reuse_invalidation_and_atomic_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            identity={"feature_names":["a","b"],"target":arrays_identity({"i":np.ones((2,10),dtype=np.int64)}),
                      "prefix":{"labels":"one"},"checkpoint":"one","extractor":"one","runtime":"one"}
            data={"X":np.arange(40,dtype=np.float32).reshape(20,2),"feature_names":np.array(["a","b"]),
                  "probs":np.full((2,10),.1+1e-10,dtype=np.float64),"raw_probs":np.full((2,10),.2,dtype=np.float32)}
            calls=[]
            def compute():calls.append(1);return data
            with ThreadPoolExecutor(2) as pool:
                outputs=list(pool.map(lambda _:get_or_compute(root,identity,compute),range(2)))
            self.assertEqual(outputs[1][0]["probs"].dtype,np.float64)
            np.testing.assert_array_equal(outputs[1][0]["probs"],data["probs"])
            self.assertEqual(len(calls),1);self.assertEqual(sorted(o[1]["hit"] for o in outputs),[False,True])
            for field in ("prefix","checkpoint","extractor","runtime","feature_names","target"):
                changed=copy.deepcopy(identity)
                if field=="feature_names":
                    changed[field]=["b","a"]; value={**data,"feature_names":np.array(["b","a"])}
                elif field=="target":
                    changed[field]=arrays_identity({"i":np.zeros((2,10),dtype=np.int64)});value=data
                else:changed[field]="different";value=data
                _,receipt=get_or_compute(root,changed,lambda:value)
                self.assertFalse(receipt["hit"])
            folder=root/"artifacts/cache/ml-workflow/v1"/canonical(identity)
            (folder/"values.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"integrity"):get_or_compute(root,identity,compute)
            (folder/"metadata.json").unlink() # interrupted write: an orphan is never a hit
            _,receipt=get_or_compute(root,identity,compute);self.assertFalse(receipt["hit"])
            before=(folder/"values.npz").read_bytes()
            (folder/"metadata.json").unlink()
            def fail():raise RuntimeError("extraction failure")
            with self.assertRaises(RuntimeError):get_or_compute(root,identity,fail)
            self.assertEqual((folder/"values.npz").read_bytes(),before)

    def test_dtype_and_order_are_dependencies(self):
        values=np.array([1,2],dtype=np.int32)
        self.assertNotEqual(arrays_identity({"x":values}),arrays_identity({"x":values[::-1]}))
        self.assertNotEqual(arrays_identity({"x":values}),arrays_identity({"x":values.astype(np.int64)}))
        with self.assertRaises(ValueError):arrays_identity({"x":np.array([object()])})

    def test_actual_request_reuses_across_trials_but_binds_all_extraction_inputs(self):
        import importlib.util
        import types
        from unittest.mock import patch
        from ml_workflow.cache import request
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ("extract.py","tools/ml_workflow/contracts.py","tools/ml_workflow/cache.py"):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("def extract(): pass\n" if name=="extract.py" else "# fixed\n")
            spec=importlib.util.spec_from_file_location("cache_fixture",root/"extract.py")
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.EXTRACTOR_FILES=("extract.py",);module.FEATURE_NAMES=["a"]
            checkpoint=root/"model.pt";checkpoint.write_bytes(b"fixed checkpoint")
            prefix={"y":np.array([[1,0]]),"t":np.array([1])}
            target={"i":np.arange(20).reshape(2,10),"rowids":np.arange(20).reshape(2,10)}
            with patch.dict(sys.modules,{"torch":types.SimpleNamespace(__version__="fixture")}):
                original=request(root,module,checkpoint,prefix,target,2)
                (root/"trainer.py").write_text("# unrelated training change")
                (root/"report.py").write_text("# unrelated reporting change")
                self.assertEqual(original,request(root,module,checkpoint,prefix,target,2))
                variants=[]
                variants.append(request(root,module,checkpoint,prefix,target,3))
                variants.append(request(root,module,checkpoint,{**prefix,"y":1-prefix["y"]},target,2))
                variants.append(request(root,module,checkpoint,prefix,{k:v[::-1] for k,v in target.items()},2))
                module.EXTRACTION_PRECISION="float64";variants.append(request(root,module,checkpoint,prefix,target,2))
                module.EXTRACTION_PRECISION="float32";checkpoint.write_bytes(b"changed checkpoint")
                variants.append(request(root,module,checkpoint,prefix,target,2))
                self.assertTrue(all(original!=v for v in variants))
