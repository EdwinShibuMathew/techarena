"""Residual identities, causal selection, frozen training and release refusal."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
sys.path.insert(0,str(ROOT/"experiments/v8-residual-event"))
import residual_core as core
import predict_v8_residual as predictor
spec = importlib.util.spec_from_file_location("residual_test_runner",ROOT/"experiments/v8-residual-event/runner.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
TORCH = importlib.util.find_spec("torch") is not None


def development(gains,late=.001,no_click=.001):
    return {"origins":[{"origin":origin,"baseline_auc":.8,"candidate_auc":.8+gain,"auc_diff":gain,
        "no_click_contribution":no_click,"late":None if origin == "jan24" else {"auc_diff":late}}
        for origin,gain in zip(core.ORIGINS,gains)]}


class ResidualTests(unittest.TestCase):
    def test_development_boundaries_preserve_first_users_and_exclude_jan25(self):
        start = core.FOLDS["jan21"]
        raw = {"u":np.array([1,2,1,2,3,3,4,5]),
               "t":start+np.array([0,7,8,9,24,25,72,96])*3600}
        fit,valid = core.development_indices(raw,"jan21")
        np.testing.assert_array_equal(fit,[0,1]); np.testing.assert_array_equal(valid,[2,3,4,6])
        self.assertFalse(np.intersect1d(fit,valid).size)
        with self.assertRaises(ValueError):
            core.development_indices(raw,"jan25")

    def test_selection_uses_worst_origin_late_and_no_click_evidence(self):
        d = {"A":development((.001,.001,.001)),"B":development((.003,.0001,.002)),"C":development((.0005,.0005,.0005))}
        self.assertEqual(core.choose(d),"A")
        d["A"] = development((.001,.001,.001),late=-.0001)
        self.assertEqual(core.choose(d),"C")
        d["C"] = development((.002,.002,.002),no_click=-.0001)
        self.assertEqual(core.choose(d),"B")
        d["B"] = development((.001,-.0001,.001))
        self.assertIsNone(core.choose(d))
        self.assertEqual(core.choose({k:development((.001,.001,.001)) for k in ("C","B","A")}),"A")
        d["A"]["origins"][0]["late"]["auc_diff"] = np.nan
        with self.assertRaises(ValueError):
            core.choose(d)

    def test_composition_preserves_original_weights_and_averaging(self):
        rng = np.random.default_rng(1)
        original = {m:rng.random((4,10)) for m in core.MEMBERS}
        baseline = core.original_ensemble(original)
        for key,members in core.COMPOSITIONS.items():
            np.testing.assert_array_equal(core.compose(baseline,original,{m:original[m] for m in members},key),baseline)
            updated = {m:rng.random((4,10)) for m in members}
            expected = core.original_ensemble({m:updated.get(m,p) for m,p in original.items()})
            np.testing.assert_array_equal(core.compose(baseline,original,updated,key),expected)
        with self.assertRaises(ValueError):
            core.compose(baseline+.01,original,{m:original[m] for m in core.COMPOSITIONS["A"]},"A")

    def test_no_eligible_candidate_blocks_training_and_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = {"code_sha256":{}}
            core.write_new(root/core.WORK/"inputs.json",{})
            core.write_new(root/core.REPORTS/"development.json",{"complete":True,"protocol_sha256":core.canonical(core.PROTOCOL),
                "source_sha256":{},"input_manifest_sha256":core.sha256(root/core.WORK/"inputs.json"),"files_sha256":{},
                "winner":None,"results":{k:development((-.001,.001,.001)) for k in core.COMPOSITIONS}})
            with patch.object(core,"load_manifest",return_value=manifest),patch.object(runner,"train_fold") as train:
                for action in (runner.screen,runner.stress,runner.finalize):
                    with self.assertRaisesRegex(ValueError,"No eligible residual composition"):
                        action(root)
                train.assert_not_called()
            self.assertFalse((root/"configs/v8.json").exists())
            self.assertFalse((root/"artifacts/submissions/v8_candidate.csv").exists())

    def test_configuration_is_published_only_after_verified_csv_and_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); events = []
            config = {"submission_ready":False}
            def exported(*args):
                self.assertFalse((root/"configs/v8.json").exists()); events.append("csv")
            with patch.object(predictor,"validate_release"),patch.object(predictor,"export_verified",side_effect=exported):
                predictor.publish_ready(root,config,np.array(["r"]),np.array([.1]),"abc",{},[])
            self.assertEqual(events,["csv"])
            self.assertTrue(core.read_json(root/"configs/v8.json")["submission_ready"])
            self.assertTrue(core.read_json(root/core.REPORTS/"release.json")["independent_csv_replay"])
            before = (root/"configs/v8.json").read_bytes()
            with patch.object(predictor,"validate_release"),patch.object(predictor,"export_verified",side_effect=ValueError("different CSV")):
                with self.assertRaises(ValueError):
                    predictor.publish_ready(root,config,np.array(["r"]),np.array([.2]),"def",{},[])
            self.assertEqual((root/"configs/v8.json").read_bytes(),before)

    def test_failed_screen_is_recomputed_and_blocks_stress_and_final_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = {"code_sha256":{}}
            core.write_new(root/core.WORK/"inputs.json",{})
            core.write_new(root/core.REPORTS/"selected.json",{})
            def metric(gain,low):
                return {"baseline_auc":.8,"candidate_auc":.8+gain,"auc_diff":gain,"ci_lower":low,
                    "ci_upper":.002,"n_bootstrap":1000,"valid_bootstrap_samples":1000}
            failed = {"complete":True,"protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":{},
                "input_manifest_sha256":core.sha256(root/core.WORK/"inputs.json"),
                "selection_sha256":core.sha256(root/core.REPORTS/"selected.json"),"composition":"A",
                "results":{"primary":metric(.001,-.0001),"secondary":metric(.001,.0001)},
                "decision_checks":{"primary_ci_lower_positive":True,"secondary_change_nonnegative":True},"passed":True}
            core.write_new(root/core.REPORTS/"screen.json",failed)
            with patch.object(core,"load_manifest",return_value=manifest),patch.object(core,"selection",return_value="A"),patch.object(runner,"train_fold") as train:
                for action in (runner.stress,runner.finalize):
                    with self.assertRaisesRegex(ValueError,"screen gate failed"):
                        action(root)
                train.assert_not_called()
            self.assertFalse((root/core.WORK/"budget.json").exists())

    def test_replay_rejects_wrong_recipe_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"version":"v8_robust_ensemble","protocol_sha256":core.canonical(core.PROTOCOL)}
            with patch.object(core,"load_manifest",return_value={}),patch.object(core,"selection",return_value="A"),patch.object(predictor,"replay_v6") as replay:
                with self.assertRaisesRegex(ValueError,"Invalid residual release recipe"):
                    predictor.compute_candidate(root,config,"cpu")
                replay.assert_not_called()


@unittest.skipUnless(TORCH,"torch unavailable in source-only CI")
class ResidualNeuralTests(unittest.TestCase):
    def make_data(self,dimension):
        import torch
        from train_final_models import PreferenceModel,EventModel
        torch.manual_seed(4)
        active = torch.arange(10) < 6
        base = EventModel(PreferenceModel(10,30,dimension,active),10,30).requires_grad_(False)
        rng = np.random.default_rng(4)
        train = {"u":np.arange(32)%6,"i":rng.integers(0,30,(32,10)),"t":np.arange(32)+100,
            "tb":np.arange(32)%4,"dt":np.zeros(32,dtype=int),"h":rng.integers(0,30,(32,20)),
            "ht":np.broadcast_to(np.arange(1,21),(32,20)).copy(),"y":np.zeros((32,10),dtype=np.int8)}
        train["y"][np.arange(0,32,2),0] = 1
        target = {k:v[:3].copy() for k,v in train.items() if k != "y"}
        target["u"] = np.array([0,7,8]); target["t"][:] = 200
        target["h"][2] = -1; target["ht"][2] = 0
        return base,train,target

    def test_zero_identity_masked_history_and_permutation_in_both_dimensions(self):
        import torch
        from residual_model import ResidualAdapter,pack,representations
        for dimension in (8,16):
            base,train,target = self.make_data(dimension)
            data = pack(target,train["u"],10,30,"cpu")
            scores,common,choice = representations(base,data)
            for kind in ("event","both"):
                adapter = ResidualAdapter(dimension,kind)
                corrected,delta,event,relative = adapter(scores,common,choice)
                torch.testing.assert_close(corrected,scores,rtol=0,atol=0)
                self.assertTrue(torch.isfinite(choice).all())
                self.assertTrue(torch.equal(choice[2,:,dimension:2*dimension],torch.zeros((10,dimension))))
                # Unsupported cold-status inputs cannot affect the event head.
                altered = common.clone(); altered[:,-1] += 100
                torch.testing.assert_close(adapter(scores,altered,choice)[0],corrected,rtol=0,atol=0)
                if kind == "both":
                    with torch.no_grad():
                        adapter.choice[-1].weight.fill_(.1)
                permutation = torch.arange(9,-1,-1)
                changed = list(data); changed[1] = data[1][:,permutation]
                s,c,q = representations(base,tuple(changed))
                actual = adapter(s,c,q)[0]
                expected = adapter(scores,common,choice)[0][:,permutation]
                torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
                relative = adapter(scores,common,choice)[3]
                torch.testing.assert_close(relative.mean(1),torch.zeros(3),rtol=0,atol=1e-7)

    def test_equal_future_history_and_bad_ids_are_rejected(self):
        from residual_model import pack
        _,train,target = self.make_data(8)
        for bad in (200,201):
            changed = copy.deepcopy(target); changed["ht"][0,0] = bad
            with self.assertRaisesRegex(ValueError,"History crosses"):
                pack(changed,train["u"],10,30,"cpu")
        target["i"][0,0] = 30
        with self.assertRaises(ValueError):
            pack(target,train["u"],10,30,"cpu")

    def test_zero_pipeline_matches_raw_fallback_and_all_37_features(self):
        import torch
        from residual_model import pipeline,ResidualAdapter
        from predict_v6 import portable_pipeline
        from build_submission import neural_pipeline
        for dimension in (8,16):
            base,train,target = self.make_data(dimension)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); checkpoint = root/"base.pt"; positive = root/"positive.npz"
                torch.save({"n_users":10,"n_items":30,"dimension":dimension,"epoch":8,"model_state":base.state_dict()},checkpoint)
                np.savez(positive,users=np.repeat(np.arange(6),4),items=np.arange(24)%30)
                adapter = ResidualAdapter(dimension,"both").eval()
                actual,names,r,p,old_r,old_p,_ = pipeline(checkpoint,adapter,train,target,positive,torch.device("cpu"))
                expected,expected_names,*_ = portable_pipeline(neural_pipeline)(checkpoint,train,target,positive,torch.device("cpu"))
                self.assertEqual(names,expected_names)
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_allclose(r,old_r,rtol=0,atol=1e-15)
                np.testing.assert_allclose(p,old_p,rtol=0,atol=1e-15)
                self.assertTrue((p.sum(1)<=1+1e-8).all())

    def test_real_adapter_training_changes_only_heads_and_reloads(self):
        import torch
        from residual_model import ResidualAdapter
        from train_adapter import fit
        for dimension,member in ((8,"dim8_seed42"),(16,"dim16_seed42")):
            base,arrays,_ = self.make_data(dimension)
            arrays["indices_sha256"] = "synthetic"
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); checkpoint = root/"base.pt"; output = root/"adapter.pt"
                torch.save({"n_users":10,"n_items":30,"dimension":dimension,"epoch":8,"model_state":base.state_dict()},checkpoint)
                before = checkpoint.read_bytes()
                result = fit(checkpoint,arrays,member,"both",1000,output,"cpu")
                self.assertEqual(checkpoint.read_bytes(),before)
                self.assertTrue(result["base_weights_unchanged"])
                self.assertTrue(torch.count_nonzero(result["adapter_state"]["event.2.weight"]))
                reloaded = torch.load(output,weights_only=True)
                adapter = ResidualAdapter(dimension,"both"); adapter.load_state_dict(reloaded["adapter_state"])
                for k,v in result["adapter_state"].items():
                    torch.testing.assert_close(adapter.state_dict()[k],v,rtol=0,atol=0)
                with self.assertRaisesRegex(ValueError,"Preserve existing"):
                    fit(checkpoint,arrays,member,"both",1000,output,"cpu")

    def test_cached_features_reject_changed_sources_and_tampered_bytes(self):
        _,raw,_ = self.make_data(8)
        raw["t"][0] = core.FOLDS["jan21"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/"base.pt").write_bytes(b"bound-base")
            manifest = {"original_models":{"dim8_seed42":{"jan21":"base.pt"}},"inputs_sha256":{},"code_sha256":{},
                "prefixes":{"jan21":"positive.npz"}}
            result = (np.ones((10,37),dtype=np.float32),list(core.BASE_COLUMNS))
            with patch("predict_v6.portable_pipeline",return_value=lambda *args:result) as pipeline:
                actual = runner.features(root,manifest,raw,"dim8_seed42","jan21",np.array([0]),"stale",SimpleNamespace(check=lambda:None))
                np.testing.assert_array_equal(actual,result[0])
                manifest["code_sha256"] = {"changed":"0"*64}
                with self.assertRaisesRegex(ValueError,"Stale residual feature cache"):
                    runner.features(root,manifest,raw,"dim8_seed42","jan21",np.array([0]),"stale",SimpleNamespace(check=lambda:None))
                manifest["code_sha256"] = {}
                path = root/core.WORK/"cache/stale.npz"
                with path.open("ab") as stream:
                    stream.write(b"tampered")
                with self.assertRaisesRegex(ValueError,"Stale residual feature cache"):
                    runner.features(root,manifest,raw,"dim8_seed42","jan21",np.array([0]),"stale",SimpleNamespace(check=lambda:None))
                self.assertEqual(pipeline.call_count,1)


if __name__ == "__main__":
    unittest.main()
