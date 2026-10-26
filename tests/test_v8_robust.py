"""Full-ensemble temporal selection, typed neural training and release regressions."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
sys.path.insert(0,str(ROOT/"experiments/v8-robust-ensemble"))
import robust_core as core
import train_member
import predict_v8_candidate as predictor
spec = importlib.util.spec_from_file_location("robust_test_runner",ROOT/"experiments/v8-robust-ensemble/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def development(gains,no_click=(.0001,.0001)):
    return {"folds":[{"fold":i,"baseline_auc":.8,"candidate_auc":.8+gain,"auc_diff":gain,"no_click_contribution":no_click[i]}
                     for i,gain in enumerate(gains)]}


class RobustTests(unittest.TestCase):
    def test_only_jan21_forward_folds_keep_events_whole(self):
        start = core.FOLDS["jan21"]
        times = start+np.array([0,7,8,15,16,23])*3600
        a,b = core.development_folds(times)
        np.testing.assert_array_equal(np.flatnonzero(a[0]),[0,1])
        np.testing.assert_array_equal(np.flatnonzero(a[1]),[2,3])
        np.testing.assert_array_equal(np.flatnonzero(b[0]),[0,1,2,3])
        np.testing.assert_array_equal(np.flatnonzero(b[1]),[4,5])
        self.assertEqual(np.repeat(a[0],10).sum(),20)
        for bad in (start-1,start+core.DAY,core.FOLDS["jan25"]):
            with self.assertRaises(ValueError):
                core.development_folds(np.r_[times,bad])

    def test_worst_fold_beats_larger_average_and_negative_no_click_is_rejected(self):
        results = {"A":development((.0005,.0004)),"B":development((.002,.00001))}
        self.assertEqual(core.choose(results),"A")
        results["A"] = development((.0005,.0005),(-.002,.0001))
        self.assertEqual(core.choose(results),"B")
        results["B"] = development((.002,-.00001))
        self.assertIsNone(core.choose(results))
        self.assertEqual(core.choose({"B":development((.0004,.0004)),"A":development((.0004,.0004))}),"A")
        results = {"A":development((0,0)),"B":development((0,0))}
        self.assertIsNone(core.choose(results))
        results["A"]["folds"][0]["no_click_contribution"] = np.nan
        with self.assertRaises(ValueError):
            core.choose(results)

    def test_fixed_weight_member_replacement_and_identity(self):
        rng = np.random.default_rng(42)
        original = {m:rng.uniform(.01,.9,(10,10)) for m in core.MEMBERS}
        baseline = core.original_ensemble(original)
        for key,members in core.COMPOSITIONS.items():
            np.testing.assert_array_equal(core.compose(baseline,original,{m:original[m] for m in members},key),baseline)
            new = {m:rng.uniform(.01,.9,(10,10)) for m in members}
            expected = core.original_ensemble({m:new.get(m,p) for m,p in original.items()})
            np.testing.assert_allclose(core.compose(baseline,original,new,key),expected,rtol=0,atol=1e-15)
        with self.assertRaises(ValueError):
            core.compose(baseline,original,{"dim8_seed44":original["dim8_seed44"]},"A")

    def test_cross_group_pair_decomposition_matches_independent_enumeration(self):
        y = np.zeros((4,10),dtype=np.int8)
        y[[0,2],[0,1]] = 1
        rng = np.random.default_rng(42)
        baseline = np.round(rng.random(y.shape),1)
        candidate = np.round(rng.random(y.shape),1)
        def brute(p):
            pos,neg = p[y==1],p[y==0]
            return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
        result = core.pair_decomposition(y,baseline,candidate,np.array([0,0,1,1]))
        self.assertAlmostEqual(result["reconstructed_change"],brute(candidate)-brute(baseline),places=12)
        self.assertEqual(sum(c["pairs"] for c in result["cells"].values()),int(y.sum())*int((y==0).sum()))

    def test_no_winner_blocks_all_later_stages_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"code_sha256":{}}
            core.write_new(root/core.WORK/"inputs.json",{})
            core.write_new(root/core.REPORTS/"development.json",{"complete":True,"source_sha256":{},
                "protocol_sha256":core.canonical(core.PROTOCOL),"input_manifest_sha256":core.sha256(root/core.WORK/"inputs.json"),
                "results":{"A":development((-.01,-.01)),"B":development((.01,-.01))},"winner":None,
                "promotion_labels_used_for_selection":False})
            with patch.object(core,"load_manifest",return_value=manifest),patch.object(runner,"train_fold") as train:
                for action in (runner.screen,runner.stress,runner.finalize):
                    with self.assertRaisesRegex(ValueError,"No qualifying composition"):
                        action(root)
                train.assert_not_called()
            self.assertFalse((root/core.WORK/"budget.json").exists())

    def test_missing_ready_marker_and_interrupted_publication_cannot_generate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core.write_new(root/"configs/v8.json",{"submission_ready":False})
            with patch.object(predictor,"compute_candidate") as compute,self.assertRaisesRegex(ValueError,"not submission-ready"):
                predictor.generate(root,root/"artifacts/out.csv")
            compute.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(predictor,"validate_release"),patch.object(predictor,"export_verified",side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    predictor.publish_ready(root,{},[],[],"hash",{},[])
            self.assertFalse((root/"configs/v8.json").exists())

    def test_failed_primary_cannot_be_overridden_by_passed_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"code_sha256":{}}
            core.write_new(root/core.WORK/"inputs.json",{})
            core.write_new(root/core.REPORTS/"selected.json",{"composition":"A"})
            metric = {"baseline_auc":.8,"candidate_auc":.81,"auc_diff":.01,"ci_lower":-.001,"ci_upper":.02,
                      "n_bootstrap":1000,"valid_bootstrap_samples":1000}
            core.write_new(root/core.REPORTS/"screen.json",{"complete":True,"passed":True,"composition":"A",
                "protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":{},
                "input_manifest_sha256":core.sha256(root/core.WORK/"inputs.json"),
                "selection_sha256":core.sha256(root/core.REPORTS/"selected.json"),
                "results":{"primary":metric,"secondary":metric}})
            with patch.object(core,"load_manifest",return_value=manifest),patch.object(core,"selection",return_value="A"), \
                    patch.object(runner,"train_fold") as train:
                for action in (runner.stress,runner.finalize):
                    with self.assertRaisesRegex(ValueError,"screen gate failed"):
                        action(root)
                train.assert_not_called()
            self.assertFalse((root/core.WORK/"budget.json").exists())


@unittest.skipUnless(importlib.util.find_spec("torch"),"Neural tests use the existing model environment")
class MemberTests(unittest.TestCase):
    def preference(self,root,member,epoch=None):
        import torch
        from train_final_models import PreferenceModel
        item = core.MEMBERS[member]
        model = PreferenceModel(12,20,item["dimension"],torch.ones(12,dtype=torch.bool))
        path = root/(member+"_preference.pt")
        torch.save({"model_state":model.state_dict(),"dimension":item["dimension"],"epoch":epoch or item["preference_epoch"],
            "n_users":12,"n_items":20,"seed":item["seed"],"fine_tuning_sampling_seed":1044},path)
        return path

    def arrays(self):
        y = np.zeros((24,10),dtype=np.int8);y[::2,0] = 1
        return {"u":np.arange(24)%12,"i":np.tile(np.arange(10),(24,1)),"t":core.FOLDS["jan21"]-100+np.arange(24),
                "tb":np.arange(24)%4,"dt":np.arange(24)%2,"y":y,"indices_sha256":"synthetic"}

    def test_epoch196_and260_members_are_distinct_and_saved_models_replay(self):
        import torch
        from train_final_models import PreferenceModel,EventModel
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for member in core.COMPOSITIONS["B"]:
                path = self.preference(root,member)
                output = root/core.neural_path(member,"jan21")
                state = train_member.train(path,self.arrays(),core.FOLDS["jan21"],output,member,"cpu")
                manifest = {"preferences":{member:{"jan21":path.name}},"inputs_sha256":{path.name:core.sha256(path)}}
                saved = core.validate_new(root,manifest,member,"jan21")
                self.assertEqual(saved["dimension"],core.MEMBERS[member]["dimension"])
                self.assertEqual(saved["pair_sampling_seed"],core.MEMBERS[member]["seed"]+3000)
                model = EventModel(PreferenceModel(12,20,saved["dimension"],torch.ones(12,dtype=torch.bool)),12,20)
                model.load_state_dict(saved["model_state"],strict=True)
                for key in state["model_state"]:
                    torch.testing.assert_close(state["model_state"][key],saved["model_state"][key],rtol=0,atol=0)
            wrong = self.preference(root,"dim16_seed42",260)
            with self.assertRaisesRegex(ValueError,"epoch/dimension"):
                train_member.train(wrong,self.arrays(),core.FOLDS["jan21"],root/"bad.pt","dim16_seed42","cpu")

    def test_future_training_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = self.preference(root,"dim16_seed42")
            arrays = self.arrays();arrays["t"][0] = core.FOLDS["jan21"]
            with self.assertRaisesRegex(ValueError,"timeline"):
                train_member.train(p,arrays,core.FOLDS["jan21"],root/"bad.pt","dim16_seed42","cpu")


@unittest.skipUnless(importlib.util.find_spec("catboost") and importlib.util.find_spec("torch"),"Release integration uses the model environment")
class ReleaseTests(unittest.TestCase):
    def test_selected_two_member_release_and_inventory_tampering(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maker = MemberTests()
            manifest = {"preferences":{},"inputs_sha256":{},"code_sha256":{},"runtime":{}}
            for member in core.PAIR:
                pref = maker.preference(root,member)
                manifest["preferences"][member] = {fold:pref.name for fold in core.FOLDS}
                manifest["inputs_sha256"][pref.name] = core.sha256(pref)
                state = train_member.train(pref,maker.arrays(),core.FOLDS["jan21"],root/(member+".pt"),member,"cpu")
                for fold in core.FOLDS:
                    path = root/core.neural_path(member,fold);path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save({**state,"cutoff":core.FOLDS[fold]},path)
                    core.write_new(path.with_suffix(".json"),{"synthetic":True})
            (root/"source.py").write_text("frozen code")
            manifest["code_sha256"] = {"source.py":core.sha256(root/"source.py")}
            core.write_new(root/core.WORK/"inputs.json",{"synthetic":True})
            core.write_new(root/core.REPORTS/"preparation.json",{"complete":True})
            development_path = root/core.REPORTS/"development.json"
            core.write_new(development_path,{"complete":True,"source_sha256":manifest["code_sha256"],
                "protocol_sha256":core.canonical(core.PROTOCOL),"input_manifest_sha256":core.sha256(root/core.WORK/"inputs.json"),
                "promotion_labels_used_for_selection":False,"results":{"A":development((.01,.01)),"B":development((.02,.001))},
                "winner":"A","files_sha256":{}})
            core.write_new(root/core.REPORTS/"selected.json",{"composition":"A","members":list(core.PAIR),
                "protocol_sha256":core.canonical(core.PROTOCOL),"development_sha256":core.sha256(development_path)})
            from catboost import CatBoostClassifier,Pool
            params = core.prior.previous.prior.classifier_parameters({"boosting_type":"Plain"})
            manifest["parameters"] = {m:{case:params for case in core.CASE_FOLDS} for m in core.PAIR}
            for stage in ("screen","stress"):
                results,files = {},{}
                for case in (("primary","secondary") if stage=="screen" else ("five_day",)):
                    fits = {}
                    for member in core.PAIR:
                        path = root/core.WORK/"combiners"/f"{case}_{member}.cbm";path.parent.mkdir(parents=True,exist_ok=True)
                        model = CatBoostClassifier(**params)
                        model.fit(Pool(np.random.default_rng(42).normal(size=(240,37)),label=maker.arrays()["y"].ravel(),feature_names=list(core.BASE_COLUMNS)),verbose=False)
                        model.save_model(str(path));files[path.relative_to(root).as_posix()] = core.sha256(path)
                        neural = {core.neural_path(member,f).as_posix():core.sha256(root/core.neural_path(member,f)) for f in core.CASE_FOLDS[case]}
                        files.update(neural)
                        fits[member] = {"combiner":path.relative_to(root).as_posix(),"parameters":params,"neural_sha256":neural}
                    result = {"baseline_auc":.8,"candidate_auc":.81,"auc_diff":.01,"ci_lower":.001,"ci_upper":.02,
                              "n_bootstrap":1000,"valid_bootstrap_samples":1000,"member_models":fits}
                    results[case] = result
                if stage=="stress":
                    results = {**results["five_day"],"late_48_120h":{"baseline_auc":.8,"candidate_auc":.81,"auc_diff":.01}}
                core.write_new(root/core.REPORTS/(stage+".json"),{"complete":True,"passed":True,"composition":"A",
                    "protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":manifest["code_sha256"],
                    "input_manifest_sha256":core.sha256(root/core.WORK/"inputs.json"),"results":results,"files_sha256":files,
                    "selection_sha256":core.sha256(root/core.REPORTS/"selected.json")})
            ids,p = np.array([f"opaque-{i}" for i in range(240)]),np.full((24,10),.1)
            def export(root,path,ids,p,expected_sha=None,protected=()):
                content = np.asarray(p,dtype=float).tobytes();path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():path.write_bytes(content)
                self.assertEqual(path.read_bytes(),content)
                value = core.sha256(path);self.assertTrue(expected_sha is None or value==expected_sha)
                return value
            class Budget:
                def check(self):pass
            with patch.object(core,"load_manifest",return_value=manifest),patch.object(core,"CODE_FILES",("source.py",)), \
                    patch.object(predictor,"compute_candidate",return_value=(ids,p,{})),patch.object(predictor,"export_verified",side_effect=export):
                predictor.release_candidate(root,manifest,Budget())
                config = core.read_json(root/"configs/v8.json")
                predictor.validate_release(root,config)
                self.assertTrue(config["submission_ready"])
                self.assertEqual(set(config["models"]),set(core.PAIR))
                for field in ("files_sha256","code_sha256"):
                    bad = copy.deepcopy(config);bad[field].pop(next(iter(bad[field])))
                    with self.assertRaises(ValueError):predictor.validate_release(root,bad)
                bad = {**config,"composition":"B"}
                with self.assertRaises(ValueError):predictor.validate_release(root,bad)
