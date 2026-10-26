"""Required CPU model behavior, including actual fitting and serialized reload."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"legacy"))
sys.path.insert(0,str(ROOT/"tools"))
AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ("torch","catboost","sklearn"))


@unittest.skipUnless(AVAILABLE,"CPU ML dependencies required; enforced by model-contracts CI")
class ModelContractTests(unittest.TestCase):
    def test_v6_dimensions_gradients_cold_users_and_saved_reload(self):
        import torch
        from train_final_models import PreferenceModel,EventModel,include_no_click
        torch.set_num_threads(2)
        for dimension in (8,16):
            with self.subTest(dimension=dimension), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(42)
                active=torch.tensor([True,False,True])
                preference=PreferenceModel(3,12,dimension,active)
                users=torch.tensor([0,1,2]);items=torch.arange(10).repeat(3,1);context=torch.tensor([0,3,7])
                with torch.no_grad():
                    before=preference(users,items).clone();preference.user.weight[1].fill_(999)
                    torch.testing.assert_close(preference(users,items)[1],before[1],rtol=0,atol=0)
                model=EventModel(preference,3,12)
                optimizer=torch.optim.SGD(model.parameters(),lr=.01)
                scores=model(users,items,context)
                probabilities=include_no_click(scores).softmax(1)
                self.assertEqual(tuple(probabilities.shape),(3,11))
                torch.testing.assert_close(probabilities.sum(1),torch.ones(3))
                loss=torch.nn.functional.cross_entropy(include_no_click(scores),torch.tensor([1,0,9]))
                loss.backward()
                self.assertIsNotNone(preference.user.weight.grad)
                self.assertGreater(float(preference.user.weight.grad[active].abs().sum()),0)
                self.assertGreater(float(preference.item.weight.grad.abs().sum()),0)
                self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
                optimizer.step();path=Path(tmp)/"model.pt";torch.save(model.state_dict(),path)
                expected=model(users,items,context).detach()
                reloaded=EventModel(PreferenceModel(3,12,dimension,active),3,12)
                reloaded.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
                torch.testing.assert_close(reloaded(users,items,context),expected,rtol=0,atol=0)

    def test_adapter_cannot_change_expected_checkpoint_metadata(self):
        import torch
        from ml_workflow import engine
        for member in ("dim8_seed42","dim16_seed42"):
            for change in ({"cutoff":999}, {"dimension":99}, {"preference_epoch":1},
                           {"cutoff":999,"dimension":99,"preference_epoch":1}):
                with self.subTest(member=member,change=change), tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp)
                    manifest={"spec_sha256":"f"*64,"spec":{"purpose":"synthetic_validation",
                        "candidates":{"candidate":{"members":[member]}}}}
                    before=copy.deepcopy(manifest)
                    def train(fitting,metadata,path,params):
                        metadata.update(change)
                        torch.save({"metadata":metadata,"model_state":{"weight":torch.ones(1)}},path)
                    with self.assertRaisesRegex(ValueError,"Model cutoff/epoch/lineage differs"):
                        engine._model(root,root,manifest,SimpleNamespace(train=train),{"t":np.array([1])},
                                      "candidate",member,2)
                    path=root/"models/candidate"/member/"2/model.pt"
                    self.assertTrue(path.exists(),"Preserve the rejected checkpoint for diagnosis")
                    self.assertFalse(path.with_suffix(".json").exists(),"Do not attest invalid metadata")
                    self.assertEqual(manifest,before)

    def test_adapter_mutations_do_not_change_manifest_or_success_receipt(self):
        import torch
        from ml_workflow import engine
        from ml_workflow.storage import read
        for member in ("dim8_seed42","dim16_seed42"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                manifest={"spec_sha256":"f"*64,"spec":{"purpose":"synthetic_validation",
                    "candidates":{"candidate":{"members":[member],"nested":{"weights":[1.,2.]}}}}}
                before=copy.deepcopy(manifest)
                def train(fitting,metadata,path,params):
                    torch.save({"metadata":copy.deepcopy(metadata),"model_state":{"weight":torch.ones(1)}},path)
                    metadata.update(cutoff=999,dimension=99,preference_epoch=1)
                    params["members"].append("unrequested-member")
                    params["nested"]["weights"][0]=99.
                path=engine._model(root,root,manifest,SimpleNamespace(train=train),{"t":np.array([1])},
                                   "candidate",member,2)
                self.assertEqual(manifest,before)
                receipt=read(path.with_suffix(".json"))
                saved=torch.load(path,map_location="cpu",weights_only=True)
                self.assertEqual(receipt["metadata"],saved["metadata"])
                self.assertEqual(receipt["metadata"]["cutoff"],2)
                self.assertEqual(receipt["metadata"]["dimension"],engine.MEMBERS[member][0])
                self.assertEqual(receipt["metadata"]["preference_epoch"],engine.MEMBERS[member][2])
                adapter=SimpleNamespace(train=Mock(side_effect=AssertionError("Must reuse the validated checkpoint")))
                self.assertEqual(engine._model(root,root,manifest,adapter,{"t":np.array([1])},
                                              "candidate",member,2),path)
                adapter.train.assert_not_called()

    def test_explicitly_frozen_base_is_unchanged_by_head_training(self):
        import torch
        from train_final_models import PreferenceModel,EventModel,include_no_click
        torch.manual_seed(42)
        preference=PreferenceModel(3,12,16,torch.ones(3,dtype=torch.bool))
        for parameter in preference.parameters():parameter.requires_grad_(False)
        before={k:v.clone() for k,v in preference.state_dict().items()}
        model=EventModel(preference,3,12)
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
        loss=torch.nn.functional.cross_entropy(include_no_click(model(torch.tensor([0,1]),torch.arange(10).repeat(2,1),torch.tensor([0,7]))),torch.tensor([0,3]))
        loss.backward();optimizer.step()
        for name,value in preference.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in preference.parameters()))
        self.assertGreater(float(model.global_bias.detach().abs()),0)

    def test_row_loss_independent_value_and_gradient_oracle(self):
        from row_auc_objective import self_test
        self_test()

    def test_empty_history_is_finite_and_padding_does_not_contribute(self):
        import torch
        from attention_history_model import AttentionHistory
        torch.manual_seed(42)
        model=AttentionHistory(torch.randn(12,16),torch.zeros(12))
        candidates=torch.arange(10).repeat(2,1);history=torch.full((2,20),-1)
        age=torch.zeros((2,20));context=torch.tensor([0,7])
        before=model(candidates,history,age,context)
        self.assertTrue(bool(torch.isfinite(before).all()))
        with torch.no_grad():model.items.weight[12].fill_(999)
        torch.testing.assert_close(model(candidates,history,age,context),before,rtol=0,atol=0)
        before.sum().backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
