"""Required CPU model behavior, including actual fitting and serialized reload."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"legacy"))
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
                loss.backward();self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
                optimizer.step();path=Path(tmp)/"model.pt";torch.save(model.state_dict(),path)
                expected=model(users,items,context).detach()
                reloaded=EventModel(PreferenceModel(3,12,dimension,active),3,12)
                reloaded.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
                torch.testing.assert_close(reloaded(users,items,context),expected,rtol=0,atol=0)

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
