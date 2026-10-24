"""Read-only extension of the original 37-feature pipeline for repeated users.

Only cold-user prediction changes: infer one vector per event and never write
those vectors into a shared user table. All feature formulas remain original.
"""

import ast
import inspect
import tempfile
from pathlib import Path

import numpy as np


def user_equal_weights(users):
    unique, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    assert len(unique) > 0
    # Equal summed event weight per user; mean weight one for explicit scaling.
    event_weight = (len(users) / len(unique)) / counts[inverse]
    totals = np.bincount(inverse, weights=event_weight)
    np.testing.assert_allclose(totals, len(users) / len(unique), rtol=1e-12, atol=1e-12)
    assert abs(event_weight.mean() - 1) < 1e-12
    return np.repeat(event_weight, 10), {
        "events": len(users), "users": len(unique), "candidate_rows": 10 * len(users),
        "mean_row_weight": float(event_weight.mean()),
        "total_row_weight_per_user": float(10 * len(users) / len(unique)),
        "minimum_event_weight": float(event_weight.min()), "maximum_event_weight": float(event_weight.max()),
        "equal_total_weight_per_user_verified": True,
    }


def adapt_source(source):
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, f"Original pipeline changed: {old[:100]}"
        source = source.replace(old, new, 1)
    replace('assert dimension == 8 and checkpoint["epoch"] == 8',
            'assert dimension in (8, 16) and checkpoint["epoch"] == 8')
    replace('        assert len(np.unique(target["u"][cold])) == int(cold.sum())\n', '')
    replace('''        with torch.inference_mode():
            cold_users = torch.tensor(target["u"][cold], dtype=torch.long, device=device)
            model.preference.user.weight[cold_users] = torch.tensor(
                estimated_vectors, dtype=torch.float32, device=device
            )
            model.preference.active_users[cold_users] = True
''', '')
    replace('cold_probabilities = predict_events(model, cold_target, device)',
            'cold_probabilities = predict_cold_profiles(model, cold_target, estimated_vectors, device)')
    ast.parse(source)
    return source


def predict_cold_profiles(model, target, profiles, device):
    import torch
    from train_final_models import include_no_click
    assert profiles.shape == (len(target["u"]), model.preference.item.weight.shape[1])
    result = np.empty(target["i"].shape, dtype=np.float64)
    with torch.inference_mode():
        for begin in range(0, len(profiles), 4096):
            end = begin + 4096
            u, i, context = [torch.tensor(v, dtype=torch.long, device=device) for v in
                             [target["u"][begin:end], target["i"][begin:end],
                              target["tb"][begin:end] + 4 * target["dt"][begin:end]]]
            vector = torch.tensor(profiles[begin:end], dtype=torch.float32, device=device)
            # Same operations and addition order as PreferenceModel/EventModel.
            scores = (vector[:, None, :] * model.preference.item(i)).sum(-1)
            scores = scores + model.preference.item_bias(i).squeeze(-1)
            item_effect = model.item_context(i).gather(2, context[:, None, None].expand(-1, i.shape[1], 1)).squeeze(-1)
            user_effect = model.user_context(u).gather(1, context[:, None])
            scores = scores + model.user_bias(u) + model.global_bias + item_effect + user_effect
            result[begin:end] = include_no_click(scores).softmax(1)[:, 1:].cpu().numpy()
    return result


def make_pipeline():
    import build_submission as original
    namespace = dict(vars(original))
    namespace["predict_cold_profiles"] = predict_cold_profiles
    exec(compile(adapt_source(inspect.getsource(original.neural_pipeline)),
                 "<expanded-meta-read-only-pipeline>", "exec"), namespace)
    return namespace["neural_pipeline"]


def self_test():
    import torch
    import build_submission as original
    from train_final_models import PreferenceModel, EventModel
    from submission_export import sha256
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    device = torch.device("cuda")
    pipeline = make_pipeline()
    rng = np.random.default_rng(2026)
    users = np.array([1, 1, 2, 3, 3, 3])
    weights, report = user_equal_weights(users)
    np.testing.assert_allclose(np.bincount(np.repeat(users, 10), weights=weights)[1:], [20, 20, 20])
    assert report["equal_total_weight_per_user_verified"]
    with tempfile.TemporaryDirectory(prefix="expanded_meta_selftest_", dir=Path(__file__).resolve().parents[1]) as temporary:
        folder = Path(temporary)
        positive_path = folder / "positive.npz"
        np.savez(positive_path, users=np.array([0, 1, 2, 0, 1, 2]), items=np.array([1, 2, 3, 4, 5, 6]))
        training = {"u": np.array([0, 1, 2, 0, 1, 2]), "i": rng.integers(0, 30, (6, 10)),
                    "y": np.eye(10, dtype=np.int8)[[0, 1, 2, 3, 4, 5]]}
        target = {"u": np.array([0, 3, 3, 3, 4, 1]), "i": rng.integers(0, 30, (6, 10)),
                  "t": np.full(6, 1737763200, dtype=np.int64), "tb": np.array([0, 1, 2, 3, 0, 1], dtype=np.int8),
                  "dt": np.array([0, 0, 1, 1, 0, 1], dtype=np.int8),
                  "h": np.full((6, 20), -1, dtype=np.int32), "ht": np.zeros((6, 20), dtype=np.int64)}
        for row, history in [(0, [1, 2]), (1, [1, 2]), (2, [3, 5, 7]), (4, [4]), (5, [8, 9])]:
            target["h"][row, :len(history)] = history
            target["ht"][row, :len(history)] = target["t"][row] - 3600 * np.arange(len(history), 0, -1)
        for dimension in [8, 16]:
            torch.manual_seed(2026)
            model = EventModel(PreferenceModel(8, 30, dimension, torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)), 8, 30)
            with torch.no_grad():
                for parameter in [model.user_bias.weight, model.user_context.weight, model.item_context.weight]:
                    parameter.normal_(0, 0.1)
                model.global_bias.fill_(-1.0)
            checkpoint = folder / f"event{dimension}.pt"
            torch.save({"model_state": model.state_dict(), "n_users": 8, "n_items": 30,
                        "dimension": dimension, "epoch": 8}, checkpoint)
            before = sha256(checkpoint)
            X, names, raw, p, *_ = pipeline(checkpoint, training, target, positive_path, device)
            # Original unique-user scoring is the reference, one event at a time.
            namespace = dict(vars(original))
            text = inspect.getsource(original.neural_pipeline).replace(
                'assert dimension == 8 and checkpoint["epoch"] == 8',
                f'assert dimension == {dimension} and checkpoint["epoch"] == 8')
            exec(compile(text, "<original-unique-user-reference>", "exec"), namespace)
            for row in range(6):
                subset = {key: value[row:row + 1] for key, value in target.items()}
                expected, cols, raw_expected, p_expected, *_ = namespace["neural_pipeline"](checkpoint, training, subset, positive_path, device)
                assert names == cols and len(names) == 37
                np.testing.assert_allclose(X[row * 10:(row + 1) * 10], expected, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(raw[row:row + 1], raw_expected, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(p[row:row + 1], p_expected, rtol=2e-5, atol=2e-6)
            permutation = np.array([3, 2, 5, 0, 4, 1])
            shuffled, cols, _, sp, *_ = pipeline(checkpoint, training, {k: v[permutation] for k, v in target.items()}, positive_path, device)
            np.testing.assert_allclose(shuffled.reshape(6, 10, 37), X.reshape(6, 10, 37)[permutation], rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(sp, p[permutation], rtol=2e-5, atol=2e-6)
            assert sha256(checkpoint) == before
            torch.cuda.empty_cache()
    print("Expanded-feature self-test passed: equal user weights; original 37 features for dim8/dim16; repeated cold-user histories; empty histories; order invariance; unchanged checkpoints.", flush=True)
