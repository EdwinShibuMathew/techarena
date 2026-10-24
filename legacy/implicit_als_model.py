"""Compact implicit-feedback ALS; never builds a users-by-items dense matrix.

Confidence-weighted squared error, followed separately by the original event
training stage. This implementation uses chunked normal-equation accumulation.
"""

import time

import numpy as np

CONFIG = {
    "algorithm": "implicit_als_v1", "dimension": 16, "iterations": 30,
    "confidence_alpha": 40.0, "confidence_rule": "1 + alpha * log1p(click_count)",
    "regularization": 0.1, "seed": 2028, "initial_item_std": 0.01,
    "edge_chunk_size": 32768, "score_multiplier_before_event_training": 8.0,
    "event_epochs": 8, "event_learning_rate": 0.0015,
    "early_stopping": False,
}


def aggregate_clicks(users, items, n_users, n_items):
    assert users.shape == items.shape and users.ndim == 1 and len(users)
    assert ((users >= 0) & (users < n_users)).all()
    assert ((items >= 0) & (items < n_items)).all()
    keys, counts = np.unique(users.astype(np.int64) * n_items + items, return_counts=True)
    row = keys // n_items
    col = keys % n_items
    confidence = 1 + CONFIG["confidence_alpha"] * np.log1p(counts)
    return row, col, confidence.astype(np.float32), counts


def solve_side(fixed, rows, columns, confidence, n_rows, regularization, chunk_size):
    import torch
    d = fixed.shape[1]
    gram = fixed.T @ fixed + regularization * torch.eye(d, dtype=fixed.dtype, device=fixed.device)
    A = gram.expand(n_rows, d, d).clone()
    b = torch.zeros(n_rows, d, dtype=fixed.dtype, device=fixed.device)
    for begin in range(0, len(rows), chunk_size):
        end = begin + chunk_size
        r, c, weight = rows[begin:end], columns[begin:end], confidence[begin:end]
        v = fixed[c]
        outer = v[:, :, None] * v[:, None, :]
        A.index_add_(0, r, (weight - 1)[:, None, None] * outer)
        b.index_add_(0, r, weight[:, None] * v)
    factors, info = torch.linalg.solve_ex(A, b[:, :, None])
    assert (info == 0).all(), "ALS normal equations could not be solved"
    result = factors.squeeze(-1)
    assert torch.isfinite(result).all(), "Nonfinite ALS factors"
    return result


def objective(users, items, row, col, confidence, regularization, chunk_size):
    import torch
    # Sum over all unobserved pairs algebraically through the two small Grams.
    value = ((users.T @ users) * (items.T @ items)).sum()
    value = value + regularization * (users.square().sum() + items.square().sum())
    for begin in range(0, len(row), chunk_size):
        end = begin + chunk_size
        score = (users[row[begin:end]] * items[col[begin:end]]).sum(1)
        value = value + (confidence[begin:end] * (1 - score).square() - score.square()).sum()
    return float(value)


def fit_factors(positive_users, positive_items, n_users, n_items, device, config=None, progress=True):
    import torch
    cfg = CONFIG if config is None else config
    row, col, confidence, counts = aggregate_clicks(positive_users, positive_items, n_users, n_items)
    # aggregate_clicks uses the frozen production alpha; self-test changes only
    # dimension, iterations and chunk size.
    torch.manual_seed(cfg["seed"])
    r = torch.tensor(row, dtype=torch.long, device=device)
    c = torch.tensor(col, dtype=torch.long, device=device)
    weight = torch.tensor(confidence, dtype=torch.float32, device=device)
    u = torch.zeros(n_users, cfg["dimension"], device=device)
    v = torch.randn(n_items, cfg["dimension"], device=device) * cfg["initial_item_std"]
    active_items = torch.bincount(c, minlength=n_items) > 0
    v[~active_items] = 0
    start = time.perf_counter()
    records = []
    with torch.inference_mode():
        before = objective(u, v, r, c, weight, cfg["regularization"], cfg["edge_chunk_size"])
        for iteration in range(1, cfg["iterations"] + 1):
            u = solve_side(v, r, c, weight, n_users, cfg["regularization"], cfg["edge_chunk_size"])
            v = solve_side(u, c, r, weight, n_items, cfg["regularization"], cfg["edge_chunk_size"])
            loss = objective(u, v, r, c, weight, cfg["regularization"], cfg["edge_chunk_size"])
            assert np.isfinite(loss) and loss >= -1e-3
            assert loss <= before * 1.001 + 1e-3, "ALS objective increased unexpectedly"
            records.append({"iteration": iteration, "weighted_objective": loss,
                            "elapsed_seconds": round(time.perf_counter() - start, 1)})
            before = loss
            if progress and (iteration == 1 or iteration % 5 == 0):
                print(records[-1], flush=True)
    return u.cpu(), v.cpu(), records, {"click_examples": len(positive_users), "unique_user_item_pairs": len(row),
                                      "maximum_pair_click_count": int(counts.max())}


def train_preference(positive_path, cutoff, destination, device):
    import torch
    from train_final_models import PreferenceModel
    from submission_export import sha256
    assert not destination.exists()
    with np.load(positive_path) as saved:
        assert int(saved["cutoff"]) == cutoff and (saved["timestamps"] < cutoff).all()
        pu, pi = saved["users"], saved["items"]
        n_users, n_items = int(saved["n_users"]), int(saved["n_items"])
    u, v, records, sizes = fit_factors(pu, pi, n_users, n_items, device)
    active = torch.tensor(np.bincount(pu, minlength=n_users) > 0)
    model = PreferenceModel(n_users, n_items, CONFIG["dimension"], active)
    scale = float(np.sqrt(CONFIG["score_multiplier_before_event_training"]))
    with torch.no_grad():
        model.user.weight.copy_(u * scale)
        model.item.weight.copy_(v * scale)
        model.item_bias.weight.zero_()
    result = {"model_state": model.state_dict(), "optimizer_state": {}, "dimension": CONFIG["dimension"],
              "n_users": n_users, "n_items": n_items, "seed": CONFIG["seed"], "cutoff": cutoff,
              "stage": "preference", "epoch": CONFIG["iterations"], "algorithm": CONFIG["algorithm"],
              "config": CONFIG.copy(), "positive_cache_sha256": sha256(positive_path), "records": records, **sizes}
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(result, temporary)
    temporary.replace(destination)
    del model, u, v
    torch.cuda.empty_cache()
    return {"cutoff": cutoff, **sizes, "last_iteration": records[-1]}


def self_test():
    import torch
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    # Check the sparse GPU normal equations against a tiny dense CPU reference.
    pu, pi = np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 2, 1, 2, 3])
    row, col, confidence, counts = aggregate_clicks(pu, pi, 4, 5)
    assert counts.sum() == len(pu) and counts.max() == 2
    rng = np.random.default_rng(2028)
    fixed = rng.normal(size=(5, 3))
    target = np.zeros((4, 5))
    C = np.ones((4, 5))
    target[row, col], C[row, col] = 1, confidence
    expected = np.stack([np.linalg.solve(fixed.T @ (C[i, :, None] * fixed) + .1 * np.eye(3),
                                        fixed.T @ (C[i] * target[i])) for i in range(4)])
    actual = solve_side(torch.tensor(fixed, dtype=torch.float64, device="cuda"),
                        torch.tensor(row, device="cuda"), torch.tensor(col, device="cuda"),
                        torch.tensor(confidence, dtype=torch.float64, device="cuda"), 4, .1, 2).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(actual[3], np.zeros(3))
    u = rng.normal(size=(4, 3))
    dense_loss = (C * (target - u @ fixed.T) ** 2).sum() + .1 * ((u * u).sum() + (fixed * fixed).sum())
    sparse_loss = objective(torch.tensor(u, device="cuda"), torch.tensor(fixed, device="cuda"),
                            torch.tensor(row, device="cuda"), torch.tensor(col, device="cuda"),
                            torch.tensor(confidence, dtype=torch.float64, device="cuda"), .1, 2)
    np.testing.assert_allclose(sparse_loss, dense_loss, rtol=1e-10)
    cfg = dict(CONFIG, dimension=3, iterations=4, edge_chunk_size=2)
    uu, vv, history, _ = fit_factors(pu, pi, 4, 5, torch.device("cuda"), cfg, progress=False)
    assert len(history) == 4 and history[-1]["weighted_objective"] <= history[0]["weighted_objective"]
    assert torch.count_nonzero(uu[3]) == 0 and torch.count_nonzero(vv[4]) == 0
    print("ALS GPU self-test passed: dense-reference solves, duplicate-click confidence, full objective, inactive factors and alternating descent.", flush=True)
