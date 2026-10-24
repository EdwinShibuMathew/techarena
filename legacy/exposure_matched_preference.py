"""Exposure-only comparison sampler and exact source adaptations; no eager torch import."""

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

RECIPE = {
    "architecture": "original_dim8_preference_and_event",
    "seed": 42, "dimension": 8,
    "comparison_distribution": "(past_candidate_displays + 20) / (total_displays + 20*n_items)",
    "display_smoothing": 20, "comparisons_per_click": 31,
    "sampling_with_replacement": True, "positive_collisions_retained": True,
    "importance_log_q_correction": False,
    "preference_initial_epochs": 196, "preference_extra_epochs": 64,
    "preference_initial_learning_rate": .012, "preference_fine_tuning_learning_rate": .003,
    "preference_regularization": .02, "preference_batch_size": 4096,
    "adam_moments_reused": True, "fine_tuning_sampling_seed": 1042,
    "event_epochs": 8, "posthoc_exposure_power": 0,
    "baseline_weight": .85, "expert_weight": .15,
}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def distribution(items, times, cutoff, n_items):
    """All displayed candidates, including no-click events; strictly prefix-only."""
    assert items.shape == (len(times), 10) and items.dtype.kind in "iu"
    fitting = items[times < cutoff]
    assert len(fitting) > 0 and ((fitting >= 0) & (fitting < n_items)).all()
    counts = np.bincount(fitting.ravel(), minlength=n_items)
    assert len(counts) == n_items and counts.sum() == len(fitting) * 10
    q = counts.astype(np.float64) + RECIPE["display_smoothing"]
    q /= q.sum()
    assert np.isfinite(q).all() and (q > 0).all()
    return counts, q


def alias_table(q):
    """O(items) construction, O(1) per draw; avoids a GPU binary search per sample."""
    q = np.asarray(q, dtype=np.float64)
    assert q.ndim == 1 and len(q) > 0 and np.isfinite(q).all() and (q > 0).all()
    np.testing.assert_allclose(q.sum(), 1, rtol=0, atol=1e-12)
    size = len(q)
    scaled = q * size
    accept, alias = np.ones(size), np.arange(size, dtype=np.int64)
    small = list(np.flatnonzero(scaled < 1))
    large = list(np.flatnonzero(scaled >= 1))
    while small and large:
        lo, hi = small.pop(), large.pop()
        accept[lo], alias[lo] = scaled[lo], hi
        scaled[hi] -= 1 - scaled[lo]
        (small if scaled[hi] < 1 else large).append(hi)
    assert ((accept >= 0) & (accept <= 1)).all()
    # Independent exact mass reconstruction, including the self/alias branches.
    mass = accept / size + np.bincount(alias, weights=(1 - accept) / size, minlength=size)
    np.testing.assert_allclose(mass, q, rtol=1e-11, atol=1e-14)
    return accept, alias


class ExposureSampler:
    def __init__(self, path, device):
        import torch
        with np.load(path, allow_pickle=False) as data:
            q = data["probabilities"]
            accept, alias = alias_table(q)
            np.testing.assert_array_equal(accept, data["alias_accept"])
            np.testing.assert_array_equal(alias, data["alias_index"])
        self.accept = torch.tensor(accept, dtype=torch.float64, device=device)
        self.alias = torch.tensor(alias, dtype=torch.long, device=device)
        self.size, self.device = len(q), device

    def sample(self, shape):
        import torch
        columns = torch.randint(self.size, shape, device=self.device)
        coins = torch.rand(shape, dtype=torch.float64, device=self.device)
        return torch.where(coins < self.accept[columns], columns, self.alias[columns])


def training_guard(script):
    """Every generated trainer is bound to its source and prefix-only inputs."""
    script = Path(script).resolve()
    trial = script.parents[2]
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["recipe"] == RECIPE
    for name, expected in protocol["prepared_sha256"].items():
        assert file_hash(trial / name) == expected, f"Changed generated source: {name}"
    for name, expected in protocol["training_input_sha256"].items():
        assert file_hash(Path(name)) == expected, f"Changed training input: {name}"
    assert str(script.relative_to(trial)) in protocol["prepared_sha256"]
    return file_hash(trial / "outputs/frozen_protocol.json")


def class_ast(text):
    return [ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, ast.ClassDef)]


def preference_source(source, fold, fine=False):
    """Keep the original SGD update, replacing only the comparison-item sampler.

    Validation loading/selection is removed entirely. Fine stage retains the
    original tested Adam resume/lower-rate/seed transition.
    """
    from prepare_preference_finetune_trial import adapt_preference, replace_once
    original = source
    if fine:
        source = adapt_preference(source, 42, fold)
    else:
        source = replace_once(source, "torch.manual_seed(43)", "torch.manual_seed(42)")
        source = replace_once(source, "split_jan26.npz", f"split_{fold}.npz")
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in main.body if isinstance(n, ast.For) and getattr(n.target, "id", "") == "epoch")
    sampler_node = next(n for n in ast.walk(loop) if isinstance(n, ast.Assign) and
                        any(getattr(t, "id", "") == "sampled_items" for t in n.targets))
    old_sampling = ast.get_source_segment(source, sampler_node.value)
    # Assert the entire original minibatch optimizer update after the sole substitution.
    update = next(n for n in loop.body if isinstance(n, ast.For))
    expected_update = ast.dump(update, include_attributes=False)
    marker = '    with np.load("data/cache/split_' + fold + '.npz") as split:'
    start, stop = source.index(marker), source.index("    active_users = torch.tensor(")
    source = source[:start] + source[stop:]
    source = source[:source.index("        # Measure validation AUC")]
    # Discard the complete validation suffix, including fine-stage monitor logs.
    source = replace_once(source, "from sklearn.metrics import roc_auc_score",
                          "from exposure_matched_preference import ExposureSampler, training_guard, file_hash, RECIPE")
    source = replace_once(source, "    start = time.perf_counter()", "    protocol_hash = training_guard(Path(__file__))\n    start = time.perf_counter()")
    source = replace_once(source, "    records = []", '    sampler = ExposureSampler("data/cache/exposure_sampling.npz", device)\n    records = []')
    source = replace_once(source, old_sampling, "sampler.sample((len(batch), negative_count))")
    filename, epoch = ("preference_best.pt", 260) if fine else ("preference_initial.pt", 196)
    source = replace_once(source, "    device = torch.device(\"cuda\")", f'    assert not Path("work/{filename}").exists(), "Use the orchestrator to resume."\n    device = torch.device("cuda")')
    source += f'''        if epoch % 40 == 0 or epoch == max_epochs:
            print(json.dumps({{"epoch": epoch, "training_loss": float(loss_total.item() / len(users))}}), flush=True)

    assert epoch == {epoch}
    assert torch.isfinite(loss_total)
    checkpoint = {{
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "n_users": n_users, "n_items": n_items, "dimension": dimension, "epoch": epoch,
        "seed": 42, "stage": "preference", "cutoff": cutoff,
        "sampler_recipe": RECIPE, "frozen_protocol_sha256": protocol_hash,
        "sampling_sha256": file_hash("data/cache/exposure_sampling.npz"),
        "fine_tuning_sampling_seed": {1042 if fine else 'None'},
        "initial_checkpoint_sha256": {('file_hash("work/preference_initial.pt")' if fine else 'None')},
    }}
    temporary = Path("work/{filename}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace("work/{filename}")
    print(json.dumps({{"stage": "preference", "epoch": epoch,
                      "runtime_seconds": round(time.perf_counter() - start, 2)}}), flush=True)


if __name__ == "__main__":
    main()
'''
    assert class_ast(source) == class_ast(original)
    new_tree = ast.parse(source)
    new_loop = next(n for n in ast.walk(new_tree) if isinstance(n, ast.For) and getattr(n.target, "id", "") == "epoch")
    new_update = next(n for n in new_loop.body if isinstance(n, ast.For))
    # Reverse only the intentional sampling expression and require AST identity.
    restored = ast.parse(ast.unparse(new_update).replace("sampler.sample((len(batch), negative_count))", old_sampling)).body[0]
    assert ast.dump(restored, include_attributes=False) == expected_update, "Preference optimizer update changed"
    assert "valid_idx" not in source and "roc_auc_score" not in source
    return source


def event_source(source):
    from prepare_preference_finetune_trial import replace_once
    original = ast.parse(source)
    loop = next(n for n in ast.walk(original) if isinstance(n, ast.For) and getattr(n.target, "id", "") == "epoch")
    expected = ast.dump(loop.body[0], include_attributes=False)
    text = replace_once(source, "from row_auc_objective import mixed_row_auc_loss",
                        "from row_auc_objective import mixed_row_auc_loss\nfrom exposure_matched_preference import training_guard, file_hash, RECIPE")
    text = replace_once(text, "    start = time.perf_counter()", "    protocol_hash = training_guard(Path(__file__))\n    start = time.perf_counter()")
    text = replace_once(text, '    n_users = checkpoint["n_users"]',
                        '    assert checkpoint["epoch"] == 260 and checkpoint["sampler_recipe"] == RECIPE\n'
                        '    assert checkpoint["frozen_protocol_sha256"] == protocol_hash\n'
                        '    assert checkpoint["sampling_sha256"] == file_hash("data/cache/exposure_sampling.npz")\n'
                        '    assert not Path("work/event_best.pt").exists(), "Use the orchestrator to resume."\n'
                        '    n_users = checkpoint["n_users"]')
    text = replace_once(text, '            "pair_sampling_seed": 3042,',
                        '            "pair_sampling_seed": 3042,\n'
                        '            "sampler_recipe": RECIPE,\n'
                        '            "frozen_protocol_sha256": protocol_hash,\n'
                        '            "preference_checkpoint_sha256": file_hash("work/preference_best.pt"),')
    for name in ["event_last.pt", "event_best.pt"]:
        text = replace_once(text, f'torch.save(state, "work/{name}")',
                            f'atomic_save(state, "work/{name}")')
    position = text.index('\nif __name__ == "__main__":')
    text = text[:position] + '''
def atomic_save(state, name):
    path = Path(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)

''' + text[position:]
    new_loop = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.For) and getattr(n.target, "id", "") == "epoch")
    assert ast.dump(new_loop.body[0], include_attributes=False) == expected
    assert class_ast(text) == class_ast(source)
    return text


def unadjusted(original, items, exposure):
    assert original.shape == items.shape and np.isfinite(original).all()
    assert ((original >= 0) & (original <= 1)).all()
    return original.copy()


def make_pipeline():
    """Private function namespace; never patch build_submission's module globals."""
    from build_submission import neural_pipeline
    scope = dict(neural_pipeline.__globals__)
    scope["adjust_exposure"] = unadjusted
    text = inspect.getsource(neural_pipeline)
    assert text.count("adjust_exposure(") == 2
    exec(compile(text, "<exposure_power_zero_pipeline>", "exec"), scope)
    assert neural_pipeline.__globals__["adjust_exposure"] is not unadjusted
    return scope["neural_pipeline"]


def numeric_self_test():
    times = np.array([10, 20, 30, 40])
    items = np.array([[0] * 10, [1] * 10, [2] * 10, [3] * 10])
    counts, q = distribution(items, times, 30, 5)
    np.testing.assert_array_equal(counts, [10, 10, 0, 0, 0])
    np.testing.assert_allclose(q, np.array([30, 30, 20, 20, 20]) / 120, rtol=0, atol=0)
    changed = items.copy()
    changed[2:] = 4
    np.testing.assert_array_equal(distribution(changed, times, 30, 5)[1], q)
    for probabilities in [q, np.ones(11) / 11, np.array([.999, .0003, .0007]), np.ones(1)]:
        alias_table(probabilities)
    original = np.linspace(.001, .09, 20).reshape(2, 10)
    result = unadjusted(original, np.zeros((2, 10), dtype=int), np.ones(5))
    np.testing.assert_array_equal(result, original)
    result[0, 0] = .5
    assert original[0, 0] == .001
    print("Numeric checks passed: strict cutoff, all candidate counts, smoothing, exact alias mass and exposure-power zero.", flush=True)
