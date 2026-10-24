"""Small time-of-day interaction extension; torch is imported only when needed."""

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

RECIPE = {
    "architecture": "additive_event_plus_time_bilinear_v1",
    "seed": 42, "dimension": 8, "time_buckets": 4, "new_parameters": 256,
    "interaction": "user_vector.T @ delta[time_bucket] @ item_vector",
    "initial_delta": 0.0, "interaction_regularization": .05,
    "day_type": "original additive effects retained; bilinear matrices pooled across day types",
    "preference_epoch": 260, "preference_reused": True,
    "event_epochs": 8, "event_learning_rate": .0015,
    "anchor_regularization": .1, "effect_regularization": .05,
    "early_stopping": False, "posthoc_exposure_power": .5,
    "updated_member": "dim8_seed42", "integration": "replace seed42 only at original weight 2/9",
}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def make_event_class(base):
    import torch
    from torch import nn

    class TimeBilinearEventModel(base):
        def __init__(self, preference, n_users, n_items):
            super().__init__(preference, n_users, n_items)
            d = preference.item.embedding_dim
            # zeros consumes no RNG: all original initialization/shuffle streams survive.
            self.time_interaction = nn.Parameter(torch.zeros(4, d, d))

        def forward(self, users, items, context):
            scores = super().forward(users, items, context)
            u = self.preference.user(users) * self.preference.active_users[users, None]
            v = self.preference.item(items)
            delta = self.time_interaction[context % 4]
            correction = torch.einsum("bd,bde,bke->bk", u, delta, v)
            return scores + correction

    return TimeBilinearEventModel


def training_guard(script):
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


def adapt_event(source):
    """Preserve original classes and update terms; add one model term and penalty."""
    from prepare_preference_finetune_trial import replace_once, classes
    original = ast.parse(source)
    loop = next(n for n in ast.walk(original) if isinstance(n, ast.For) and getattr(n.target, "id", "") == "epoch")
    expected = ast.dump(loop.body[0], include_attributes=False)
    text = replace_once(source, "from row_auc_objective import mixed_row_auc_loss",
                        "from row_auc_objective import mixed_row_auc_loss\nfrom time_bilinear_model import make_event_class, training_guard, file_hash, RECIPE")
    text = replace_once(text, "def include_no_click(scores):", "EventModel = make_event_class(EventModel)\n\n\ndef include_no_click(scores):")
    text = replace_once(text, "    start = time.perf_counter()", "    protocol_hash = training_guard(Path(__file__))\n    start = time.perf_counter()")
    text = replace_once(text, '    n_users = checkpoint["n_users"]',
                        '    assert checkpoint["epoch"] == 260 and checkpoint["dimension"] == 8\n'
                        '    assert checkpoint["fine_tuning_sampling_seed"] == 1042\n'
                        '    assert not Path("work/event_best.pt").exists(), "Use the orchestrator to resume."\n'
                        '    n_users = checkpoint["n_users"]')
    addition = "                loss += 0.05 * model.time_interaction.square().mean()\n\n"
    text = replace_once(text, "                loss.backward()", addition + "                loss.backward()")
    text = replace_once(text, '            "pair_sampling_seed": 3042,',
                        '            "pair_sampling_seed": 3042,\n'
                        '            "time_bilinear_recipe": RECIPE,\n'
                        '            "frozen_protocol_sha256": protocol_hash,\n'
                        '            "preference_checkpoint_sha256": file_hash("work/preference_best.pt"),')
    for name in ["event_last.pt", "event_best.pt"]:
        text = replace_once(text, f'torch.save(state, "work/{name}")', f'atomic_save(state, "work/{name}")')
    position = text.index('\nif __name__ == "__main__":')
    text = text[:position] + '''
def atomic_save(state, name):
    path = Path(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)

''' + text[position:]
    restored = text.replace(addition, "", 1)
    restored_loop = next(n for n in ast.walk(ast.parse(restored)) if isinstance(n, ast.For) and getattr(n.target, "id", "") == "epoch")
    assert ast.dump(restored_loop.body[0], include_attributes=False) == expected, "Original event updates changed"
    assert classes(text) == classes(source), "Original model class definitions changed"
    assert "for epoch in range(9):" in text and "if epoch != 8:" in text
    return text


def make_pipeline():
    from build_submission import neural_pipeline
    scope = dict(neural_pipeline.__globals__)
    original_class = scope["EventModel"]
    scope["EventModel"] = make_event_class(original_class)
    exec(compile(inspect.getsource(neural_pipeline), "<time_bilinear_pipeline>", "exec"), scope)
    assert neural_pipeline.__globals__["EventModel"] is original_class
    return scope["neural_pipeline"]


def numpy_interaction(user, items, delta, bucket, active):
    """Explicit-loop oracle independent of the torch einsum."""
    out = np.zeros(items.shape[:2], dtype=np.float64)
    for row in range(len(user)):
        for candidate in range(items.shape[1]):
            for left in range(user.shape[1]):
                for right in range(user.shape[1]):
                    out[row, candidate] += active[row] * user[row, left] * delta[bucket[row], left, right] * items[row, candidate, right]
    return out


def model_self_test():
    import torch
    from train_final_models import PreferenceModel, EventModel
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        torch.manual_seed(2036)
        p = PreferenceModel(5, 13, 8, torch.tensor([True, True, True, False, False]))
        torch.manual_seed(2037)
        original = EventModel(p, 5, 13).to(device).double()
        expected_rng = torch.rand(5, device=device)
        p2 = PreferenceModel(5, 13, 8, torch.ones(5, dtype=torch.bool))
        p2.load_state_dict(p.state_dict())
        torch.manual_seed(2037)
        extended = make_event_class(EventModel)(p2, 5, 13).to(device).double()
        torch.testing.assert_close(torch.rand(5, device=device), expected_rng, rtol=0, atol=0)
        users = torch.tensor([0, 1, 2, 3], device=device)
        items = torch.arange(40, device=device).reshape(4, 10) % 13
        contexts = torch.tensor([0, 1, 6, 7], device=device)
        torch.testing.assert_close(original(users, items, contexts), extended(users, items, contexts), rtol=0, atol=0)
        assert extended.time_interaction.numel() == 256
        with torch.no_grad():
            values = np.random.default_rng(2038).normal(scale=.1, size=(4, 8, 8))
            extended.time_interaction.copy_(torch.tensor(values, device=device))
        correction = extended(users, items, contexts) - original(users, items, contexts)
        u = p.user(users).detach().cpu().numpy()
        v = p.item(items).detach().cpu().numpy()
        expected = numpy_interaction(u, v, values, contexts.cpu().numpy() % 4, p.active_users[users].cpu().numpy())
        np.testing.assert_allclose(correction.detach().cpu().numpy(), expected, rtol=1e-10, atol=1e-12)
        extended.zero_grad()
        correction.sum().backward()
        analytical = extended.time_interaction.grad.detach().cpu().numpy()
        eps = 1e-5
        for index in [(0, 0, 1), (1, 4, 3), (2, 7, 5), (3, 1, 6)]:
            plus, minus = values.copy(), values.copy()
            plus[index] += eps
            minus[index] -= eps
            numerical = (numpy_interaction(u, v, plus, contexts.cpu().numpy() % 4, p.active_users[users].cpu().numpy()).sum()
                         - numpy_interaction(u, v, minus, contexts.cpu().numpy() % 4, p.active_users[users].cpu().numpy()).sum()) / (2 * eps)
            np.testing.assert_allclose(analytical[index], numerical, rtol=1e-7, atol=1e-10)
        # Same hour bucket on a weekend uses the same interaction, while the
        # original additive weekday/weekend effects remain in the base model.
        shifted = extended(users, items, (contexts + 4) % 8) - original(users, items, (contexts + 4) % 8)
        torch.testing.assert_close(shifted, correction, rtol=0, atol=1e-15)
        torch.testing.assert_close(correction[-1], torch.zeros_like(correction[-1]), rtol=0, atol=0)
        assert torch.isfinite(extended.time_interaction.grad).all()
    print("Bilinear self-test passed: zero-extension identity/RNG, 256 parameters, NumPy score oracle, finite-difference gradients, day-type pooling and inactive users on " + ", ".join(devices), flush=True)
