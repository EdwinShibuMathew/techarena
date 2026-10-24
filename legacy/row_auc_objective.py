"""Mixed event likelihood and global row-pair logistic ranking objective.

Pairs come from labeled rows in the current fitting minibatch, including other
users and no-click events. This is a sampled surrogate, not exact AUC or BPR.
Inference remains the original eleven-outcome softmax.
"""

import numpy as np

CONFIG = {"objective": "half_event_ce_half_global_row_pair_v1",
          "event_ce_weight": .5, "pairwise_weight": .5,
          "sampled_negative_rows_per_positive": 16,
          "pair_score": "log odds of original softmax candidate marginal",
          "negative_pool": "all unclicked candidate rows in current fitting minibatch, across users",
          "pair_sampling_seed_offset": 3000, "event_epochs": 8,
          "event_learning_rate": .0015, "anchor_regularization": .1,
          "effect_regularization": .05, "early_stopping": False}


def candidate_log_odds(scores):
    import torch
    assert scores.ndim == 2 and scores.shape[1] == 10
    all_logits = torch.cat([torch.zeros_like(scores[:, :1]), scores], dim=1)
    # Compute the sum over the OTHER outcomes directly. Subtracting softmax
    # probabilities from one would be unstable for probabilities close to one.
    mask = torch.arange(11, device=scores.device)[None, :] == torch.arange(1, 11, device=scores.device)[:, None]
    other_logits = all_logits[:, None, :].expand(-1, 10, -1).masked_fill(mask[None, :, :], -torch.inf)
    return scores - other_logits.logsumexp(dim=2)


def mixed_row_auc_loss(scores, targets, generator):
    import torch
    from torch.nn import functional as F
    all_logits = torch.cat([torch.zeros_like(scores[:, :1]), scores], dim=1)
    event_loss = F.cross_entropy(all_logits, targets)
    labels = (targets[:, None] == torch.arange(1, 11, device=scores.device)[None, :]).ravel()
    odds = candidate_log_odds(scores).ravel()
    positives, negatives = odds[labels], odds[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        return event_loss
    sampled = torch.randint(negatives.numel(),
                            (positives.numel(), CONFIG["sampled_negative_rows_per_positive"]),
                            device=scores.device, generator=generator)
    pair_loss = F.softplus(negatives[sampled] - positives[:, None]).mean()
    return CONFIG["event_ce_weight"] * event_loss + CONFIG["pairwise_weight"] * pair_loss


def numpy_oracle(scores, targets, samples):
    """Independent tiny implementation: explicit sums over other outcomes."""
    logits = np.column_stack([np.zeros(len(scores)), scores])
    normalizers = np.logaddexp.reduce(logits, axis=1)
    event_loss = np.mean(normalizers - logits[np.arange(len(scores)), targets])
    odds = np.empty_like(scores)
    for event in range(len(scores)):
        for item in range(10):
            others = np.delete(logits[event], item + 1)
            odds[event, item] = scores[event, item] - np.logaddexp.reduce(others)
    labels = (targets[:, None] == np.arange(1, 11)[None, :]).ravel()
    positive, negative = odds.ravel()[labels], odds.ravel()[~labels]
    if not len(positive):
        return event_loss, odds
    pair_loss = np.logaddexp(0, negative[samples] - positive[:, None]).mean()
    return .5 * event_loss + .5 * pair_loss, odds


def self_test():
    import torch
    from torch.nn import functional as F
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    base = np.random.default_rng(2031).normal(size=(4, 10))
    targets_np = np.array([1, 0, 8, 0])
    for device in devices:
        targets = torch.tensor(targets_np, dtype=torch.long, device=device)
        scores = torch.tensor(base, dtype=torch.float64, device=device, requires_grad=True)
        seed = 3042
        draw = torch.Generator(device=device).manual_seed(seed)
        samples = torch.randint(38, (2, 16), device=device, generator=draw).cpu().numpy()
        expected, odds = numpy_oracle(base, targets_np, samples)
        actual = mixed_row_auc_loss(scores, targets, torch.Generator(device=device).manual_seed(seed))
        np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(candidate_log_odds(scores).detach().cpu().numpy(), odds, rtol=1e-12, atol=1e-12)
        actual.backward()
        numerical = np.empty_like(base)
        epsilon = 1e-5
        for q in np.ndindex(base.shape):
            plus, minus = base.copy(), base.copy()
            plus[q] += epsilon
            minus[q] -= epsilon
            numerical[q] = (numpy_oracle(plus, targets_np, samples)[0] - numpy_oracle(minus, targets_np, samples)[0]) / (2 * epsilon)
        np.testing.assert_allclose(scores.grad.cpu().numpy(), numerical, rtol=1e-5, atol=1e-8)
        # Includes saturated probabilities, all-no-click batches, finite gradients
        # and a dedicated sampler that does not consume the shuffle RNG stream.
        extreme_np = np.where(np.arange(40).reshape(4, 10) % 2, 1000., -1000.)
        extreme_np[0] = -1000
        extreme_np[0, 0] = 1000
        extreme_np[1] = -1000
        extreme = torch.tensor(extreme_np,
                               dtype=torch.float32, device=device, requires_grad=True)
        loss = mixed_row_auc_loss(extreme, targets, torch.Generator(device=device).manual_seed(seed))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
        no_click = torch.zeros(4, dtype=torch.long, device=device)
        expected_ce = F.cross_entropy(torch.cat([torch.zeros_like(scores[:, :1]), scores], 1), no_click)
        torch.testing.assert_close(mixed_row_auc_loss(scores, no_click, draw), expected_ce)
        torch.manual_seed(77)
        expected_rng = torch.rand(7, device=device)
        torch.manual_seed(77)
        mixed_row_auc_loss(scores, targets, torch.Generator(device=device).manual_seed(seed))
        torch.testing.assert_close(torch.rand(7, device=device), expected_rng, rtol=0, atol=0)
    print("Row-ranking loss self-test passed: independent loss oracle, finite-difference gradients, extreme scores, no-click batches and separate sampling RNG on " + ", ".join(devices), flush=True)
