"""Combine the four validated neural feature sets with common columns once."""

import numpy as np

MODEL_KEYS = ("dim8_seed42", "dim8_seed43", "dim16_seed42", "dim16_seed43")
COMMON_FEATURES = frozenset({
    "item_ctr", "user_ctr", "log_item_displays", "log_user_events",
    "time_bucket", "day_type", "history_length", "history_unique_items",
    "log_last_click_age", "clicks_last_6h", "clicks_last_24h", "clicks_last_72h",
    "repeat_count", "repeat_recency", "log_last_repeat_age", "unseen_user",
})


def combine_matrices(matrices, names_by_model):
    assert set(matrices) == set(names_by_model) == set(MODEL_KEYS)
    reference_names = list(names_by_model[MODEL_KEYS[0]])
    assert len(reference_names) == 37 and len(set(reference_names)) == 37
    assert COMMON_FEATURES.issubset(reference_names)
    common = [index for index, name in enumerate(reference_names) if name in COMMON_FEATURES]
    specific = [index for index, name in enumerate(reference_names) if name not in COMMON_FEATURES]
    assert len(common) == 16 and len(specific) == 21
    first = matrices[MODEL_KEYS[0]]
    assert first.ndim == 2 and first.shape[1] == 37
    chunks = [first[:, common]]
    names = [reference_names[index] for index in common]

    for key in MODEL_KEYS:
        matrix = matrices[key]
        assert list(names_by_model[key]) == reference_names
        assert matrix.shape == first.shape
        assert np.isfinite(matrix).all()
        assert np.array_equal(matrix[:, common], first[:, common]), f"Common features differ for {key}"
        chunks.append(matrix[:, specific])
        names.extend(f"{key}__{reference_names[index]}" for index in specific)
    X = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    assert X.shape == (len(first), 100)
    assert len(names) == len(set(names)) == 100
    return X, names


def build_joint_features(training, target, sources, pipeline8, pipeline16, device):
    matrices, names = {}, {}
    for key in MODEL_KEYS:
        print(f"Building joint inputs: {key}", flush=True)
        source = sources[key]
        pipeline = pipeline8 if key.startswith("dim8_") else pipeline16
        X, feature_names, _, _, _, _ = pipeline(
            source["checkpoint"], training, target, source["positive_path"], device,
            final_seed=source.get("final_seed"),
        )
        matrices[key], names[key] = X, feature_names
    return combine_matrices(matrices, names)
