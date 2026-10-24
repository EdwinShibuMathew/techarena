"""The same 37 feature formulas used by the validated combiners."""

import numpy as np

def build_features(training, target, state, coefficients, intercept, n_users, n_items, probabilities, raw_probabilities):
    target_idx = np.arange(len(target['u']))

    def array(name):
        return state[name].detach().numpy()
    item_vectors = array('preference.item.weight')
    user_vectors = array('preference.user.weight')
    active_users = array('preference.active_users')
    item_bias = array('preference.item_bias.weight')[:, 0]
    user_bias = array('user_bias.weight')[:, 0]
    item_context = array('item_context.weight')
    user_context = array('user_context.weight')
    fit_users = training['u']
    fit_items = training['i']
    fit_labels = training['y']
    user_events = np.bincount(fit_users, minlength=n_users)
    user_clicks = np.bincount(fit_users, weights=fit_labels.sum(axis=1), minlength=n_users)
    item_displays = np.bincount(fit_items.ravel(), minlength=n_items)
    item_clicks = np.bincount(fit_items.ravel(), weights=fit_labels.ravel(), minlength=n_items)
    global_ctr = float(fit_labels.mean())
    item_ctr = (item_clicks + 50 * global_ctr) / (item_displays + 50)
    user_ctr = (user_clicks + 50 * global_ctr) / (10 * user_events + 50)
    chunks = []
    feature_names = None
    for start in range(0, len(target_idx), 4096):
        q = target_idx[start:start + 4096]
        users = target['u'][q]
        items = target['i'][q]
        h = target['h'][q]
        present = h >= 0
        length = present.sum(axis=1)
        context = target['tb'][q] + 4 * target['dt'][q]
        age = np.where(present, (target['t'][q, None] - target['ht'][q]) / 3600, 1000000)
        candidate_vectors = item_vectors[items]
        history_vectors = item_vectors[h.clip(min=0)] * present[:, :, None]
        history_mean = history_vectors.sum(axis=1) / np.maximum(length[:, None], 1)
        inferred_users = history_mean @ coefficients.T + intercept
        scoring_users = user_vectors[users].copy()
        scoring_users *= active_users[users, None]
        cold_with_history = (user_events[users] == 0) & (length > 0)
        scoring_users[cold_with_history] = inferred_users[cold_with_history]
        similarity = np.einsum('bkd,bhd->bkh', candidate_vectors, history_vectors, optimize=True)
        matches = items[:, :, None] == h[:, None, :]
        weights = np.exp(-age / 24) * present

        def broadcast(values):
            return np.broadcast_to(values[:, None], items.shape)

        def masked_max(values, mask):
            maximum = np.where(mask, values, -np.inf).max(axis=2)
            return np.where(mask.any(axis=2), maximum, 0)
        history_max = masked_max(similarity, present[:, None, :])
        nonrepeat_max = masked_max(similarity, present[:, None, :] & ~matches)
        candidate_norm = np.linalg.norm(candidate_vectors, axis=2)
        history_norm = np.linalg.norm(history_vectors, axis=2)
        cosine = similarity / np.maximum(candidate_norm[:, :, None] * history_norm[:, None, :], 1e-08)
        last_position = np.maximum(length - 1, 0)
        last_vector = history_vectors[np.arange(len(q)), last_position]
        repeat_age = np.where(matches, age[:, None, :], 1000000).min(axis=2)
        p = probabilities[start:start + len(q)]
        raw_p = raw_probabilities[start:start + len(q)]
        features = {'pipeline_probability': p, 'raw_event_probability': raw_p, 'log_probability': np.log(np.maximum(p, 1e-08)), 'event_click_probability': broadcast(p.sum(axis=1)), 'event_max_probability': broadcast(p.max(axis=1)), 'relative_probability': p - p.mean(axis=1, keepdims=True), 'item_ctr': item_ctr[items], 'user_ctr': broadcast(user_ctr[users]), 'log_item_displays': np.log1p(item_displays[items]), 'log_user_events': broadcast(np.log1p(user_events[users])), 'time_bucket': broadcast(target['tb'][q]), 'day_type': broadcast(target['dt'][q]), 'history_length': broadcast(length), 'history_unique_items': broadcast(np.array([len(np.unique(row[row >= 0])) for row in h])), 'log_last_click_age': broadcast(np.log1p(np.minimum(age.min(axis=1), 720))), 'clicks_last_6h': broadcast((present & (age < 6)).sum(axis=1)), 'clicks_last_24h': broadcast((present & (age < 24)).sum(axis=1)), 'clicks_last_72h': broadcast((present & (age < 72)).sum(axis=1)), 'repeat_count': matches.sum(axis=2), 'repeat_recency': (matches * weights[:, None, :]).sum(axis=2), 'log_last_repeat_age': np.log1p(np.minimum(repeat_age, 720)), 'history_dot_mean': similarity.sum(axis=2) / np.maximum(length[:, None], 1), 'history_dot_max': history_max, 'history_dot_recent': (similarity * weights[:, None, :]).sum(axis=2) / np.maximum(weights.sum(axis=1)[:, None], 1e-08), 'history_dot_last': np.einsum('bkd,bd->bk', candidate_vectors, last_vector), 'history_dot_max_nonrepeat': nonrepeat_max, 'history_cosine_mean': cosine.sum(axis=2) / np.maximum(length[:, None], 1), 'history_cosine_max': masked_max(cosine, present[:, None, :]), 'encoded_history_affinity': np.einsum('bkd,bd->bk', candidate_vectors, inferred_users), 'user_affinity': np.einsum('bkd,bd->bk', candidate_vectors, scoring_users), 'item_vector_norm': candidate_norm, 'user_vector_norm': broadcast(np.linalg.norm(scoring_users, axis=1)), 'item_bias': item_bias[items], 'user_bias': broadcast(user_bias[users]), 'unseen_user': broadcast(user_events[users] == 0), 'item_time_effect': item_context[items, context[:, None]], 'user_time_effect': broadcast(user_context[users, context])}
        names = list(features)
        if feature_names is None:
            feature_names = names
        assert names == feature_names
        matrix = np.stack(list(features.values()), axis=2)
        matrix = matrix.reshape(-1, len(names)).astype('float32')
        assert np.isfinite(matrix).all()
        chunks.append(matrix)
    X = np.concatenate(chunks)
    return (X, feature_names)
