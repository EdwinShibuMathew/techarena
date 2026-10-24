"""Sixteen additional features from labeled impressions strictly before cutoff.

NumPy only. No aggregate is updated from targets or target histories. Non-click
counts describe observed outcomes, not a claim that an ignored item is disliked.
Time-of-day aggregates use four six-hour buckets, pooled across day types.
"""

import numpy as np


SMOOTHING = 50.0
RECENT_HOURS = 24
MAX_AGE_HOURS = 720.0
NAMES = [
    "log_ui_prior_displays", "log_ui_prior_nonclicks", "log_ui_prior_clicks",
    "log_ui_last_display_age", "log_user_last_event_age", "log_user_last_observed_click_age",
    "log_user_recent24h_events", "user_recent24h_ctr",
    "log_item_recent24h_displays", "item_recent24h_ctr",
    "log_user_bucket_events", "user_bucket_ctr",
    "log_item_bucket_displays", "item_bucket_ctr",
    "log_user_bucket_last_event_age", "log_item_last_display_age",
]


def grouped(keys, labels, times, size):
    counts = np.bincount(keys, minlength=size)
    clicks = np.bincount(keys, weights=labels, minlength=size)
    last = np.full(size, -1, dtype=np.int64)
    np.maximum.at(last, keys, times)
    return counts, clicks, last


def age_feature(times, last):
    assert np.all((last < 0) | (last < times)), "Fitting observation is not strictly before target"
    age = np.where(last >= 0, (times - last) / 3600, MAX_AGE_HOURS)
    return np.log1p(np.minimum(age, MAX_AGE_HOURS))


class ExposureFeatures:
    def __init__(self, raw, cutoff):
        self.cutoff = cutoff
        fit = np.flatnonzero(raw["t"] < cutoff)
        assert len(fit) > 0
        # Cardinalities only allocate tables. Future labels/outcomes are never used.
        self.n_users, self.n_items = int(raw["u"].max()) + 1, int(raw["i"].max()) + 1
        u = raw["u"][fit].astype(np.int64)
        i = raw["i"][fit].astype(np.int64)
        t = raw["t"][fit].astype(np.int64)
        y = raw["y"][fit]
        tb = raw["tb"][fit].astype(np.int64)
        assert i.shape == y.shape == (len(u), 10)
        assert (u >= 0).all() and (i >= 0).all() and np.isin(y, [0, 1]).all()
        assert (y.sum(1) <= 1).all() and ((tb >= 0) & (tb < 4)).all()
        assert t.min() >= 0 and (t < cutoff).all()
        global_ctr = float(y.mean())
        user_n, user_y, self.user_last = grouped(u, y.sum(1), t, self.n_users)
        item_n, item_y, self.item_last = grouped(i.ravel(), y.ravel(), np.repeat(t, 10), self.n_items)
        self.user_prior = (user_y + SMOOTHING * global_ctr) / (10 * user_n + SMOOTHING)
        self.item_prior = (item_y + SMOOTHING * global_ctr) / (item_n + SMOOTHING)
        clicked = y.sum(1) > 0
        self.user_click_last = np.full(self.n_users, -1, dtype=np.int64)
        np.maximum.at(self.user_click_last, u[clicked], t[clicked])

        recent = t >= cutoff - RECENT_HOURS * 3600
        self.recent_user_n, recent_user_y, _ = grouped(u[recent], y[recent].sum(1), t[recent], self.n_users)
        recent_item_n, recent_item_y, _ = grouped(i[recent].ravel(), y[recent].ravel(), np.repeat(t[recent], 10), self.n_items)
        self.recent_item_n = recent_item_n
        self.recent_user_ctr = ((recent_user_y + SMOOTHING * self.user_prior)
                                / (10 * self.recent_user_n + SMOOTHING))
        self.recent_item_ctr = ((recent_item_y + SMOOTHING * self.item_prior)
                                / (recent_item_n + SMOOTHING))

        ub = u * 4 + tb
        ib = (i * 4 + tb[:, None]).ravel()
        self.bucket_user_n, bucket_user_y, self.bucket_user_last = grouped(ub, y.sum(1), t, self.n_users * 4)
        self.bucket_item_n, bucket_item_y, _ = grouped(ib, y.ravel(), np.repeat(t, 10), self.n_items * 4)
        self.bucket_user_ctr = ((bucket_user_y + SMOOTHING * np.repeat(self.user_prior, 4))
                                / (10 * self.bucket_user_n + SMOOTHING))
        self.bucket_item_ctr = ((bucket_item_y + SMOOTHING * np.repeat(self.item_prior, 4))
                                / (self.bucket_item_n + SMOOTHING))

        # Compact sparse lookup, never a dense users-by-items allocation.
        keys = (u[:, None] * self.n_items + i).ravel()
        self.ui_keys, inverse, self.ui_n = np.unique(keys, return_inverse=True, return_counts=True)
        self.ui_y = np.bincount(inverse, weights=y.ravel(), minlength=len(self.ui_keys))
        self.ui_last = np.full(len(self.ui_keys), -1, dtype=np.int64)
        np.maximum.at(self.ui_last, inverse, np.repeat(t, 10))
        self.fitting_events = len(fit)

    def transform(self, raw, indices):
        u = raw["u"][indices].astype(np.int64)
        i = raw["i"][indices].astype(np.int64)
        t = raw["t"][indices].astype(np.int64)
        tb = raw["tb"][indices].astype(np.int64)
        assert len(indices) > 0 and i.shape == (len(indices), 10)
        assert (t >= self.cutoff).all() and ((tb >= 0) & (tb < 4)).all()
        assert ((u >= 0) & (u < self.n_users)).all()
        assert ((i >= 0) & (i < self.n_items)).all()
        keys = u[:, None] * self.n_items + i
        where = np.searchsorted(self.ui_keys, keys)
        safe = np.minimum(where, len(self.ui_keys) - 1)
        found = (where < len(self.ui_keys)) & (self.ui_keys[safe] == keys)
        ui_n = np.where(found, self.ui_n[safe], 0)
        ui_y = np.where(found, self.ui_y[safe], 0)
        ui_last = np.where(found, self.ui_last[safe], -1)
        ub, ib = u * 4 + tb, i * 4 + tb[:, None]

        def broadcast(x):
            return np.broadcast_to(x[:, None], i.shape)

        features = [np.log1p(ui_n), np.log1p(ui_n - ui_y), np.log1p(ui_y),
                    age_feature(t[:, None], ui_last),
                    broadcast(age_feature(t, self.user_last[u])),
                    broadcast(age_feature(t, self.user_click_last[u])),
                    broadcast(np.log1p(self.recent_user_n[u])),
                    broadcast(self.recent_user_ctr[u]),
                    np.log1p(self.recent_item_n[i]), self.recent_item_ctr[i],
                    broadcast(np.log1p(self.bucket_user_n[ub])),
                    broadcast(self.bucket_user_ctr[ub]),
                    np.log1p(self.bucket_item_n[ib]), self.bucket_item_ctr[ib],
                    broadcast(age_feature(t, self.bucket_user_last[ub])),
                    age_feature(t[:, None], self.item_last[i])]
        X = np.stack(features, axis=2).reshape(-1, len(NAMES)).astype(np.float32)
        assert X.shape == (len(indices) * 10, len(NAMES)) and np.isfinite(X).all()
        coverage = {"events": len(indices), "fitting_events": self.fitting_events,
                    "candidate_rows_with_prior_ui_display": int(found.sum()),
                    "candidate_rows_with_prior_ui_display_percent": float(100 * found.mean()),
                    "events_without_fitting_activity": int((self.user_last[u] < 0).sum()),
                    "events_without_recent24h_activity": int((self.recent_user_n[u] == 0).sum())}
        return X, coverage


def slow_features(raw, cutoff, indices):
    """Independent, tiny row-by-row reference; only for the self-test."""
    f = np.flatnonzero(raw["t"] < cutoff)
    result = []
    global_ctr = raw["y"][f].mean()
    for q in indices:
        u, t, bucket = raw["u"][q], raw["t"][q], raw["tb"][q]
        uf = f[raw["u"][f] == u]
        ur = uf[raw["t"][uf] >= cutoff - 86400]
        ub = uf[raw["tb"][uf] == bucket]
        uc = uf[raw["y"][uf].sum(1) > 0]
        uprior = (raw["y"][uf].sum() + 50 * global_ctr) / (10 * len(uf) + 50)

        def age(values):
            return np.log1p(min((t - max(values)) / 3600, 720)) if len(values) else np.log1p(720)

        for item in raw["i"][q]:
            rows, columns = np.nonzero(raw["i"][f] == item)
            it, iy = raw["t"][f[rows]], raw["y"][f[rows], columns]
            ir = it >= cutoff - 86400
            ib = raw["tb"][f[rows]] == bucket
            ui = raw["u"][f[rows]] == u
            iprior = (iy.sum() + 50 * global_ctr) / (len(iy) + 50)
            result.append([np.log1p(ui.sum()), np.log1p(ui.sum() - iy[ui].sum()), np.log1p(iy[ui].sum()),
                           age(it[ui]), age(raw["t"][uf]), age(raw["t"][uc]), np.log1p(len(ur)),
                           (raw["y"][ur].sum() + 50 * uprior) / (10 * len(ur) + 50),
                           np.log1p(ir.sum()), (iy[ir].sum() + 50 * iprior) / (ir.sum() + 50),
                           np.log1p(len(ub)), (raw["y"][ub].sum() + 50 * uprior) / (10 * len(ub) + 50),
                           np.log1p(ib.sum()), (iy[ib].sum() + 50 * iprior) / (ib.sum() + 50),
                           age(raw["t"][ub]), age(it)])
    return np.asarray(result, dtype=np.float32)


def self_test():
    rng = np.random.default_rng(2030)
    cutoff = 200000
    n = 21
    raw = {"u": rng.integers(0, 4, n), "i": rng.integers(0, 13, (n, 10)),
           "t": np.r_[cutoff - 90000, np.arange(13) * 5000 + cutoff - 86400,
                       cutoff - 1, cutoff, cutoff + np.arange(1, 6) * 9000],
           "tb": rng.integers(0, 4, n), "y": np.zeros((n, 10), dtype=np.int8)}
    raw["y"][::2, 3] = 1
    raw["u"][-2:] = 4  # cold users with repeated targets
    raw["i"][-1, 0] = 13  # a genuinely unseen item
    raw["u"][0] = 5  # user without any recent event
    raw["u"][-3] = 5
    indices = np.flatnonzero(raw["t"] >= cutoff)
    features = ExposureFeatures(raw, cutoff)
    X, _ = features.transform(raw, indices)
    np.testing.assert_allclose(X, slow_features(raw, cutoff, indices), rtol=2e-7, atol=2e-7)
    # Arbitrary target/future labels cannot affect any feature, including at the
    # exact cutoff. Histories and their outcomes never enter these aggregates.
    changed = {key: value.copy() for key, value in raw.items()}
    changed["y"][indices] = 0
    changed["y"][indices, 8] = 1
    X2, _ = ExposureFeatures(changed, cutoff).transform(changed, indices)
    np.testing.assert_array_equal(X, X2)
    shuffled = indices[::-1]
    X3, _ = features.transform(raw, shuffled)
    np.testing.assert_array_equal(X3.reshape(-1, 10, 16), X.reshape(-1, 10, 16)[::-1])
    # Fitting row order also must not change counts, maxima or smoothing.
    order = rng.permutation(n)
    unordered = {key: value[order] for key, value in raw.items()}
    restored = np.argsort(order)[indices]
    X4, _ = ExposureFeatures(unordered, cutoff).transform(unordered, restored)
    np.testing.assert_array_equal(X, X4)
    print("Exposure feature self-test passed: 16-feature slow oracle, exact cutoff, future-label exclusion, cold/unseen keys, duplicate impressions and row order.")


if __name__ == "__main__":
    self_test()
