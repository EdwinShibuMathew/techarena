"""Local orchestration regression; CUDA/training/CatBoost/symlinks are stand-ins."""

import copy
import pickle
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_time_bilinear_trial as app
import run_exposure_matched_trial as baseline_tools
from time_bilinear_model import RECIPE, file_hash, numpy_interaction, training_guard
from run_row_auc_trial import prepare_sources
from row_auc_objective import CONFIG
from audit_v5 import auc


class Tensor(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=float).view(cls)

    def square(self):
        return self * self

    def sqrt(self):
        return np.sqrt(self)

    def abs(self):
        return np.abs(self)

    def is_floating_point(self):
        return self.dtype.kind == "f"


def numeric_test():
    rng = np.random.default_rng(2040)
    u, v = rng.normal(size=(4, 8)), rng.normal(size=(4, 10, 8))
    delta, bucket, active = rng.normal(size=(4, 8, 8)), np.arange(4), np.array([1, 1, 0, 1])
    np.testing.assert_allclose(numpy_interaction(u, v, delta, bucket, active),
                               np.einsum("bd,bde,bke->bk", u * active[:, None], delta[bucket], v), rtol=1e-12, atol=1e-12)
    for b, left, right in [(0, 1, 3), (1, 2, 5), (2, 7, 4), (3, 4, 6)]:
        plus, minus = delta.copy(), delta.copy()
        plus[b, left, right] += 1e-5
        minus[b, left, right] -= 1e-5
        grad = (numpy_interaction(u, v, plus, bucket, active).sum() - numpy_interaction(u, v, minus, bucket, active).sum()) / 2e-5
        expected = sum(active[k] * u[k, left] * v[k, :, right].sum() for k in range(4) if bucket[k] == b)
        np.testing.assert_allclose(grad, expected, rtol=1e-7, atol=1e-8)


def main():
    numeric_test()
    src = Path(__file__).resolve().parent
    attachment = Path(r"C:\Users\Edwin\.codex\attachments\1c10c644-cac2-4cd1-ad16-6b69900d344b\Pasted text.txt").read_text(encoding="utf-8")
    pref = attachment.split("========== train_preferences.py ==========")[1].split("========== train_events.py ==========")[0].strip() + "\n"
    event = attachment.split("========== train_events.py ==========")[1].split("==========")[0].strip() + "\n"
    with tempfile.TemporaryDirectory(prefix="time_bilinear_tools_", dir=src.parent) as temporary:
        projects = Path(temporary)
        root, trial = projects / "techarena-final", projects / "techarena-time-bilinear-trial"
        (root / "src").mkdir(parents=True)
        (root / "data/cache").mkdir(parents=True)
        for path in src.glob("*.py"):
            shutil.copy2(path, root / "src" / path.name)
        c, rng = app.CUTOFFS, np.random.default_rng(2041)
        times = np.array([c["jan24"] - 1000 + i for i in range(4)] + [c[day] + i for day in c for i in range(3)])
        n = len(times)
        raw = {"t": times, "u": np.array([0, 1, 0, 1, 0, 1, 2, 0, 1, 3, 0, 2, 4]),
               "i": rng.integers(0, 11, (n, 10)), "y": np.zeros((n, 10), dtype=np.int8),
               "tb": (np.arange(n) % 4).astype(np.int8), "dt": np.zeros(n, dtype=np.int8),
               "h": np.full((n, 20), -1, dtype=np.int32), "ht": np.zeros((n, 20), dtype=np.int64)}
        raw["y"][np.arange(n), np.arange(n) % 10] = 1
        np.savez(root / "data/cache/train_events.npz", **raw)
        short = projects / "techarena-row-auc-trial"
        (short / "outputs").mkdir(parents=True)
        (short / "work").mkdir()
        v5refs, expected = {}, {}
        for case, idx in [("two_day", np.array([7, 8, 9, 11, 12])), ("jan26", np.array([10, 11, 12]))]:
            members = [rng.uniform(.01, .15, (len(idx), 10)) for _ in range(5)]
            updated = [rng.uniform(.01, .15, (len(idx), 10)) for _ in range(2)]
            baseline = app.ensemble(updated + members[2:])
            v5refs[case] = {"indices": idx, "labels": raw["y"][idx], "baseline": app.ensemble(members), "members": members}
            np.savez(short / "work" / f"{case}_ensemble_predictions.npz", valid_idx=idx, labels=raw["y"][idx], weights=app.WEIGHTS,
                     candidate=baseline, new_dim8_seed42=updated[0], new_dim8_seed43=updated[1])
            expected["two_day_primary" if case == "two_day" else "jan26_secondary"] = {"candidate_row_ranking_auc": auc(raw["y"][idx], baseline)}
        app.write_json(short / "outputs/ensemble_trial_results.json", expected)
        for fold, cutoff in c.items():
            d = short / f"dim8_seed42/{fold}"
            for part in ["src", "data/cache", "work"]:
                (d / part).mkdir(parents=True)
            for name, text in prepare_sources(pref, event, 42, fold).items():
                (d / "src" / name).write_text(text, encoding="utf-8")
            valid = np.flatnonzero((times >= cutoff) & (times < cutoff + 86400))
            np.savez(d / "data/cache" / f"split_{fold}.npz", train_idx=np.flatnonzero(times < cutoff), valid_idx=valid)
            np.savez(d / "data/cache/preference_training.npz", users=[0, 1], items=[1, 2],
                     timestamps=[cutoff - 800, cutoff - 700], n_users=6, n_items=11, cutoff=cutoff)
            shutil.copy2(root / "data/cache/train_events.npz", d / "data/cache/train_events.npz")
            (d / "work/preference_best.pt").write_bytes(pickle.dumps({"epoch": 260, "dimension": 8, "n_users": 6, "n_items": 11, "fine_tuning_sampling_seed": 1042}))
        final = projects / "techarena-row-auc-final"
        (final / "outputs").mkdir(parents=True)
        old, current = projects / "v5.csv", final / "outputs/techarena_submission_v6_candidate.csv"
        old.write_text("preserved v5")
        current.write_text("preserved v6")
        fp = {"input_sha256": app.hashes([old]), "prepared_sha256": {}, "v5_protected_sha256": app.hashes([old]),
              "candidate_csv": "outputs/techarena_submission_v6_candidate.csv"}
        app.write_json(final / "outputs/frozen_protocol.json", fp)
        app.write_json(final / "outputs/training_results.json", {})
        app.write_json(final / "outputs/candidate_submission_checks.json", {"version": "v6_candidate_row_ranking", "submission_ready": True,
            "v5_files_unchanged": True, "validation_pipeline_check_passed": True, "submission_sha256": file_hash(current), "artifact_sha256": {}})
        finalmod = types.ModuleType("run_row_auc_final")
        finalmod.verify = lambda f, p: app.verify_files(Path(), p["input_sha256"])
        finalmod.verify_training = lambda *a: None
        finalmod.check_v5 = lambda *a: None
        finalmod.validation_replay = lambda *a: (raw, {k: {"passed": True, "member_max_abs_differences": {"42": 0., "43": 0.}} for k in ["two_day", "jan26"]})
        longmod = types.ModuleType("run_row_auc_five_day")
        longmod.verify_artifacts = app.verify_files
        torch = types.ModuleType("torch")
        torch.load = lambda path, **kw: pickle.loads(Path(path).read_bytes())
        torch.is_tensor, torch.isfinite, torch.device = lambda x: isinstance(x, Tensor), np.isfinite, str
        calls, fitting = [], []

        def launch(args, cwd, check):
            folder = Path(cwd)
            fold = folder.name
            saved = {"epoch": 8, "dimension": 8, "seed": 42, "cutoff": c[fold], "n_users": 6, "n_items": 11,
                "time_bilinear_recipe": RECIPE, "frozen_protocol_sha256": file_hash(trial / "outputs/frozen_protocol.json"),
                "model_state": {"time_interaction": Tensor(np.full((4, 8, 8), .001))}, "optimizer_state": {"state": {}},
                "event_objective": CONFIG["objective"], "pair_sampling_seed": 3042,
                "preference_checkpoint_sha256": file_hash(folder / "work/preference_best.pt")}
            (folder / "work/event_best.pt").write_bytes(pickle.dumps(saved))
            calls.append(fold)

        def pipeline(cp, training, target, positive, device):
            X = np.zeros((len(target["u"]) * 10, 37))
            X[:, 0], X[:, 1] = target["i"].ravel() / 20, np.repeat(target["t"], 10)
            p = (.02 + X[:, 0] / 10).reshape(-1, 10)
            return X, [f"f{i}" for i in range(37)], p, p.copy(), None, {}

        class Model:
            def save_model(self, path):
                Path(path).write_text("545 trees")

        def fit(X, labels, names):
            ts = X[:, 1]
            assert ((ts >= c["jan24"]) & (ts < c["jan25"])).all() or ((ts >= c["jan25"]) & (ts < c["jan26"])).all()
            fitting.append(len(X))
            return Model()

        build = types.ModuleType("build_submission")
        build.FIT_FIELDS, build.TARGET_FIELDS = ["u", "i", "y"], ["u", "i", "t", "tb", "dt", "h", "ht"]
        build.predict_combiner = lambda model, X, names: .02 + X[:, 0] / 10
        build.load_combiner = lambda path: Model() if Path(path).read_text() == "545 trees" else 1 / 0
        build.check_close = lambda a, b, name, **kw: np.testing.assert_allclose(a, b, **kw)
        meta = types.ModuleType("evaluate_dim16_trial")
        meta.fit_combiner = fit
        original_resolve = Path.resolve

        def resolve(path, *a, **kw):
            if path.name == "train_events.npz" and "dim8_seed42" in path.parts:
                return original_resolve(root / "data/cache/train_events.npz", *a, **kw)
            return original_resolve(path, *a, **kw)

        with patch.dict(sys.modules, {"torch": torch, "run_row_auc_final": finalmod, "run_row_auc_five_day": longmod,
                "build_submission": build, "evaluate_dim16_trial": meta}), \
             patch.object(baseline_tools, "load_references", lambda *a: copy.deepcopy(v5refs)), \
             patch.object(Path, "symlink_to", lambda path, target: shutil.copy2(target, path)):
            with patch.object(Path, "resolve", resolve):
                app.setup(root, trial)
            protocol = app.read_json(trial / "outputs/frozen_protocol.json")
            app.verify(trial, protocol)
            training_guard(trial / "jan24/src/train_events.py")
            app.setup(root, trial)
            with patch.object(app.subprocess, "run", launch):
                app.train(trial, protocol)
                app.train(trial, protocol)
            assert len(calls) == 3
            with patch.object(app, "make_pipeline", lambda: pipeline):
                app.evaluate(root, trial, protocol)
                app.evaluate(root, trial, protocol)
            assert len(fitting) == 2
            report = app.read_json(trial / app.REPORT)
            assert report["v5_files_unchanged"] and report["v6_files_unchanged"] and report["unchanged_four_members_reused_exactly"]
            for case in ["two_day", "jan26"]:
                with np.load(trial / "work" / f"{case}_ensemble_predictions.npz") as saved:
                    np.testing.assert_array_equal(saved["v6_members"][1:], saved["candidate_members"][1:])
            assert old.read_text() == "preserved v5" and current.read_text() == "preserved v6"
            current.write_text("tampered")
            try:
                app.verify(trial, protocol)
            except AssertionError:
                pass
            else:
                raise AssertionError("V6 modification accepted")
            current.write_text("preserved v6")
            cache = trial / "work/two_day_ensemble_predictions.npz"
            content = cache.read_bytes()
            cache.write_bytes(content + b"tampered")
            try:
                app.evaluate(root, trial, protocol)
            except AssertionError:
                pass
            else:
                raise AssertionError("Changed predictions accepted")
            cache.write_bytes(content)
            app.verify(trial, protocol)
    print("Local time-bilinear checks passed: score/gradient oracles, original source-update ASTs, V6 reconstruction, atomic setup, three fits/two honest meta fits, four retained members exact, resume and tamper rejection. GPU/training/CatBoost/symlinks were stand-ins.")


if __name__ == "__main__":
    main()
