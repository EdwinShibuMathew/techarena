"""Local orchestration regression; GPU/model libraries are explicit stand-ins.

Run with the bundled Windows Python. Real training is checked separately by
run_exposure_matched_trial.py self-test inside Edwin's CUDA WSL environment.
"""
import copy
import pickle
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_exposure_matched_trial as app
from exposure_matched_preference import file_hash, RECIPE, training_guard, numeric_self_test
from run_row_auc_trial import prepare_sources
from row_auc_objective import CONFIG
from audit_v5 import auc


def main():
    src = Path(__file__).resolve().parent
    attachment = Path(r"C:\Users\Edwin\.codex\attachments\1c10c644-cac2-4cd1-ad16-6b69900d344b\Pasted text.txt").read_text(encoding="utf-8")
    pref = attachment.split("========== train_preferences.py ==========")[1].split("========== train_events.py ==========")[0].strip() + "\n"
    event = attachment.split("========== train_events.py ==========")[1].split("==========")[0].strip() + "\n"
    numeric_self_test()
    with tempfile.TemporaryDirectory(prefix="exposure_tools_", dir=src.parent) as tmp:
        projects = Path(tmp)
        root, trial = projects / "techarena-final", projects / "techarena-exposure-matched-trial"
        (root / "src").mkdir(parents=True)
        (root / "data/cache").mkdir(parents=True)
        for path in src.glob("*.py"):
            shutil.copy2(path, root / "src" / path.name)
        original = projects / "techarena-seed43/src"
        original.mkdir(parents=True)
        (original / "train_preferences.py").write_text(pref, encoding="utf-8")
        c = app.CUTOFFS
        times = np.array([c["jan24"] - 1000, c["jan24"] - 900,
                          *[c[day] + i for day in c for i in range(3)]])
        n, rng = len(times), np.random.default_rng(400)
        raw = {"t": times, "u": np.array([0, 1, 0, 1, 2, 0, 1, 3, 0, 2, 4]),
               "i": rng.integers(0, 11, (n, 10)), "y": np.zeros((n, 10), dtype=np.int8),
               "tb": np.zeros(n, dtype=np.int8), "dt": np.zeros(n, dtype=np.int8),
               "h": np.full((n, 20), -1, dtype=np.int32), "ht": np.zeros((n, 20), dtype=np.int64)}
        raw["y"][np.arange(n), np.arange(n) % 10] = 1
        np.savez(root / "data/cache/train_events.npz", **raw)
        v5refs, expected = {}, {}
        short = projects / "techarena-row-auc-trial"
        (short / "outputs").mkdir(parents=True)
        (short / "work").mkdir()
        for case, idx in [("two_day", np.array([5, 6, 7, 9, 10])), ("jan26", np.array([8, 9, 10]))]:
            members = [rng.uniform(.01, .15, (len(idx), 10)) for _ in range(5)]
            updated = [rng.uniform(.01, .15, (len(idx), 10)) for _ in range(2)]
            baseline = app.ensemble(updated + members[2:])
            v5refs[case] = {"indices": idx, "labels": raw["y"][idx], "baseline": app.ensemble(members), "members": members}
            np.savez(short / "work" / f"{case}_ensemble_predictions.npz", valid_idx=idx, labels=raw["y"][idx],
                     weights=app.WEIGHTS, candidate=baseline, new_dim8_seed42=updated[0], new_dim8_seed43=updated[1])
            expected["two_day_primary" if case == "two_day" else "jan26_secondary"] = {"candidate_row_ranking_auc": auc(raw["y"][idx], baseline)}
        app.write_json(short / "outputs/ensemble_trial_results.json", expected)
        for fold, cutoff in c.items():
            d = short / f"dim8_seed42/{fold}"
            (d / "src").mkdir(parents=True)
            (d / "data/cache").mkdir(parents=True)
            scripts = prepare_sources(pref, event, 42, fold)
            (d / "src/train_events.py").write_text(scripts["train_events.py"], encoding="utf-8")
            valid = np.flatnonzero((times >= cutoff) & (times < cutoff + 86400))
            np.savez(d / "data/cache" / f"split_{fold}.npz", train_idx=np.flatnonzero(times < cutoff), valid_idx=valid)
            np.savez(d / "data/cache/preference_training.npz", users=[0, 1], items=[1, 2],
                     timestamps=[cutoff - 800, cutoff - 700], n_users=6, n_items=11, cutoff=cutoff)
            shutil.copy2(root / "data/cache/train_events.npz", d / "data/cache/train_events.npz")
        final = projects / "techarena-row-auc-final"
        (final / "outputs").mkdir(parents=True)
        oldfile, newfile = projects / "v5.csv", final / "outputs/techarena_submission_v6_candidate.csv"
        oldfile.write_text("unchanged v5")
        newfile.write_text("unchanged v6")
        fp = {"input_sha256": app.hashes([oldfile]), "prepared_sha256": {},
              "v5_protected_sha256": app.hashes([oldfile]), "candidate_csv": "outputs/techarena_submission_v6_candidate.csv"}
        app.write_json(final / "outputs/frozen_protocol.json", fp)
        app.write_json(final / "outputs/training_results.json", {})
        app.write_json(final / "outputs/candidate_submission_checks.json", {"version": "v6_candidate_row_ranking",
            "submission_ready": True, "v5_files_unchanged": True, "validation_pipeline_check_passed": True,
            "submission_sha256": file_hash(newfile), "artifact_sha256": {}})
        finalmod = types.ModuleType("run_row_auc_final")
        finalmod.verify = lambda f, p: app.verify_files(Path(), p["input_sha256"])
        finalmod.verify_training = lambda *a: None
        finalmod.check_v5 = lambda *a: None
        finalmod.validation_replay = lambda *a: (raw, {k: {"passed": True, "member_max_abs_differences": {"42": 0., "43": 0.}} for k in ["two_day", "jan26"]})
        longmod = types.ModuleType("run_row_auc_five_day")
        longmod.verify_artifacts = app.verify_files
        torch = types.ModuleType("torch")
        torch.load = lambda path, **kw: pickle.loads(Path(path).read_bytes())
        torch.is_tensor, torch.device = lambda x: False, str
        calls, fitting = [], []

        def launch(args, cwd, check):
            folder = Path(cwd)
            fold, script = folder.name, Path(args[-1]).name
            stage, _, filename, epoch = next(s for s in app.STAGES if s[1] == script)
            saved = {"epoch": epoch, "dimension": 8, "seed": 42, "cutoff": c[fold], "n_users": 6, "n_items": 11,
                "sampler_recipe": RECIPE, "frozen_protocol_sha256": file_hash(trial / "outputs/frozen_protocol.json"),
                "model_state": {}, "optimizer_state": {"state": {}, "param_groups": [{"lr": .012 if stage == "initial" else .003}]}}
            if stage == "event":
                saved.update(event_objective=CONFIG["objective"], pair_sampling_seed=3042,
                             preference_checkpoint_sha256=file_hash(folder / "work/preference_best.pt"))
            else:
                saved.update(stage="preference", sampling_sha256=file_hash(folder / "data/cache/exposure_sampling.npz"))
                if stage == "fine":
                    saved.update(initial_checkpoint_sha256=file_hash(folder / "work/preference_initial.pt"), fine_tuning_sampling_seed=1042)
            (folder / "work" / filename).write_bytes(pickle.dumps(saved))
            calls.append((fold, stage))

        def pipeline(cp, training, target, positive, device):
            X = np.zeros((len(target["u"]) * 10, 37))
            X[:, 0], X[:, 1] = target["i"].ravel() / 20, np.repeat(target["t"], 10)
            p = (.02 + X[:, 0] / 10).reshape(-1, 10)
            return X, [f"f{i}" for i in range(37)], p, p.copy(), None, {}

        class Model:
            def save_model(self, path):
                Path(path).write_text("545 unchanged trees")

        def fit(X, labels, names):
            ts = X[:, 1]
            assert ((ts >= c["jan24"]) & (ts < c["jan25"])).all() or ((ts >= c["jan25"]) & (ts < c["jan26"])).all()
            fitting.append(len(X))
            return Model()

        buildmod = types.ModuleType("build_submission")
        buildmod.FIT_FIELDS = ["u", "i", "y"]
        buildmod.TARGET_FIELDS = ["u", "i", "t", "tb", "dt", "h", "ht"]
        buildmod.predict_combiner = lambda model, X, names: .02 + X[:, 0] / 10
        buildmod.load_combiner = lambda path: Model() if Path(path).read_text() == "545 unchanged trees" else 1 / 0
        buildmod.check_close = lambda a, b, name, **kw: np.testing.assert_allclose(a, b, **kw)
        fitmod = types.ModuleType("evaluate_dim16_trial")
        fitmod.fit_combiner = fit
        original_resolve = Path.resolve

        def resolve(path, *a, **kw):
            if path.name == "train_events.npz" and "dim8_seed42" in path.parts:
                return original_resolve(root / "data/cache/train_events.npz", *a, **kw)
            return original_resolve(path, *a, **kw)

        # V6 historical reconstruction stays real; only V5 source lookup is a fixture.
        with patch.dict(sys.modules, {"torch": torch, "run_row_auc_final": finalmod, "run_row_auc_five_day": longmod,
                "build_submission": buildmod, "evaluate_dim16_trial": fitmod}), \
             patch.object(app, "load_references", lambda *a: copy.deepcopy(v5refs)), \
             patch.object(Path, "symlink_to", lambda path, target: shutil.copy2(target, path)):
            with patch.object(Path, "resolve", resolve):
                app.setup(root, trial)
            protocol = app.read_json(trial / "outputs/frozen_protocol.json")
            app.verify(trial, protocol)
            training_guard(trial / "jan24/src/train_preferences.py")
            app.setup(root, trial)
            with patch.object(app.subprocess, "run", launch):
                app.train(trial, protocol)
                app.train(trial, protocol)
            assert len(calls) == 9
            with patch.object(app, "make_pipeline", lambda: pipeline):
                app.evaluate(root, trial, protocol)
                app.evaluate(root, trial, protocol)
            assert len(fitting) == 2
            report = app.read_json(trial / app.REPORT)
            assert report["v5_files_unchanged"] and report["v6_files_unchanged"] and not report["submission_created"]
            assert oldfile.read_text() == "unchanged v5" and newfile.read_text() == "unchanged v6"
            oldfile.write_text("tampered")
            try:
                app.verify(trial, protocol)
            except AssertionError:
                pass
            else:
                raise AssertionError("Protected V5 tamper accepted")
            oldfile.write_text("unchanged v5")
            cache = trial / "work/two_day_ensemble_predictions.npz"
            content = cache.read_bytes()
            cache.write_bytes(content + b"tampered")
            try:
                app.evaluate(root, trial, protocol)
            except AssertionError:
                pass
            else:
                raise AssertionError("Prediction tamper accepted")
            cache.write_bytes(content)
            app.verify(trial, protocol)
    print("Synthetic lifecycle passed: exact source adaptation, V6 reconstruction, atomic setup, nine stages, honest meta fits, resume, V5/V6 protection and tamper rejection. GPU/training/CatBoost/symlinks were stand-ins.")


if __name__ == "__main__":
    main()
