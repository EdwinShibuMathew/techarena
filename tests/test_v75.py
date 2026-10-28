"""V7.5 chronological replicas, feature schema, leases and delivery contracts."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments/v7.5-history-age-v1'), str(ROOT / 'tools'),
                str(ROOT / 'legacy'), str(ROOT / 'experiments/v7.4-rebuild-v1')]
import protocol as v75
import campaign
import pipeline
from ml_workflow.storage import read, write_new, save_arrays
from ml_workflow.contracts import digest


def test_fitting_prefixes_are_causal_and_cover_staleness():
    for hours in (24, 48, 96, 120):
        meta = v75.JAN20 + hours * 3600
        prefixes = v75.fitting_prefixes(meta, True)
        assert prefixes and all(0 <= meta - c <= 96 * 3600 for c in prefixes)
        assert v75.FULL not in prefixes and max(prefixes) == meta
        assert v75.fitting_prefixes(meta, False) == [meta]
    assert len(v75.ADDED_NAMES) == len(set(v75.ADDED_NAMES)) == 10
    assert list(v75.CANDIDATES) == ['A', 'B', 'C']


def test_v75_cache_rejects_changed_identity_and_bytes(tmp_path):
    path = tmp_path / 'features.npz'
    pipeline.cache(tmp_path, path, {'source': 'a'}, lambda: {'x': np.zeros(2)})
    with pytest.raises(ValueError, match='Cache'):
        pipeline.cache(tmp_path, path, {'source': 'b'}, lambda: {})
    with path.open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='Cache'):
        pipeline.cache(tmp_path, path, {'source': 'a'}, lambda: {})


def test_internal_worker_requires_controller(tmp_path):
    with patch.object(sys, 'argv', ['campaign.py', 'train', '--root', str(tmp_path)]):
        with pytest.raises(SystemExit) as failure: campaign.main()
    assert failure.value.code == 1
    assert not (tmp_path / 'artifacts/rebuild/compute-ledger.json').exists()


def test_new_model_loader_enforces_feature_width_and_receipt(tmp_path):
    pytest.importorskip('catboost')
    n = 100; rng = np.random.default_rng(2026)
    X = rng.normal(size=(n, 47)).astype(np.float32); labels = (X[:, 0] > 0).astype(np.int8)
    names = ['f' + str(i) for i in range(47)]; path = tmp_path / 'model.cbm'
    pipeline.fit_model(tmp_path, path, X, names, labels, {'test': 'schema'})
    p = pipeline.predict_model(path, X, names, 47)
    assert p.shape == (10, 10) and np.isfinite(p).all()
    with pytest.raises(ValueError, match='schema'): pipeline.predict_model(path, X, names, 37)


def test_baseline_publication_requires_two_accounted_replays(tmp_path):
    work = tmp_path / 'work'; work.mkdir()
    with patch.object(campaign, 'completed', return_value=False):
        with pytest.raises(ValueError, match='Two accounted'): pipeline.publish_baseline(tmp_path, work)
    assert not (tmp_path / 'artifacts/submissions/rebuilt_v6_submission.csv').exists()
