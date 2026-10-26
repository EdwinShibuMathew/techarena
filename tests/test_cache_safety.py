"""Cache containment and failure-safe replacement."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from feature_cache import CombinerFeatureCache


class CacheSafetyTests(unittest.TestCase):
    def test_reject_unsafe_keys_and_symlinks_for_every_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=CombinerFeatureCache(Path(tmp)/"cache")
            args=(np.zeros((10,1)),["x"],np.zeros((1,10)),np.zeros((1,10)))
            for bad in ("../outside","/absolute","a/b","a\\b","", ".", ".."):
                for action in (lambda:cache.save(bad,"split",*args),lambda:cache.load(bad,"split"),lambda:cache.has_cache(bad,"split")):
                    with self.subTest(key=bad),self.assertRaises(ValueError): action()
            outside=Path(tmp)/"outside";outside.write_bytes(b"protected")
            (cache.cache_dir/"m_s_features.npz").symlink_to(outside)
            for action in (lambda:cache.save("m","s",*args),lambda:cache.load("m","s"),lambda:cache.has_cache("m","s")):
                with self.assertRaises(ValueError):action()
            self.assertEqual(outside.read_bytes(),b"protected")

    def test_failed_save_preserves_previous_and_concurrent_updates_are_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=CombinerFeatureCache(Path(tmp)); p=np.full((1,10),.1,dtype=np.float64)
            def save(value):cache.save("m","s",np.full((10,1),value),["x"],p,p)
            save(1); path=Path(tmp)/"m_s_features.npz"; before=path.read_bytes()
            with patch("feature_cache.np.savez_compressed",side_effect=OSError("interruption")),self.assertRaises(OSError):save(2)
            self.assertEqual(path.read_bytes(),before)
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(save,[3,4]))
            x=cache.load("m","s")["X"]
            self.assertTrue(np.all(x==3) or np.all(x==4))
            self.assertEqual(list(Path(tmp).glob(".features-*")),[])
