"""Verify historical report preservation without executing retired campaign code."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"docs/results/v8-archive/manifest.json"
CAMPAIGNS={"v8-pairlogit","v8-enriched-logloss","v8-temporal-combiner-search",
           "v8-seed44-row-auc","v8-robust-ensemble","v8-residual-event"}
REVISION="3a902173cf111dcb748140a31293425ae66b2142"


class V8ArchiveTests(unittest.TestCase):
    def test_all_reports_preserve_original_bytes_or_only_documented_annotations(self):
        manifest=json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["source_revision"],REVISION)
        files=manifest["files"]
        self.assertEqual(len(files),32)
        self.assertEqual(sum(name.endswith(".json") for name in files),26)
        self.assertEqual({Path(name).parent.name for name in files},CAMPAIGNS)
        actual={path.relative_to(ROOT).as_posix() for campaign in CAMPAIGNS
                for path in (ROOT/"docs/results"/campaign).iterdir() if path.is_file()}
        self.assertEqual(actual,set(files))
        for name,record in files.items():
            with self.subTest(name=name):
                path=ROOT/name
                self.assertEqual(record["source_url"],f"https://github.com/EdwinShibuMathew/techarena/blob/{REVISION}/{name}")
                original=archived=path.read_bytes()
                self.assertEqual(hashlib.sha256(archived).hexdigest(),record["archived_sha256"])
                if name.endswith(".md"):
                    prefix=manifest["historical_summaries_prefix"].encode()
                    self.assertTrue(archived.startswith(prefix))
                    link=record["protocol_link"]
                    self.assertIn('/blob/'+REVISION+'/',link["pinned"])
                    self.assertEqual(archived.count((']('+link["pinned"]+')').encode()),1)
                    original=archived[len(prefix):].replace((']('+link["pinned"]+')').encode(),(']('+link["original"]+')').encode())
                else:
                    self.assertEqual(record["source_sha256"],record["archived_sha256"])
                    json.loads(archived)
                self.assertEqual(hashlib.sha256(original).hexdigest(),record["source_sha256"])

    def test_archive_links_are_pinned_or_resolve_locally(self):
        manifest=json.loads(MANIFEST.read_text())
        pages=[ROOT/name for name in manifest["files"] if name.endswith(".md")]
        pages.append(MANIFEST.with_name("README.md"))
        for page in pages:
            for target in re.findall(r'\]\(([^)]+)\)',page.read_text()):
                with self.subTest(page=page.name,target=target):
                    if target.startswith("https://"):
                        if "/blob/" in target or "/commit/" in target:self.assertIn(REVISION,target)
                    else:
                        path=(page.parent/target.split("#",1)[0]).resolve()
                        self.assertTrue(path.is_relative_to(ROOT))
                        self.assertTrue(path.is_file(),str(path))

    def test_closed_decisions_and_recorded_budget_remain_explicit(self):
        for campaign in CAMPAIGNS:
            value=json.loads((ROOT/"docs/results"/campaign/"decision.json").read_text())
            self.assertFalse(value["submission_ready"])
            self.assertEqual(value["status"],"rejected_at_five_day_gate" if campaign=="v8-seed44-row-auc" else "rejected_at_primary_gate")
        manifest=json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["recorded_cumulative_compute_seconds"],1974.359347462001)
        self.assertEqual(manifest["original_compute_limit_seconds"],14400)
        residual=json.loads((ROOT/"docs/results/v8-residual-event/decision.json").read_text())
        self.assertEqual(residual["total_compute_seconds"],manifest["recorded_cumulative_compute_seconds"])

    def test_historical_execution_entrypoints_are_not_imported(self):
        for campaign in CAMPAIGNS:
            self.assertFalse((ROOT/"experiments"/campaign).exists(),campaign)
        for suffix in ("","_candidate","_enriched","_residual","_seed44","_selected"):
            self.assertFalse((ROOT/"tools"/("predict_v8"+suffix+".py")).exists())
