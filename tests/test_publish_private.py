"""Publication cannot stage on main or accidentally push the protected branch."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import publish_private as publish


class PublicationTests(unittest.TestCase):
    def fake_run(self, branch="ci/test", private=True, origin=None):
        def run(args, capture=True):
            if args == ["git", "rev-parse", "--show-toplevel"]:
                return str(publish.ROOT)
            if args == ["git", "branch", "--show-current"]:
                return branch
            if args == ["gh", "api", "user"]:
                return json.dumps({"login": publish.OWNER, "id": 1})
            if args[:3] == ["gh", "repo", "view"]:
                return json.dumps({"isPrivate": private, "nameWithOwner": publish.OWNER + "/" + publish.NAME,
                                   "url": "https://github.com/" + publish.OWNER + "/" + publish.NAME})
            if args == ["git", "remote"]:
                return "origin"
            if args == ["git", "remote", "get-url", "origin"]:
                return origin or "https://github.com/" + publish.OWNER + "/" + publish.NAME + ".git"
            if args == ["git", "ls-files", "-z"]:
                return "README.md\0"
            return ""
        return run

    def test_main_and_detached_head_fail_before_staging_or_network(self):
        for branch in ("main", ""):
            with self.subTest(branch=branch), patch.object(publish, "run", side_effect=self.fake_run(branch)) as run:
                with self.assertRaisesRegex(SystemExit, "feature branch"):
                    publish.main()
                self.assertEqual(len(run.call_args_list), 2)

    def test_feature_branch_is_pushed_and_pr_link_is_printed(self):
        output = io.StringIO()
        with patch.object(publish, "run", side_effect=self.fake_run()) as run, \
                patch.object(publish, "read_json", return_value={"files": {}}), \
                patch.object(publish, "validate_files") as validate, \
                patch.object(publish.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                contextlib.redirect_stdout(output):
            publish.main()
        validate.assert_called_once_with(publish.ROOT, ["README.md"])
        run.assert_any_call(["git", "push", "--set-upstream", "origin", "ci/test"], capture=False)
        self.assertIn("/compare/main...ci%2Ftest", output.getvalue())
        self.assertFalse(any(call.args[0][:3] == ["gh", "repo", "create"] for call in run.call_args_list))

    def test_privacy_and_remote_mismatch_stop_before_staging(self):
        for private, origin in ((False, None), (True, "https://github.com/someone/else.git")):
            with self.subTest(private=private, origin=origin), \
                    patch.object(publish, "run", side_effect=self.fake_run(private=private, origin=origin)) as run, \
                    patch.object(publish, "read_json", return_value={"files": {}}):
                with self.assertRaises(SystemExit):
                    publish.main()
                self.assertFalse(any(call.args[0][:2] == ["git", "add"] for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
