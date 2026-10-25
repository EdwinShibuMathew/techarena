"""Unit tests validating agentic CLI configurations and rules for Codex, Copilot, and Antigravity."""

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AgentConfigurationTests(unittest.TestCase):
    def test_required_root_files_exist(self):
        for path in [
            ROOT / "AGENTS.md",
            ROOT / "GEMINI.md",
            ROOT / ".agentignore",
            ROOT / ".ignore",
        ]:
            self.assertTrue(path.is_file(), f"Missing required agent file: {path.name}")
            self.assertGreater(path.stat().st_size, 0, f"Empty file: {path.name}")

    def test_codex_configuration(self):
        codex_dir = ROOT / ".codex"
        self.assertTrue(codex_dir.is_dir(), ".codex directory must exist")

        instructions = codex_dir / "instructions.md"
        self.assertTrue(instructions.is_file(), ".codex/instructions.md must exist")
        self.assertGreater(instructions.stat().st_size, 0)

        config_path = codex_dir / "config.json"
        self.assertTrue(config_path.is_file(), ".codex/config.json must exist")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.assertEqual(config.get("project"), "techarena-chaiscript")
        self.assertEqual(config.get("baseline"), "v6")
        self.assertIn("local/", config.get("exclude_paths", []))
        self.assertIn("data/", config.get("exclude_paths", []))
        self.assertIn("artifacts/", config.get("exclude_paths", []))

    def test_github_copilot_instructions(self):
        copilot_file = ROOT / ".github/copilot-instructions.md"
        self.assertTrue(copilot_file.is_file(), ".github/copilot-instructions.md must exist")
        content = copilot_file.read_text(encoding="utf-8")
        self.assertIn("predict_v6.py", content)
        self.assertIn("legacy/", content)
        self.assertIn("Global row ROC AUC", content)

    def test_antigravity_rules_budget_and_limits(self):
        rules_dir = ROOT / ".agents/rules"
        self.assertTrue(rules_dir.is_dir(), ".agents/rules directory must exist")
        expected_rules = [
            "00-token-efficiency.md",
            "01-architecture-entrypoints.md",
            "02-ml-protocol.md",
        ]
        for rule_name in expected_rules:
            rule_path = rules_dir / rule_name
            self.assertTrue(rule_path.is_file(), f"Missing rule: {rule_name}")
            # Antigravity CLI enforces a 24,000 byte limit per rule file
            size = rule_path.stat().st_size
            self.assertLessEqual(
                size,
                24000,
                f"Rule {rule_name} exceeds Antigravity 24KB limit ({size} bytes)",
            )

    def test_antigravity_skills_frontmatter(self):
        skills_dir = ROOT / ".agents/skills"
        self.assertTrue(skills_dir.is_dir(), ".agents/skills directory must exist")
        expected_skills = [
            "reproduce-v6",
            "run-experiment",
            "verify-pipeline",
            "token-saver",
        ]
        for skill_name in expected_skills:
            skill_file = skills_dir / skill_name / "SKILL.md"
            self.assertTrue(skill_file.is_file(), f"Missing SKILL.md for {skill_name}")
            content = skill_file.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("---"), f"{skill_name}/SKILL.md must start with YAML frontmatter")
            parts = content.split("---", 2)
            self.assertGreaterEqual(len(parts), 3, f"Invalid YAML frontmatter in {skill_name}/SKILL.md")
            frontmatter = parts[1]
            self.assertIn(f"name: {skill_name}", frontmatter)
            self.assertIn("description:", frontmatter)


if __name__ == "__main__":
    unittest.main()
