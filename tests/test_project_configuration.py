from pathlib import Path
import unittest


class ProjectConfigurationTests(unittest.TestCase):
    def test_pyproject_declares_lint_and_test_configuration(self) -> None:
        text = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("[tool.ruff]", text)
        self.assertIn("[tool.ruff.lint]", text)
        self.assertIn("[tool.unittest]", text)
        self.assertIn("tests", text)

    def test_windows_exe_packaging_configuration_is_documented(self) -> None:
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        build_script = Path("packaging/build_exe.ps1")
        packaging_readme = Path("packaging/README.md")

        self.assertIn('"pyinstaller"', pyproject.lower())
        self.assertTrue(build_script.exists())
        self.assertTrue(packaging_readme.exists())


if __name__ == "__main__":
    unittest.main()
