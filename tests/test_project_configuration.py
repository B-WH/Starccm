from pathlib import Path
import tomllib
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

    def test_scipy_is_declared_as_optional_speed_dependency(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

        self.assertNotIn("scipy", [line.strip().lower() for line in requirements])
        self.assertNotIn("scipy", pyproject["project"]["dependencies"])
        self.assertIn("scipy", pyproject["project"]["optional-dependencies"]["speed"])

    def test_docs_explain_alignment_preview_and_optional_speed_install(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        packaging_readme = Path("packaging/README.md").read_text(encoding="utf-8")

        self.assertIn("预览坐标对齐", readme)
        self.assertIn("比例系数", readme)
        self.assertIn("平移 dx,dy,dz", readme)
        self.assertIn("轴顺序", readme)
        self.assertIn("pip install .[speed]", readme)
        self.assertIn("pip install .[speed]", packaging_readme)

    def test_optional_dependency_loader_module_exists(self) -> None:
        from starccm_pressure.optional_deps import load_ckdtree

        self.assertTrue(callable(load_ckdtree))

    def test_mapping_gui_is_split_from_mapping_core(self) -> None:
        from starccm_pressure.mapping_gui import run_gui

        source = Path("starccm_pressure/map_cgns_pressure_to_inp.py").read_text(encoding="utf-8")

        self.assertTrue(callable(run_gui))
        self.assertIn("starccm_pressure.mapping_gui", source)
        self.assertNotIn("tk.Tk()", source)


if __name__ == "__main__":
    unittest.main()
