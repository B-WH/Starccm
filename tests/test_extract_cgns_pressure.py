from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from extract_cgns_pressure import (
    PressureTimeSeries,
    build_extraction_metadata,
    build_input_file_order_message,
    compute_triangle_surface_geometry,
    gui_default_export_options,
    load_surface_geometry_npz,
    run_gui_extraction_job,
    sort_time_step_paths,
    write_surface_geometry_npz,
)


TEST_OUTPUT_DIR = Path("work/test-output")


def _unlink_test_file(path: Path) -> None:
    """Best-effort cleanup for one explicit test artifact on Windows."""
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


class ExtractCgnsPressureTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        _unlink_test_file(TEST_OUTPUT_DIR / "surface_geometry_cache_test.npz")
        _unlink_test_file(TEST_OUTPUT_DIR / "invalid_surface_geometry_cache_test.npz")

    def test_sort_time_step_paths_orders_embedded_numbers_naturally(self) -> None:
        paths = [
            Path("604@10.cgns"),
            Path("604@2.cgns"),
            Path("604@1.cgns"),
        ]

        sorted_names = [path.name for path in sort_time_step_paths(paths)]

        self.assertEqual(sorted_names, ["604@1.cgns", "604@2.cgns", "604@10.cgns"])

    def test_build_input_file_order_message_reports_sorted_first_and_last(self) -> None:
        message = build_input_file_order_message(
            [Path("604@10.cgns"), Path("604@1.cgns"), Path("604@2.cgns")]
        )

        self.assertIn("3 个 CGNS 文件", message)
        self.assertIn("604@1.cgns", message)
        self.assertIn("604@10.cgns", message)

    def test_gui_default_export_options_use_lightweight_summary_outputs(self) -> None:
        defaults = gui_default_export_options()

        self.assertTrue(defaults["skip_legacy_json"])
        self.assertFalse(defaults["export_complex_spectrum"])
        self.assertTrue(defaults["export_surface_geometry"])
        self.assertTrue(defaults["export_equivalent_force"])

    def test_extraction_metadata_includes_sampling_quality_fields(self) -> None:
        series = PressureTimeSeries(
            node_ids=np.array([1, 2]),
            pressures=np.zeros((4, 2), dtype=float),
            dataset_path="/Base/Zone/Pressure/data",
            file_paths=[
                Path("604@1.cgns"),
                Path("604@2.cgns"),
                Path("604@3.cgns"),
                Path("604@4.cgns"),
            ],
        )

        metadata = build_extraction_metadata(
            series,
            dt=0.25,
            include_dc=False,
            remove_mean=True,
        )

        self.assertEqual(metadata["sample_count"], 4)
        self.assertAlmostEqual(metadata["record_duration_s"], 1.0)
        self.assertAlmostEqual(metadata["frequency_resolution_hz"], 1.0)
        self.assertAlmostEqual(metadata["nyquist_hz"], 2.0)

    def test_surface_geometry_cache_writes_summary_metadata(self) -> None:
        cache_path = TEST_OUTPUT_DIR / "surface_geometry_cache_test.npz"
        geometry = compute_triangle_surface_geometry(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            np.array([[0, 1, 2]], dtype=int),
        )
        try:
            write_surface_geometry_npz(cache_path, geometry)

            with np.load(cache_path) as saved:
                self.assertEqual(int(saved["geometry_cache_schema_version"]), 2)
                self.assertEqual(int(saved["node_count"]), 3)
                self.assertEqual(int(saved["face_count"]), 1)
                np.testing.assert_allclose(saved["coordinate_min"], [0.0, 0.0, 0.0])
                np.testing.assert_allclose(saved["coordinate_max"], [1.0, 1.0, 0.0])
                self.assertAlmostEqual(float(saved["total_area"]), 0.5)
        finally:
            if cache_path.exists():
                _unlink_test_file(cache_path)

    def test_load_surface_geometry_cache_rejects_out_of_range_faces(self) -> None:
        cache_path = TEST_OUTPUT_DIR / "invalid_surface_geometry_cache_test.npz"
        try:
            np.savez_compressed(
                cache_path,
                faces=np.array([[0, 1, 3]], dtype=int),
                centers=np.zeros((1, 3), dtype=float),
                area_vectors=np.array([[0.0, 0.0, 0.5]], dtype=float),
                areas=np.array([0.5], dtype=float),
                normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
                coordinates=np.zeros((3, 3), dtype=float),
            )

            with self.assertRaisesRegex(ValueError, "超出坐标数组范围"):
                load_surface_geometry_npz(cache_path, expected_face_count=1)
        finally:
            if cache_path.exists():
                _unlink_test_file(cache_path)

    def test_gui_extraction_job_passes_surface_geometry_cache_to_streaming_outputs(
        self,
    ) -> None:
        cache_path = "cache/surface_geometry.npz"
        with (
            patch(
                "extract_cgns_pressure.expand_input_files",
                return_value=[Path("604@1.cgns")],
            ),
            patch(
                "extract_cgns_pressure.write_streaming_summary_outputs",
                return_value={"extraction_metadata": Path("out/extraction_metadata.json")},
            ) as write_streaming,
            patch(
                "extract_cgns_pressure.build_streaming_success_message",
                return_value="ok",
            ),
        ):
            message = run_gui_extraction_job(
                ["604@1.cgns"],
                dt=0.001,
                output_dir="out",
                pressure_name="Pressure",
                include_dc=False,
                skip_legacy_json=True,
                export_complex_spectrum=False,
                export_surface_geometry=True,
                export_equivalent_force=True,
                surface_geometry_cache=cache_path,
            )

        self.assertEqual(message, "ok")
        self.assertEqual(
            write_streaming.call_args.kwargs["surface_geometry_cache"],
            cache_path,
        )


if __name__ == "__main__":
    unittest.main()
