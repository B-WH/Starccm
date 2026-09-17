from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from extract_cgns_pressure import (
    PressureTimeSeries,
    SurfaceGeometry,
    build_extraction_metadata,
    build_input_file_order_message,
    compute_equivalent_force_spectrum,
    compute_pressure_complex_spectrum,
    compute_streaming_equivalent_force_summary,
    compute_triangle_surface_geometry,
    gui_default_export_options,
    load_surface_geometry_npz,
    run_gui_extraction_job,
    sort_time_step_paths,
    write_pressure_complex_spectrum_npz,
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
        _unlink_test_file(TEST_OUTPUT_DIR / "normalized_complex_spectrum_test.npz")

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

    def test_complex_spectrum_stores_single_sided_peak_complex_amplitude(self) -> None:
        sample_count = 8
        phase_rad = 0.3
        amplitude = 2.5
        nyquist_amplitude = 1.75
        sample_index = np.arange(sample_count, dtype=float)
        pressures = np.column_stack(
            [
                amplitude
                * np.cos(2.0 * np.pi * sample_index / sample_count + phase_rad),
                nyquist_amplitude * np.cos(np.pi * sample_index),
            ]
        )

        spectrum = compute_pressure_complex_spectrum(
            pressures,
            dt=1.0 / sample_count,
            include_dc=False,
        )
        pressure_complex = spectrum.pressure_real + 1j * spectrum.pressure_imag
        expected = amplitude * np.exp(1j * phase_rad)

        np.testing.assert_allclose(pressure_complex[0, 0], expected, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(
            spectrum.pressure_amplitude[0, 0],
            amplitude,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            pressure_complex[-1, 1],
            nyquist_amplitude,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

        force = compute_equivalent_force_spectrum(
            compute_pressure_complex_spectrum(
                pressures[:, :1],
                dt=1.0 / sample_count,
                include_dc=False,
            ),
            np.array([[0.0, 0.0, 2.0]], dtype=float),
        )
        force_complex_z = force["force_real_z"][0] + 1j * force["force_imag_z"][0]
        np.testing.assert_allclose(force_complex_z, 2.0 * expected, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(
            force["force_amplitude_z"][0],
            2.0 * amplitude,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_streaming_summary_reads_each_cgns_file_once(self) -> None:
        sample_count = 8
        sample_index = np.arange(sample_count, dtype=float)
        pressure_rows = np.column_stack(
            [
                2.0 * np.cos(2.0 * np.pi * sample_index / sample_count),
                3.0 * np.sin(4.0 * np.pi * sample_index / sample_count),
            ]
        )
        paths = [Path(f"step_{index}.cgns") for index in range(sample_count)]
        row_by_path = {path: pressure_rows[index] for index, path in enumerate(paths)}
        geometry = SurfaceGeometry(
            coordinates=np.zeros((3, 3), dtype=float),
            faces=np.array([[0, 1, 2], [0, 2, 1]], dtype=int),
            centers=np.zeros((2, 3), dtype=float),
            area_vectors=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float),
            areas=np.ones(2, dtype=float),
            normals=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        )

        def read_full(_h5, path, _pressure_name, _selected_path=None):
            return "/Base/Zone/Pressure/data", row_by_path[Path(path)].copy()

        def read_slice(_h5, path, _pressure_name, start, end, _selected_path=None):
            return (
                "/Base/Zone/Pressure/data",
                row_by_path[Path(path)][start:end].copy(),
                pressure_rows.shape[1],
            )

        with (
            patch(
                "extract_cgns_pressure._read_pressure_vector_from_file",
                side_effect=read_full,
            ) as read_full_mock,
            patch(
                "extract_cgns_pressure._read_pressure_vector_slice_from_file",
                side_effect=read_slice,
            ) as read_slice_mock,
        ):
            streamed = compute_streaming_equivalent_force_summary(
                paths,
                geometry,
                dt=1.0 / sample_count,
                pressure_block_size=1,
                h5_module=object(),
            )

        reference = compute_equivalent_force_spectrum(
            compute_pressure_complex_spectrum(
                pressure_rows,
                dt=1.0 / sample_count,
                include_dc=False,
            ),
            geometry.area_vectors,
        )
        self.assertEqual(read_full_mock.call_count, sample_count)
        read_slice_mock.assert_not_called()
        self.assertEqual(list(Path("work/tmp").glob("starccm_pressure_*.dat")), [])
        for key in reference:
            np.testing.assert_allclose(streamed[key], reference[key], rtol=1.0e-12, atol=1.0e-12)

    def test_complex_spectrum_npz_records_normalized_value_convention(self) -> None:
        output_path = TEST_OUTPUT_DIR / "normalized_complex_spectrum_test.npz"
        spectrum = compute_pressure_complex_spectrum(
            np.cos(2.0 * np.pi * np.arange(8, dtype=float) / 8.0)[:, np.newaxis],
            dt=0.125,
        )

        write_pressure_complex_spectrum_npz(output_path, spectrum)

        with np.load(output_path) as saved:
            self.assertEqual(int(saved["complex_spectrum_schema_version"]), 2)
            self.assertEqual(
                str(saved["complex_value_convention"]),
                "single_sided_peak_amplitude",
            )

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
