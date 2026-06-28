import json
import gc
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

from map_cgns_pressure_to_inp import (
    FileSizeLimitError,
    apply_coordinate_transform,
    estimate_time_load_records,
    find_frequency_index,
    map_complex_pressure_to_nodes,
    parse_args,
    parse_inp_text,
    parse_frequency_text,
    resolve_requested_frequencies,
    run_mapping,
    select_target_faces,
    write_frequency_load_include,
)


def _unlink_test_file(path: Path) -> None:
    for _ in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.05)


class MapCgnsPressureToInpTests(unittest.TestCase):
    def tearDown(self) -> None:
        for name in (
            "map_test_model.inp",
            "map_test_model_mapped.inp",
            "map_test_model_mapped_100Hz.inp",
            "map_test_model_mapped_200Hz.inp",
            "map_test_model_mapped_loads.inc",
            "map_test_model_mapped_100Hz_loads.inc",
            "map_test_model_mapped_200Hz_loads.inc",
            "map_test_surface_geometry.npz",
            "map_test_pressure_complex_spectrum.npz",
            "mapping_report.json",
        ):
            _unlink_test_file(Path(name))

    def test_gitignore_covers_input_fixture_file(self) -> None:
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        self.assertIn("map_test_*.inp", gitignore)

    def test_parse_inp_nodes_elements_sets_and_step_kind(self) -> None:
        model = parse_inp_text(
            "\n".join(
                [
                    "*Heading",
                    "*Node",
                    "1, 0, 0, 0",
                    "2, 1, 0, 0",
                    "3, 1, 1, 0",
                    "4, 0, 1, 0",
                    "*Element, type=S4, elset=SURF",
                    "10, 1, 2, 3, 4",
                    "*Nset, nset=EDGE",
                    "1, 2",
                    "*Elset, elset=SURF_COPY",
                    "10",
                    "*Step, name=HARMONIC",
                    "*Steady State Dynamics",
                    "100., 200., 2",
                    "*End Step",
                ]
            )
        )

        self.assertEqual(model.nodes[3].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(model.elements[10].node_ids, [1, 2, 3, 4])
        self.assertEqual(model.elsets["SURF"], {10})
        self.assertEqual(model.elsets["SURF_COPY"], {10})
        self.assertEqual(model.nsets["EDGE"], {1, 2})
        self.assertEqual(model.steps[0].kind, "steady_state")

    def test_select_target_faces_from_elset_computes_area_vector(self) -> None:
        model = parse_inp_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0, 0, 0",
                    "2, 1, 0, 0",
                    "3, 1, 1, 0",
                    "4, 0, 1, 0",
                    "*Element, type=S4, elset=SURF",
                    "10, 1, 2, 3, 4",
                    "*Step",
                    "*Dynamic, Explicit",
                    "*End Step",
                ]
            )
        )

        faces = select_target_faces(model, "SURF", "elset")

        self.assertEqual(faces.node_ids, [[1, 2, 3, 4]])
        np.testing.assert_allclose(faces.centers, [[0.5, 0.5, 0.0]])
        np.testing.assert_allclose(faces.area_vectors, [[0.0, 0.0, 1.0]])

    def test_map_complex_pressure_to_nodes_conserves_total_force(self) -> None:
        target_centers = np.array([[0.5, 0.5, 0.0]], dtype=float)
        target_area_vectors = np.array([[0.0, 0.0, 2.0]], dtype=float)
        target_node_ids = [[1, 2, 3, 4]]
        source_centers = np.array([[0.5, 0.5, 0.0]], dtype=float)
        source_pressure = np.array([3.0 + 4.0j], dtype=complex)

        node_forces, stats = map_complex_pressure_to_nodes(
            target_centers,
            target_area_vectors,
            target_node_ids,
            source_centers,
            source_pressure,
            k=4,
        )

        total = sum(node_forces.values(), np.zeros(3, dtype=complex))
        np.testing.assert_allclose(total, [0.0 + 0.0j, 0.0 + 0.0j, -6.0 - 8.0j])
        np.testing.assert_allclose(node_forces[1], [0.0 + 0.0j, 0.0 + 0.0j, -1.5 - 2.0j])
        self.assertEqual(stats["target_face_count"], 1)

    def test_write_frequency_load_include_writes_real_and_imag_cloads(self) -> None:
        include_path = Path("map_test_model_mapped_loads.inc")
        node_forces = {
            2: np.array([1.0 + 10.0j, 0.0 + 0.0j, -2.0 - 20.0j]),
        }

        write_frequency_load_include(include_path, node_forces, frequency_hz=100.0)

        text = include_path.read_text(encoding="utf-8")
        self.assertIn("frequency_hz=100", text)
        self.assertIn("** Real part", text)
        self.assertIn("2, 1, 1", text)
        self.assertIn("2, 3, -2", text)
        self.assertIn("** Imaginary part", text)
        self.assertIn("2, 1, 10", text)
        self.assertIn("2, 3, -20", text)

    def test_run_mapping_generates_mapped_inp_include_and_report(self) -> None:
        inp_path = Path("map_test_model.inp")
        output_path = Path("map_test_model_mapped.inp")
        inp_path.write_text(
            "\n".join(
                [
                    "*Heading",
                    "*Node",
                    "1, 0, 0, 0",
                    "2, 1, 0, 0",
                    "3, 1, 1, 0",
                    "4, 0, 1, 0",
                    "*Element, type=S4, elset=SURF",
                    "10, 1, 2, 3, 4",
                    "*Step, name=HARMONIC",
                    "*Steady State Dynamics",
                    "100., 100., 1",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            "map_test_surface_geometry.npz",
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            coordinates=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2, 3]], dtype=int),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
            areas=np.array([1.0], dtype=float),
            normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            "map_test_pressure_complex_spectrum.npz",
            frequencies_hz=np.array([100.0], dtype=float),
            pressure_real=np.array([[8.0]], dtype=float),
            pressure_imag=np.array([[6.0]], dtype=float),
        )

        result = run_mapping(
            inp_path=inp_path,
            extracted_dir=Path("."),
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        self.assertEqual(result.output_inp_path, output_path)
        self.assertTrue(Path("map_test_model_mapped_loads.inc").exists())
        self.assertIn(
            "*INCLUDE, INPUT=map_test_model_mapped_loads.inc",
            output_path.read_text(encoding="utf-8"),
        )
        report = json.loads(Path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["step_kind"], "steady_state")
        self.assertEqual(report["target_set"], "SURF")

    def test_run_mapping_writes_one_inp_per_requested_frequency(self) -> None:
        inp_path = Path("map_test_model.inp")
        output_path = Path("map_test_model_mapped.inp")
        inp_path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0, 0, 0",
                    "2, 1, 0, 0",
                    "3, 1, 1, 0",
                    "4, 0, 1, 0",
                    "*Element, type=S4, elset=SURF",
                    "10, 1, 2, 3, 4",
                    "*Step, name=HARMONIC",
                    "*Steady State Dynamics",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            "map_test_surface_geometry.npz",
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            coordinates=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2, 3]], dtype=int),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
            areas=np.array([1.0], dtype=float),
            normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            "map_test_pressure_complex_spectrum.npz",
            frequencies_hz=np.array([100.0, 200.0], dtype=float),
            pressure_real=np.array([[8.0], [2.0]], dtype=float),
            pressure_imag=np.array([[6.0], [1.0]], dtype=float),
        )

        result = run_mapping(
            inp_path=inp_path,
            extracted_dir=Path("."),
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0, 200.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        self.assertEqual(
            result.output_inp_paths,
            [
                Path("map_test_model_mapped_100Hz.inp"),
                Path("map_test_model_mapped_200Hz.inp"),
            ],
        )
        self.assertTrue(Path("map_test_model_mapped_100Hz_loads.inc").exists())
        self.assertTrue(Path("map_test_model_mapped_200Hz_loads.inc").exists())
        self.assertIn(
            "map_test_model_mapped_200Hz_loads.inc",
            Path("map_test_model_mapped_200Hz.inp").read_text(encoding="utf-8"),
        )

    def test_frequency_outside_tolerance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No extracted frequency"):
            find_frequency_index(np.array([95.0, 105.0]), 100.0, tolerance_hz=1.0)

    def test_frequency_range_expands_inclusive_hz_values(self) -> None:
        args = parse_args(
            [
                "--inp",
                "model.inp",
                "--extracted",
                "out",
                "--target-set",
                "SURF",
                "--target-set-type",
                "elset",
                "--frequency-range",
                "1:800:1",
            ]
        )

        frequencies = resolve_requested_frequencies(args)

        self.assertEqual(len(frequencies), 800)
        self.assertEqual(frequencies[0], 1.0)
        self.assertEqual(frequencies[-1], 800.0)

    def test_frequency_range_combines_with_single_frequency_values(self) -> None:
        args = parse_args(
            [
                "--inp",
                "model.inp",
                "--extracted",
                "out",
                "--target-set",
                "SURF",
                "--target-set-type",
                "elset",
                "--frequency",
                "0.5",
                "--frequency-range",
                "1:3:1",
            ]
        )

        self.assertEqual(resolve_requested_frequencies(args), [0.5, 1.0, 2.0, 3.0])

    def test_gui_frequency_text_accepts_comma_values_and_ranges(self) -> None:
        self.assertEqual(parse_frequency_text("0.5, 1:3:1"), [0.5, 1.0, 2.0, 3.0])
        self.assertIsNone(parse_frequency_text(""))

    def test_missing_target_set_is_rejected(self) -> None:
        model = parse_inp_text("*Node\n1,0,0,0\n*Step\n*Dynamic\n*End Step")

        with self.assertRaisesRegex(ValueError, "Target elset 'SURF' was not found"):
            select_target_faces(model, "SURF", "elset")

    def test_time_load_record_limit_is_enforced(self) -> None:
        with self.assertRaises(FileSizeLimitError):
            estimate_time_load_records(
                node_count=200,
                component_count=3,
                sample_count=1000,
                max_records=10000,
            )

    def test_coordinate_transform_supports_scale_translation_and_axis_order(self) -> None:
        coordinates = np.array([[1.0, 2.0, 3.0]], dtype=float)

        transformed = apply_coordinate_transform(
            coordinates,
            scale=2.0,
            translate=np.array([10.0, 0.0, -1.0]),
            axis_order=(2, 0, 1),
            axis_sign=(1.0, -1.0, 1.0),
        )

        np.testing.assert_allclose(transformed, [[16.0, -2.0, 3.0]])

    def test_nearest_weight_lookup_uses_ckdtree_when_available(self) -> None:
        from starccm_pressure import map_cgns_pressure_to_inp as mapper

        calls: list[tuple[tuple[int, int], int]] = []

        class FakeTree:
            def __init__(self, points: np.ndarray) -> None:
                self.points = np.asarray(points, dtype=float)

            def query(self, targets: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
                calls.append((tuple(np.asarray(targets).shape), int(k)))
                distances = np.array([[0.0, 2.0]], dtype=float)
                indices = np.array([[1, 0]], dtype=int)
                return distances, indices

        target_centers = np.array([[1.0, 0.0, 0.0]], dtype=float)
        source_centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

        with patch.object(mapper, "_load_ckdtree", return_value=FakeTree):
            indices, weights = mapper._nearest_weights(target_centers, source_centers, k=2)

        self.assertEqual(calls, [((1, 3), 2)])
        np.testing.assert_array_equal(indices, [[1, 0]])
        np.testing.assert_allclose(weights, [[1.0, 0.0]])

    def test_mapping_report_documents_physical_mapping_assumptions(self) -> None:
        inp_path = Path("map_test_model.inp")
        output_path = Path("map_test_model_mapped.inp")
        inp_path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0, 0, 0",
                    "2, 1, 0, 0",
                    "3, 1, 1, 0",
                    "4, 0, 1, 0",
                    "*Element, type=S4, elset=SURF",
                    "10, 1, 2, 3, 4",
                    "*Step, name=HARMONIC",
                    "*Steady State Dynamics",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            "map_test_surface_geometry.npz",
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            coordinates=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2, 3]], dtype=int),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
            areas=np.array([1.0], dtype=float),
            normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            "map_test_pressure_complex_spectrum.npz",
            frequencies_hz=np.array([100.0], dtype=float),
            pressure_real=np.array([[8.0]], dtype=float),
            pressure_imag=np.array([[6.0]], dtype=float),
        )

        run_mapping(
            inp_path=inp_path,
            extracted_dir=Path("."),
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        report = json.loads(Path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["mapping_assumptions"]["pressure_sign"], "-pressure * area_vector")
        self.assertEqual(report["mapping_assumptions"]["node_force_distribution"], "equal_share_per_face_node")
        self.assertIn("supported_element_scope", report["mapping_assumptions"])


if __name__ == "__main__":
    unittest.main()
