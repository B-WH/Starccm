import ast
import json
import gc
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

from map_cgns_pressure_to_inp import (
    FileSizeLimitError,
    apply_global_conservation_correction,
    apply_coordinate_transform,
    _split_frequency_groups,
    build_alignment_preview,
    compute_node_force_moment,
    estimate_time_load_records,
    ensure_preprint_echo_off,
    find_frequency_index,
    map_complex_pressure_to_nodes,
    parse_args,
    parse_inp_text,
    parse_frequency_text,
    parse_gui_axis_order,
    parse_gui_axis_sign,
    parse_gui_translate,
    resolve_requested_frequencies,
    run_mapping,
    select_target_faces,
    transform_area_vectors,
    write_frequency_load_include,
    write_frequency_table_load_include,
)


TEST_OUTPUT_DIR = Path("work/test-output")


def _test_path(name: str) -> Path:
    return TEST_OUTPUT_DIR / name


def _unlink_test_file(path: Path) -> None:
    for _ in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.05)


def _read_real_cloads(path: Path) -> dict[int, np.ndarray]:
    forces: dict[int, np.ndarray] = {}
    in_real = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("*CLOAD, REAL"):
            in_real = True
            continue
        if stripped.upper().startswith("*CLOAD, IMAGINARY"):
            break
        if not in_real or not stripped or stripped.startswith("**"):
            continue
        node_text, component_text, value_text = [part.strip() for part in stripped.split(",")]
        node_id = int(node_text)
        component = int(component_text) - 1
        forces.setdefault(node_id, np.zeros(3, dtype=float))[component] = float(value_text)
    return forces


class MapCgnsPressureToInpTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for name in (
            "map_test_model.inp",
            "map_test_model_mapped.inp",
            "map_test_model_mapped_100Hz.inp",
            "map_test_model_mapped_200Hz.inp",
            "map_test_model_mapped_loads.inc",
            "map_test_model_mapped_100Hz_loads.inc",
            "map_test_model_mapped_200Hz_loads.inc",
            "map_test_model_mapped_g001_100Hz-200Hz.inp",
            "map_test_model_mapped_g001_100Hz-200Hz_loads.inc",
            "map_test_model_mapped_g002_300Hz-400Hz.inp",
            "map_test_model_mapped_g002_300Hz-400Hz_loads.inc",
            "map_test_model_mapped_batch_loads.inc",
            "map_test_surface_geometry.npz",
            "map_test_pressure_complex_spectrum.npz",
            "map_test_near_zero_pressure_complex_spectrum.npz",
            "map_test_near_zero_surface_geometry.npz",
            "mapping_report.json",
            "map_test_near_zero_model.inp",
            "map_test_near_zero_model_mapped.inp",
            "map_test_near_zero_model_mapped_loads.inc",
        ):
            _unlink_test_file(_test_path(name))

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

    def test_select_target_faces_uses_instance_part_mesh_when_ids_overlap(self) -> None:
        model = parse_inp_text(
            "\n".join(
                [
                    "*Part, name=Hull",
                    "*Node",
                    "1, 0, 0, 0",
                    "2, 1, 0, 0",
                    "3, 1, 1, 0",
                    "4, 0, 1, 0",
                    "*Element, type=S4",
                    "10, 1, 2, 3, 4",
                    "*End Part",
                    "*Part, name=Other",
                    "*Node",
                    "1, 0, 0, 10",
                    "2, 2, 0, 10",
                    "3, 2, 2, 10",
                    "4, 0, 2, 10",
                    "*Element, type=S4",
                    "10, 1, 2, 3, 4",
                    "*End Part",
                    "*Assembly, name=Assembly",
                    "*Instance, name=Hull-1, part=Hull",
                    "*End Instance",
                    "*Instance, name=Other-1, part=Other",
                    "*End Instance",
                    "*Elset, elset=SURF, instance=Hull-1",
                    "10",
                    "*End Assembly",
                    "*Step",
                    "*Dynamic, Explicit",
                    "*End Step",
                ]
            )
        )

        faces = select_target_faces(model, "SURF", "elset")

        np.testing.assert_allclose(faces.centers, [[0.5, 0.5, 0.0]])
        np.testing.assert_allclose(faces.area_vectors, [[0.0, 0.0, 1.0]])

    def test_transform_area_vectors_scales_area_by_square_of_coordinate_scale(self) -> None:
        transformed = transform_area_vectors(
            np.array([[0.0, 0.0, 1.0]], dtype=float),
            scale=2.0,
        )

        np.testing.assert_allclose(transformed, [[0.0, 0.0, 4.0]])

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

    def test_map_complex_pressure_to_nodes_uses_consistent_quad_nodal_forces(self) -> None:
        target_centers = np.array([[0.5, 0.5, 0.0]], dtype=float)
        target_area_vectors = np.array([[0.0, 0.0, 1.0]], dtype=float)
        target_node_ids = [[1, 2, 3, 4]]
        target_face_points = [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
        ]
        gauss = 1.0 / np.sqrt(3.0)
        natural_points = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]
        source_centers = np.array(
            [[0.5 * (1.0 + xi), 0.5 * (1.0 + eta), 0.0] for xi, eta in natural_points],
            dtype=float,
        )
        source_pressure = source_centers[:, 0].astype(complex)

        node_forces, stats = map_complex_pressure_to_nodes(
            target_centers,
            target_area_vectors,
            target_node_ids,
            source_centers,
            source_pressure,
            k=1,
            target_face_points=target_face_points,
        )

        np.testing.assert_allclose(node_forces[1], [0.0, 0.0, -1.0 / 12.0])
        np.testing.assert_allclose(node_forces[2], [0.0, 0.0, -1.0 / 6.0])
        np.testing.assert_allclose(node_forces[3], [0.0, 0.0, -1.0 / 6.0])
        np.testing.assert_allclose(node_forces[4], [0.0, 0.0, -1.0 / 12.0])
        np.testing.assert_allclose(
            sum(node_forces.values(), np.zeros(3, dtype=complex)),
            [0.0, 0.0, -0.5],
        )
        self.assertEqual(stats["integration_point_count"], 4)

    def test_consistent_force_plan_batch_matches_single_frequency_scatter(self) -> None:
        from starccm_pressure import map_cgns_pressure_to_inp as mapper

        face_points = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        ]
        node_ids = [[1, 2, 3], [2, 4, 3]]
        source_centers = np.vstack(
            [
                location
                for points in face_points
                for _, location, _ in mapper._tri3_quadrature(points)
            ]
        )
        plan = mapper.build_consistent_force_plan(face_points, node_ids, source_centers, k=1)
        pressure_batch = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=float,
        )

        batch_forces, batch_node_ids, _ = mapper.apply_consistent_force_plan_batch(
            plan,
            pressure_batch,
        )

        for row_index, pressures in enumerate(pressure_batch):
            single_forces, _ = mapper.apply_consistent_force_plan(plan, pressures)
            for node_index, node_id in enumerate(batch_node_ids):
                np.testing.assert_allclose(
                    batch_forces[row_index, node_index],
                    single_forces[int(node_id)],
                )

    def test_global_conservation_correction_matches_force_and_moment(self) -> None:
        node_forces = {
            1: np.array([0.0 + 0.0j, 0.0 + 0.0j, -0.5 - 0.25j]),
            2: np.array([0.0 + 0.0j, 0.0 + 0.0j, -0.5 - 0.25j]),
            3: np.array([0.0 + 0.0j, 0.0 + 0.0j, -0.5 - 0.25j]),
            4: np.array([0.0 + 0.0j, 0.0 + 0.0j, -0.5 - 0.25j]),
        }
        node_coordinates = {
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([1.0, 1.0, 0.0]),
            4: np.array([0.0, 1.0, 0.0]),
        }
        desired_force = np.array([1.0 + 2.0j, -0.5 + 0.25j, -4.0 - 1.0j])
        desired_moment = np.array([-2.0 - 0.5j, 2.0 + 1.0j, 0.75 - 0.25j])

        corrected, stats = apply_global_conservation_correction(
            node_forces,
            node_coordinates,
            desired_force,
            desired_moment,
        )

        force, moment = compute_node_force_moment(corrected, node_coordinates)
        np.testing.assert_allclose(force, desired_force, atol=1.0e-12)
        np.testing.assert_allclose(moment, desired_moment, atol=1.0e-12)
        self.assertGreater(stats["force_residual_norm_before"], 0.0)
        self.assertLess(stats["force_residual_norm_after"], 1.0e-12)
        self.assertLess(stats["moment_residual_norm_after"], 1.0e-12)

    def test_run_mapping_applies_global_force_and_moment_conservation(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            coordinates=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2]], dtype=int),
            area_vectors=np.array([[0.0, 0.0, 2.0]], dtype=float),
            areas=np.array([2.0], dtype=float),
            normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0], dtype=float),
            pressure_real=np.array([[5.0]], dtype=float),
            pressure_imag=np.array([[0.0]], dtype=float),
        )

        run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        node_coordinates = {
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([1.0, 1.0, 0.0]),
            4: np.array([0.0, 1.0, 0.0]),
        }
        force, moment = compute_node_force_moment(
            _read_real_cloads(_test_path("map_test_model_mapped_loads.inc")),
            node_coordinates,
        )
        np.testing.assert_allclose(force, [0.0, 0.0, -10.0], atol=1.0e-10)
        np.testing.assert_allclose(moment, [-5.0, 5.0, 0.0], atol=1.0e-10)
        report = json.loads(_test_path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["mapping_stats"]["conservation_enabled"], 1)
        self.assertLess(report["mapping_stats"]["force_residual_norm_after"], 1.0e-10)
        self.assertLess(report["mapping_stats"]["moment_residual_norm_after"], 1.0e-10)

    def test_write_frequency_load_include_writes_real_and_imag_cloads(self) -> None:
        include_path = _test_path("map_test_model_mapped_loads.inc")
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

    def test_preprint_echo_is_forced_off_before_step(self) -> None:
        lines = [
            "*Heading",
            "*Preprint, echo=YES, model=YES, history=YES",
            "*Node",
            "1, 0, 0, 0",
            "*Step",
            "*Steady State Dynamics",
            "*End Step",
        ]

        updated = ensure_preprint_echo_off(lines)

        self.assertIn("*Preprint, echo=NO, model=NO, history=NO", updated)
        self.assertNotIn("*Preprint, echo=YES, model=YES, history=YES", updated)
        self.assertLess(
            updated.index("*Preprint, echo=NO, model=NO, history=NO"),
            updated.index("*Step"),
        )

    def test_run_mapping_generates_mapped_inp_include_and_report(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            coordinates=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2, 3]], dtype=int),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
            areas=np.array([1.0], dtype=float),
            normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0], dtype=float),
            pressure_real=np.array([[8.0]], dtype=float),
            pressure_imag=np.array([[6.0]], dtype=float),
        )

        result = run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        self.assertEqual(result.output_inp_path, output_path)
        self.assertTrue(_test_path("map_test_model_mapped_loads.inc").exists())
        mapped_text = output_path.read_text(encoding="utf-8")
        self.assertIn("*Preprint, echo=NO, model=NO, history=NO", mapped_text)
        self.assertIn(
            "*INCLUDE, INPUT=map_test_model_mapped_loads.inc",
            mapped_text,
        )
        report = json.loads(_test_path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["step_kind"], "steady_state")
        self.assertEqual(report["target_set"], "SURF")

    def test_run_mapping_prefixes_cload_nodes_for_assembly_instance_set(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
        inp_path.write_text(
            "\n".join(
                [
                    "*Heading",
                    "*Part, name=Hull",
                    "*Node",
                    "1, 0, 0, 0",
                    "2, 1, 0, 0",
                    "3, 1, 1, 0",
                    "4, 0, 1, 0",
                    "*Element, type=S4",
                    "10, 1, 2, 3, 4",
                    "*End Part",
                    "*Assembly, name=Assembly",
                    "*Instance, name=Hull-1, part=Hull",
                    "*End Instance",
                    "*Elset, elset=SURF, instance=Hull-1",
                    "10",
                    "*End Assembly",
                    "*Step, name=HARMONIC",
                    "*Steady State Dynamics",
                    "100., 100., 1",
                    "*End Step",
                ]
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0], dtype=float),
            pressure_real=np.array([[8.0]], dtype=float),
            pressure_imag=np.array([[6.0]], dtype=float),
        )

        run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        include_text = _test_path("map_test_model_mapped_loads.inc").read_text(
            encoding="utf-8"
        )
        self.assertIn("Hull-1.1, 3, -2", include_text)
        self.assertNotIn("\n1, 3, -2", include_text)
        report = json.loads(_test_path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["load_node_label_prefix"], "Hull-1.")

    def test_run_mapping_writes_all_requested_frequencies_into_one_inp(self) -> None:
        _unlink_test_file(_test_path("map_test_model_mapped_100Hz_loads.inc"))
        _unlink_test_file(_test_path("map_test_model_mapped_200Hz_loads.inc"))
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            coordinates=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2, 3]], dtype=int),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
            areas=np.array([1.0], dtype=float),
            normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0, 200.0], dtype=float),
            pressure_real=np.array([[8.0], [2.0]], dtype=float),
            pressure_imag=np.array([[6.0], [1.0]], dtype=float),
        )

        result = run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0, 200.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        self.assertEqual(result.output_inp_paths, [output_path])
        self.assertEqual(result.output_inp_path, output_path)
        self.assertFalse(_test_path("map_test_model_mapped_100Hz.inp").exists())
        self.assertFalse(_test_path("map_test_model_mapped_200Hz.inp").exists())
        self.assertEqual(result.include_paths, [_test_path("map_test_model_mapped_loads.inc")])
        mapped_text = output_path.read_text(encoding="utf-8")
        self.assertEqual(mapped_text.count("*Step"), 1)
        self.assertIn("*INCLUDE, INPUT=map_test_model_mapped_loads.inc", mapped_text)
        include_text = _test_path("map_test_model_mapped_loads.inc").read_text(encoding="utf-8")
        self.assertIn("*Amplitude, name=CGNS_R_N1_D3, definition=TABULAR", include_text)
        self.assertIn("100, -2", include_text)
        self.assertIn("200, -0.5", include_text)
        self.assertIn("*CLOAD, REAL, amplitude=CGNS_R_N1_D3", include_text)
        self.assertIn("*CLOAD, IMAGINARY, amplitude=CGNS_I_N1_D3", include_text)

    def test_run_mapping_reuses_nearest_weights_for_multiple_frequencies(self) -> None:
        from starccm_pressure import map_cgns_pressure_to_inp as mapper

        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0, 200.0], dtype=float),
            pressure_real=np.array([[8.0, 4.0], [2.0, 1.0]], dtype=float),
            pressure_imag=np.array([[6.0, 3.0], [1.0, 0.5]], dtype=float),
        )
        original_nearest_weights = mapper._nearest_weights
        calls: list[tuple[int, int]] = []

        def counting_nearest_weights(
            target_centers: np.ndarray,
            source_centers: np.ndarray,
            k: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            calls.append((target_centers.shape[0], source_centers.shape[0]))
            return original_nearest_weights(target_centers, source_centers, k)

        with (
            patch.object(mapper, "_nearest_weights", side_effect=counting_nearest_weights),
            patch.object(
                mapper,
                "map_complex_pressure_to_nodes",
                side_effect=AssertionError("steady-state batch path should be used"),
            ),
        ):
            run_mapping(
                inp_path=inp_path,
                extracted_dir=TEST_OUTPUT_DIR,
                target_set="SURF",
                target_set_type="elset",
                frequencies=[100.0, 200.0],
                output_path=output_path,
                surface_geometry_name="map_test_surface_geometry.npz",
                complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
            )

        self.assertEqual(calls, [(4, 2)])

    def test_run_mapping_streams_requested_frequency_rows(self) -> None:
        from starccm_pressure import map_cgns_pressure_to_inp as mapper

        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0, 200.0, 300.0], dtype=float),
            pressure_real=np.array([[99.0], [8.0], [2.0]], dtype=float),
            pressure_imag=np.array([[99.0], [6.0], [1.0]], dtype=float),
        )

        with patch.object(
            mapper,
            "_load_complex_spectrum",
            side_effect=AssertionError("full complex spectrum should not be loaded"),
        ):
            run_mapping(
                inp_path=inp_path,
                extracted_dir=TEST_OUTPUT_DIR,
                target_set="SURF",
                target_set_type="elset",
                frequencies=[200.0, 300.0],
                output_path=output_path,
                surface_geometry_name="map_test_surface_geometry.npz",
                complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
                frequency_batch_size=1,
                show_progress=False,
            )

        include_text = _test_path("map_test_model_mapped_loads.inc").read_text(encoding="utf-8")
        self.assertIn("200, -2", include_text)
        self.assertIn("300, -0.5", include_text)
        self.assertNotIn("100,", include_text)

    def test_run_mapping_reports_progress_callback_for_steady_state_batches(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0, 200.0], dtype=float),
            pressure_real=np.array([[8.0], [2.0]], dtype=float),
            pressure_imag=np.array([[6.0], [1.0]], dtype=float),
        )
        events: list[dict[str, object]] = []

        run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0, 200.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
            frequency_batch_size=1,
            show_progress=False,
            progress_callback=events.append,
        )

        self.assertGreaterEqual(len(events), 4)
        self.assertEqual(events[-1]["current"], events[-1]["total"])
        self.assertTrue(any("频率块 1/2" in str(event["message"]) for event in events))

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

    def test_gui_transform_text_parsers_accept_defaults_and_reject_bad_triplets(self) -> None:
        self.assertIsNone(parse_gui_translate(""))
        np.testing.assert_allclose(parse_gui_translate("1, 2, -3"), [1.0, 2.0, -3.0])
        self.assertEqual(parse_gui_axis_order("2,0,1"), (2, 0, 1))
        self.assertEqual(parse_gui_axis_sign("1,-1,1"), (1.0, -1.0, 1.0))

        with self.assertRaisesRegex(ValueError, "Expected three comma-separated values"):
            parse_gui_translate("1,2")

    def test_build_alignment_preview_applies_transform_and_reports_distances(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        inp_path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 15.5, -2.5, 3",
                    "2, 16.5, -2.5, 3",
                    "3, 16.5, -1.5, 3",
                    "4, 15.5, -1.5, 3",
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        )

        preview = build_alignment_preview(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            scale=2.0,
            translate=np.array([10.0, 0.0, -1.0]),
            axis_order=(2, 0, 1),
            axis_sign=(1.0, -1.0, 1.0),
            surface_geometry_name="map_test_surface_geometry.npz",
        )

        np.testing.assert_allclose(preview.source_centers, [[16.0, -2.0, 3.0], [16.0, -4.0, 3.0]])
        np.testing.assert_allclose(preview.target_centers, [[16.0, -2.0, 3.0]])
        self.assertEqual(preview.source_count, 2)
        self.assertEqual(preview.target_face_count, 1)
        self.assertEqual(preview.target_node_count, 4)
        self.assertEqual(preview.nearest_distance_min, 0.0)
        self.assertEqual(preview.nearest_distance_mean, 1.0)
        self.assertEqual(preview.nearest_distance_max, 2.0)

    def test_build_alignment_preview_samples_large_distance_check(self) -> None:
        from starccm_pressure import map_cgns_pressure_to_inp as mapper

        inp_path = _test_path("map_test_model.inp")
        inp_path.write_text(
            "\n".join(
                [
                    "*Node",
                    "1, 0, 0, 0",
                    "2, 1, 0, 0",
                    "3, 0, 1, 0",
                    "*Element, type=S3, elset=SURF",
                    "10, 1, 2, 3",
                ]
            ),
            encoding="utf-8",
        )
        centers = np.column_stack(
            [
                np.arange(20, dtype=float),
                np.zeros(20, dtype=float),
                np.zeros(20, dtype=float),
            ]
        )
        np.savez_compressed(
            _test_path("map_test_surface_geometry.npz"),
            centers=centers,
            area_vectors=np.tile([[0.0, 0.0, 1.0]], (20, 1)),
        )
        calls: list[tuple[int, int]] = []

        def fake_nearest_distances(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
            calls.append((source_points.shape[0], target_points.shape[0]))
            return np.zeros(source_points.shape[0], dtype=float)

        with patch.object(mapper, "_nearest_distances", side_effect=fake_nearest_distances):
            preview = mapper.build_alignment_preview(
                inp_path=inp_path,
                extracted_dir=TEST_OUTPUT_DIR,
                target_set="SURF",
                target_set_type="elset",
                surface_geometry_name="map_test_surface_geometry.npz",
                distance_sample_limit=5,
            )

        self.assertEqual(preview.source_count, 20)
        self.assertEqual(calls, [(5, 1)])

    def test_nearest_weight_lookup_uses_ckdtree_when_available(self) -> None:
        from starccm_pressure import map_cgns_pressure_to_inp as mapper

        calls: list[tuple[tuple[int, int], int, int | None]] = []

        class FakeTree:
            def __init__(self, points: np.ndarray) -> None:
                self.points = np.asarray(points, dtype=float)

            def query(
                self,
                targets: np.ndarray,
                k: int,
                workers: int | None = None,
            ) -> tuple[np.ndarray, np.ndarray]:
                calls.append((tuple(np.asarray(targets).shape), int(k), workers))
                distances = np.array([[0.0, 2.0]], dtype=float)
                indices = np.array([[1, 0]], dtype=int)
                return distances, indices

        target_centers = np.array([[1.0, 0.0, 0.0]], dtype=float)
        source_centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

        with patch.object(mapper, "_load_ckdtree", return_value=FakeTree):
            indices, weights = mapper._nearest_weights(target_centers, source_centers, k=2)

        self.assertEqual(calls, [((1, 3), 2, -1)])
        np.testing.assert_array_equal(indices, [[1, 0]])
        np.testing.assert_allclose(weights, [[1.0, 0.0]])

    def test_optional_scipy_import_is_not_static_for_pylance(self) -> None:
        source = Path("starccm_pressure/map_cgns_pressure_to_inp.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]

        self.assertNotIn("scipy.spatial", imported_modules)

    def test_mapping_gui_formats_common_errors_in_chinese(self) -> None:
        from starccm_pressure.mapping_gui import format_gui_error, validate_mapping_gui_inputs

        with self.assertRaisesRegex(ValueError, "请选择 INP 文件"):
            validate_mapping_gui_inputs("", "cgns_pressure_output", "SURF")
        with self.assertRaisesRegex(ValueError, "请选择 CGNS 提取结果目录"):
            validate_mapping_gui_inputs("model.inp", "", "SURF")
        with self.assertRaisesRegex(ValueError, "请输入 Abaqus 目标集合名称"):
            validate_mapping_gui_inputs("model.inp", "cgns_pressure_output", "")

        self.assertIn(
            "找不到 surface_geometry.npz",
            format_gui_error(
                FileNotFoundError(
                    "Required surface geometry file was not found: cgns_pressure_output\\surface_geometry.npz"
                )
            ),
        )
        self.assertIn(
            "找不到目标 elset",
            format_gui_error(ValueError("Target elset 'SURF' was not found.")),
        )
        self.assertIn(
            "轴顺序必须是 0,1,2 的排列",
            format_gui_error(ValueError("axis_order must be a permutation of 0, 1, 2.")),
        )

    def test_mapping_gui_wires_progressbar_to_mapping_callback(self) -> None:
        source = Path("starccm_pressure/mapping_gui.py").read_text(encoding="utf-8")

        self.assertIn("ttk.Progressbar", source)
        self.assertIn("progress_callback=update_progress", source)

    def test_mapping_gui_wires_relative_zero_tolerance_to_mapping(self) -> None:
        source = Path("starccm_pressure/mapping_gui.py").read_text(encoding="utf-8")

        self.assertIn('relative_zero_tolerance_var = tk.StringVar(value="1e-6")', source)
        self.assertIn(
            'relative_zero_tolerance = float(relative_zero_tolerance_var.get().strip() or "1e-6")',
            source,
        )
        self.assertIn("relative_zero_tolerance=relative_zero_tolerance", source)

    def test_mapping_gui_wires_frequency_grouping_to_mapping(self) -> None:
        source = Path("starccm_pressure/mapping_gui.py").read_text(encoding="utf-8")

        self.assertIn('frequency_group_mode_var = tk.StringVar(value="none")', source)
        self.assertIn('values=("none", "groups", "bandwidth")', source)
        self.assertIn("frequency_group_value=frequency_group_value", source)

    def test_mapping_report_documents_physical_mapping_assumptions(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            coordinates=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2, 3]], dtype=int),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
            areas=np.array([1.0], dtype=float),
            normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0], dtype=float),
            pressure_real=np.array([[8.0]], dtype=float),
            pressure_imag=np.array([[6.0]], dtype=float),
        )

        run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        report = json.loads(_test_path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["mapping_assumptions"]["pressure_sign"], "-pressure * area_vector")
        self.assertEqual(
            report["mapping_assumptions"]["node_force_distribution"],
            "consistent_shape_function_integration",
        )
        self.assertEqual(
            report["mapping_assumptions"]["global_conservation"],
            "minimum_norm_total_force_and_moment_correction",
        )
        self.assertIn("supported_element_scope", report["mapping_assumptions"])

    # ── 近零力过滤与输出统计 ──────────────────────────────────────

    def test_near_zero_components_are_filtered_from_frequency_table_include(
        self,
    ) -> None:
        node_force_maps: list[dict[int, np.ndarray]] = [
            {
                1: np.array([1.0 + 0.0j, 0.0 + 0.0j, -2.0 + 0.0j]),
                2: np.array([1.0e-15 + 0.0j, 1.0e-14 + 0.0j, 0.0 + 0.0j]),
            },
            {
                1: np.array([1.5 + 0.0j, 0.0 + 0.0j, -3.0 + 0.0j]),
                2: np.array([2.0e-15 + 0.0j, 0.0 + 0.0j, 1.0e-16 + 0.0j]),
            },
        ]
        frequencies_hz = [100.0, 200.0]

        stats = write_frequency_table_load_include(
            _test_path("map_test_model_mapped_loads.inc"),
            node_force_maps,
            frequencies_hz,
            relative_zero_tolerance=1.0e-12,
        )

        text = _test_path("map_test_model_mapped_loads.inc").read_text(encoding="utf-8")
        # 节点 1 方向 1 和 3 应该保留（力幅值 ~3.0）
        self.assertIn("CGNS_R_N1_D1", text)
        self.assertIn("CGNS_R_N1_D3", text)
        # 节点 2 的近零分量应该被过滤（力幅值 ~2e-15，相对全局最大~3.0）
        # 不应出现节点 2 的 amplitude
        self.assertNotIn("CGNS_R_N2", text)
        self.assertNotIn("CGNS_I_N2", text)
        # 统计字段应正确
        self.assertEqual(stats["frequency_count"], 2)
        self.assertGreater(stats["load_table_count"], 0)
        self.assertEqual(stats["active_cload_count"], stats["load_table_count"])
        self.assertGreater(stats["skipped_near_zero_components"], 0)
        self.assertGreater(stats["global_max_abs_force"], 0.0)
        self.assertEqual(stats["relative_zero_tolerance"], 1.0e-12)

    def test_write_frequency_table_load_include_handles_zero_global_force(
        self,
    ) -> None:
        node_force_maps: list[dict[int, np.ndarray]] = [
            {1: np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])},
            {1: np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])},
        ]
        frequencies_hz = [100.0, 200.0]

        stats = write_frequency_table_load_include(
            _test_path("map_test_model_mapped_loads.inc"),
            node_force_maps,
            frequencies_hz,
        )

        text = _test_path("map_test_model_mapped_loads.inc").read_text(encoding="utf-8")
        self.assertIn("No nonzero steady-state loads", text)
        self.assertEqual(stats["global_max_abs_force"], 0.0)
        self.assertEqual(stats["active_cload_count"], 0)

    def test_frequency_table_output_stats_in_mapping_report(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0, 200.0], dtype=float),
            pressure_real=np.array([[8.0], [2.0]], dtype=float),
            pressure_imag=np.array([[6.0], [1.0]], dtype=float),
        )

        run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0, 200.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        report = json.loads(_test_path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertIn("frequency_table_output", report)
        ft = report["frequency_table_output"]
        for key in (
            "frequency_count",
            "load_table_count",
            "active_cload_count",
            "skipped_near_zero_components",
            "global_max_abs_force",
            "relative_zero_tolerance",
        ):
            self.assertIn(key, ft)
        self.assertEqual(ft["frequency_count"], 2)
        self.assertGreater(ft["load_table_count"], 0)
        self.assertIn("include_file_count", report)
        self.assertIn("output_inp_count", report)
        self.assertEqual(report["include_file_count"], 1)
        self.assertEqual(report["output_inp_count"], 1)

    def test_frequency_batch_size_produces_same_content_as_default(self) -> None:
        from starccm_pressure import map_cgns_pressure_to_inp as mapper

        inp_text = "\n".join(
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
        )
        centers = np.array([[0.5, 0.5, 0.0]], dtype=float)
        area_vectors = np.array([[0.0, 0.0, 1.0]], dtype=float)
        frequencies_hz = np.array([100.0, 200.0, 300.0], dtype=float)
        pressure_real = np.array([[8.0], [2.0], [3.0]], dtype=float)
        pressure_imag = np.array([[6.0], [1.0], [0.5]], dtype=float)

        # 默认 batch（一次性处理全部频率）
        np.savez_compressed(
            _test_path("map_test_surface_geometry.npz"),
            centers=centers,
            area_vectors=area_vectors,
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=frequencies_hz,
            pressure_real=pressure_real,
            pressure_imag=pressure_imag,
        )
        _test_path("map_test_model.inp").write_text(inp_text, encoding="utf-8")
        run_mapping(
            inp_path=_test_path("map_test_model.inp"),
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0, 200.0, 300.0],
            output_path=_test_path("map_test_model_mapped.inp"),
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )
        default_text = _test_path("map_test_model_mapped_loads.inc").read_text(
            encoding="utf-8"
        )

        # 清理第一次运行的输出
        _unlink_test_file(_test_path("map_test_model_mapped.inp"))
        _unlink_test_file(_test_path("map_test_model_mapped_loads.inc"))
        _unlink_test_file(_test_path("mapping_report.json"))

        # batch_size=1 逐频率处理
        run_mapping(
            inp_path=_test_path("map_test_model.inp"),
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0, 200.0, 300.0],
            output_path=_test_path("map_test_model_mapped.inp"),
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
            frequency_batch_size=1,
        )
        batch_text = _test_path("map_test_model_mapped_loads.inc").read_text(
            encoding="utf-8"
        )

        # 比较时忽略注释行（可能包含不同的元数据）
        def _load_entries(text: str) -> set[str]:
            entries: set[str] = set()
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("**") or not stripped:
                    continue
                entries.add(stripped)
            return entries

        self.assertEqual(_load_entries(default_text), _load_entries(batch_text))

    def test_single_frequency_report_has_output_stats(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0], dtype=float),
            pressure_real=np.array([[8.0]], dtype=float),
            pressure_imag=np.array([[6.0]], dtype=float),
        )

        run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        report = json.loads(_test_path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertIn("include_file_count", report)
        self.assertIn("output_inp_count", report)
        self.assertEqual(report["include_file_count"], 1)
        self.assertEqual(report["output_inp_count"], 1)

    def test_cli_accepts_frequency_batch_size_and_relative_tolerance(self) -> None:
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
                "100",
                "--frequency-batch-size",
                "10",
                "--relative-zero-tolerance",
                "1e-10",
                "--frequency-group-mode",
                "groups",
                "--frequency-group-value",
                "2",
            ]
        )
        self.assertEqual(args.frequency_batch_size, 10)
        self.assertEqual(args.relative_zero_tolerance, 1e-10)
        self.assertEqual(args.frequency_group_mode, "groups")
        self.assertEqual(args.frequency_group_value, 2.0)

    def test_split_frequency_groups_supports_total_groups_and_bandwidth(self) -> None:
        frequencies = [100.0, 200.0, 300.0, 400.0]

        self.assertEqual(_split_frequency_groups(frequencies, "groups", 2), [(0, 2), (2, 4)])
        self.assertEqual(
            _split_frequency_groups([100.0, 200.0, 300.0, 400.0, 500.0], "groups", 2),
            [(0, 3), (3, 5)],
        )
        self.assertEqual(
            _split_frequency_groups(frequencies, "bandwidth", 150.0),
            [(0, 2), (2, 4)],
        )

    def test_run_mapping_groups_frequency_outputs_by_total_group_count(self) -> None:
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0, 200.0, 300.0, 400.0], dtype=float),
            pressure_real=np.array([[8.0], [2.0], [3.0], [4.0]], dtype=float),
            pressure_imag=np.array([[6.0], [1.0], [0.5], [0.25]], dtype=float),
        )

        result = run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0, 200.0, 300.0, 400.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
            frequency_group_mode="groups",
            frequency_group_value=2,
        )

        expected_outputs = [
            _test_path("map_test_model_mapped_g001_100Hz-200Hz.inp"),
            _test_path("map_test_model_mapped_g002_300Hz-400Hz.inp"),
        ]
        expected_includes = [
            _test_path("map_test_model_mapped_g001_100Hz-200Hz_loads.inc"),
            _test_path("map_test_model_mapped_g002_300Hz-400Hz_loads.inc"),
        ]
        self.assertEqual(result.output_inp_paths, expected_outputs)
        self.assertEqual(result.include_paths, expected_includes)
        self.assertIn(
            "*INCLUDE, INPUT=map_test_model_mapped_g001_100Hz-200Hz_loads.inc",
            expected_outputs[0].read_text(encoding="utf-8"),
        )
        first_include = expected_includes[0].read_text(encoding="utf-8")
        second_include = expected_includes[1].read_text(encoding="utf-8")
        self.assertIn("100, -2", first_include)
        self.assertIn("200, -0.5", first_include)
        self.assertNotIn("300,", first_include)
        self.assertIn("300, -0.75", second_include)
        self.assertIn("400, -1", second_include)
        report = json.loads(_test_path("mapping_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["output_inp_count"], 2)
        self.assertEqual(report["include_file_count"], 2)
        self.assertEqual(
            [group["frequency_count"] for group in report["frequency_groups"]],
            [2, 2],
        )

    def test_multi_frequency_only_creates_one_include_with_amplitudes(
        self,
    ) -> None:
        """多频率只生成一个 loads.inc 且包含 amplitude 表定义。"""
        _unlink_test_file(_test_path("map_test_model_mapped_100Hz_loads.inc"))
        _unlink_test_file(_test_path("map_test_model_mapped_200Hz_loads.inc"))
        _unlink_test_file(_test_path("map_test_model_mapped_300Hz_loads.inc"))
        inp_path = _test_path("map_test_model.inp")
        output_path = _test_path("map_test_model_mapped.inp")
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
            _test_path("map_test_surface_geometry.npz"),
            centers=np.array([[0.5, 0.5, 0.0]], dtype=float),
            area_vectors=np.array([[0.0, 0.0, 1.0]], dtype=float),
        )
        np.savez_compressed(
            _test_path("map_test_pressure_complex_spectrum.npz"),
            frequencies_hz=np.array([100.0, 200.0, 300.0], dtype=float),
            pressure_real=np.array([[8.0], [2.0], [3.0]], dtype=float),
            pressure_imag=np.array([[6.0], [1.0], [0.5]], dtype=float),
        )

        result = run_mapping(
            inp_path=inp_path,
            extracted_dir=TEST_OUTPUT_DIR,
            target_set="SURF",
            target_set_type="elset",
            frequencies=[100.0, 200.0, 300.0],
            output_path=output_path,
            surface_geometry_name="map_test_surface_geometry.npz",
            complex_spectrum_name="map_test_pressure_complex_spectrum.npz",
        )

        # 只有一个输出 INP
        self.assertEqual(result.output_inp_paths, [output_path])
        # 只有一个 include
        self.assertEqual(
            result.include_paths,
            [_test_path("map_test_model_mapped_loads.inc")],
        )
        # 没有每频率的独立 INP 或 include
        self.assertFalse(_test_path("map_test_model_mapped_100Hz.inp").exists())
        self.assertFalse(_test_path("map_test_model_mapped_200Hz.inp").exists())
        self.assertFalse(_test_path("map_test_model_mapped_300Hz.inp").exists())
        self.assertFalse(
            _test_path("map_test_model_mapped_100Hz_loads.inc").exists()
        )
        self.assertFalse(
            _test_path("map_test_model_mapped_300Hz_loads.inc").exists()
        )

        include_text = _test_path("map_test_model_mapped_loads.inc").read_text(
            encoding="utf-8"
        )
        # 确认频率相关 amplitude 表存在
        self.assertIn("*Amplitude, name=CGNS_R_N1_D3, definition=TABULAR", include_text)
        self.assertIn("100,", include_text)
        self.assertIn("200,", include_text)
        self.assertIn("300,", include_text)
        # 确认 CLOAD 引用 amplitude
        self.assertIn("*CLOAD, REAL, amplitude=CGNS_R_N1_D3", include_text)
        self.assertIn("*CLOAD, IMAGINARY, amplitude=CGNS_I_N1_D3", include_text)


if __name__ == "__main__":
    unittest.main()
