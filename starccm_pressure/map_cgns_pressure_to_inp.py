"""将 STAR-CCM+ 提取压力映射到 Abaqus INP 结构模型。

本模块只消费 `extract_cgns_pressure.py` 已导出的结果，不重新读取
CGNS 文件。稳态动力学使用 `surface_geometry.npz` 和
`pressure_complex_spectrum.npz`，瞬态/显式动力学使用
`surface_geometry.npz` 和 `pressure_time.json.gz`。

主要入口：
    parse_inp_text: 解析映射所需的 INP 子集。
    select_target_faces: 从用户指定的 Nset/Elset 提取受载面。
    map_complex_pressure_to_nodes: 将面压力换算为节点等效力。
    run_mapping: 串联读取、映射、include 写入和新 INP 生成。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


StepKind = Literal["steady_state", "dynamic_explicit", "dynamic", "unknown"]


class FileSizeLimitError(ValueError):
    """时域载荷展开规模超过安全上限时抛出的异常。"""


@dataclass(frozen=True)
class AbaqusElement:
    """保存 INP 中一个单元的最小映射信息。

    Attributes:
        element_id: Abaqus 单元编号。
        element_type: 单元类型关键字，例如 S4 或 C3D8。
        node_ids: 单元连接的节点编号序列。
    """

    element_id: int
    element_type: str
    node_ids: list[int]


@dataclass(frozen=True)
class AbaqusStep:
    """保存 Abaqus 分析步在原始 INP 中的位置和类型。

    Attributes:
        name: 分析步名称；未命名时由解析器生成默认名。
        start_line: `*Step` 所在的行号。
        end_line: `*End Step` 所在的行号。
        kind: 由步内关键字识别出的分析类型。
    """

    name: str
    start_line: int
    end_line: int
    kind: StepKind


@dataclass(frozen=True)
class AbaqusModel:
    """映射流程需要的 INP 解析结果。

    Attributes:
        lines: 原始 INP 文本行，用于保留未解析内容并插入 include。
        nodes: 节点编号到三维坐标的映射。
        elements: 单元编号到单元连接信息的映射。
        nsets: 节点集名称到节点编号集合的映射。
        elsets: 单元集名称到单元编号集合的映射。
        steps: 文件中识别到的 Abaqus 分析步。
    """

    lines: list[str]
    nodes: dict[int, np.ndarray]
    elements: dict[int, AbaqusElement]
    nsets: dict[str, set[int]]
    elsets: dict[str, set[int]]
    steps: list[AbaqusStep]


@dataclass(frozen=True)
class TargetFaces:
    """结构模型中实际承受流体压力的目标面集合。

    Attributes:
        node_ids: 每个目标面的节点编号。
        centers: 目标面中心坐标。
        area_vectors: 目标面面积向量，方向用于确定压力等效力方向。
    """

    node_ids: list[list[int]]
    centers: np.ndarray
    area_vectors: np.ndarray


@dataclass(frozen=True)
class MappingResult:
    """一次映射运行生成的文件路径。

    Attributes:
        output_inp_path: 首个输出 INP，保留给单频率调用的兼容入口。
        output_inp_paths: 所有输出 INP；多频率时每个频率一个文件。
        include_paths: 载荷 include 文件路径。
        report_path: 映射报告 JSON 路径。
    """

    output_inp_path: Path
    output_inp_paths: list[Path]
    include_paths: list[Path]
    report_path: Path


def _keyword_name(line: str) -> str:
    """提取 Abaqus 关键字行的主关键字名称。"""
    return line.split(",", 1)[0].strip().lower()


def _parse_keyword_params(line: str) -> dict[str, str | bool]:
    """解析 Abaqus 关键字行中的逗号参数。

    Args:
        line: 以 `*` 开头的 Abaqus 关键字行。

    Returns:
        参数名到参数值的映射；无值参数使用 True 标记。
    """
    params: dict[str, str | bool] = {}
    for part in line.strip()[1:].split(",")[1:]:
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            params[key.strip().lower()] = value.strip()
        else:
            params[item.lower()] = True
    return params


def _parse_int_list(line: str) -> list[int]:
    """解析 Abaqus 逗号分隔的整数列表。"""
    return [int(value.strip()) for value in line.split(",") if value.strip()]


def _parse_set_values(lines: list[str], start: int, generated: bool) -> tuple[set[int], int]:
    """解析 Nset/Elset 的连续数据行。

    Args:
        lines: INP 全部文本行。
        start: 集合数据起始行。
        generated: 是否按 Abaqus `generate` 语法展开。

    Returns:
        `(集合编号, 下一个关键字行号)`。

    Raises:
        ValueError: `generate` 行不是 start/end/increment 三元组。
    """
    values: set[int] = set()
    index = start
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("**"):
            index += 1
            continue
        if line.startswith("*"):
            break
        numbers = _parse_int_list(line)
        if generated:
            if len(numbers) != 3:
                raise ValueError("Generated set rows must contain start, end, increment.")
            values.update(range(numbers[0], numbers[1] + 1, numbers[2]))
        else:
            values.update(numbers)
        index += 1
    return values, index


def parse_inp_text(text: str) -> AbaqusModel:
    """解析压力映射所需的 INP 子集。

    Args:
        text: Abaqus 输入文件全文。

    Returns:
        包含节点、单元、集合和分析步的解析结果。

    Raises:
        ValueError: INP 使用嵌套 include，或关键字行缺少必要参数。
    """
    lines = text.splitlines()
    nodes: dict[int, np.ndarray] = {}
    elements: dict[int, AbaqusElement] = {}
    nsets: dict[str, set[int]] = {}
    elsets: dict[str, set[int]] = {}
    raw_steps: list[tuple[str, int, int, list[str]]] = []

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        lower = stripped.lower()
        if not stripped or stripped.startswith("**"):
            index += 1
            continue
        if lower.startswith("*include"):
            raise ValueError("Nested *INCLUDE files are not supported in v1.")
        if not stripped.startswith("*"):
            index += 1
            continue

        keyword = _keyword_name(stripped)
        params = _parse_keyword_params(stripped)

        if keyword == "*node":
            index += 1
            while index < len(lines) and not lines[index].lstrip().startswith("*"):
                row = lines[index].strip()
                if row and not row.startswith("**"):
                    parts = [part.strip() for part in row.split(",")]
                    if len(parts) < 4:
                        raise ValueError(f"Invalid *Node row: {row}")
                    nodes[int(parts[0])] = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=float,
                    )
                index += 1
            continue

        if keyword == "*element":
            element_type = str(params.get("type", "")).upper()
            header_elset = str(params.get("elset", "")).strip()
            if not element_type:
                raise ValueError("*Element keyword must include type=.")
            index += 1
            while index < len(lines) and not lines[index].lstrip().startswith("*"):
                row = lines[index].strip()
                if row and not row.startswith("**"):
                    numbers = _parse_int_list(row)
                    element_id = numbers[0]
                    elements[element_id] = AbaqusElement(
                        element_id=element_id,
                        element_type=element_type,
                        node_ids=numbers[1:],
                    )
                    if header_elset:
                        elsets.setdefault(header_elset.upper(), set()).add(element_id)
                index += 1
            continue

        if keyword in {"*nset", "*elset"}:
            name_key = "nset" if keyword == "*nset" else "elset"
            set_name = str(params.get(name_key, "")).strip()
            if not set_name:
                raise ValueError(f"{keyword} must include {name_key}=.")
            generated = bool(params.get("generate", False))
            values, index = _parse_set_values(lines, index + 1, generated)
            target = nsets if keyword == "*nset" else elsets
            target.setdefault(set_name.upper(), set()).update(values)
            continue

        if keyword == "*step":
            name = str(params.get("name", f"STEP_{len(raw_steps) + 1}"))
            start_line = index
            step_keywords: list[str] = []
            index += 1
            while index < len(lines):
                row = lines[index].strip()
                if row.lower().startswith("*end step"):
                    raw_steps.append((name, start_line, index, step_keywords))
                    index += 1
                    break
                if row.startswith("*"):
                    step_keywords.append(row)
                index += 1
            continue

        index += 1

    steps = [
        AbaqusStep(name=name, start_line=start, end_line=end, kind=identify_step_kind(keys))
        for name, start, end, keys in raw_steps
    ]
    return AbaqusModel(
        lines=lines,
        nodes=nodes,
        elements=elements,
        nsets=nsets,
        elsets=elsets,
        steps=steps,
    )


def identify_step_kind(step_keywords: Iterable[str]) -> StepKind:
    """根据分析步内部关键字判断载荷应采用频域还是时域写法。

    Args:
        step_keywords: 一个 `*Step` 与 `*End Step` 之间的关键字行。

    Returns:
        识别出的分析步类型；无法识别时返回 `unknown`。
    """
    for keyword in step_keywords:
        lower = keyword.lower()
        if lower.startswith("*steady state dynamics"):
            return "steady_state"
        if lower.startswith("*dynamic") and "explicit" in lower:
            return "dynamic_explicit"
        if lower.startswith("*dynamic"):
            return "dynamic"
    return "unknown"


def parse_inp_file(path: str | Path) -> AbaqusModel:
    """读取并解析 Abaqus INP 文件。

    Args:
        path: INP 文件路径。

    Returns:
        映射流程使用的解析模型。
    """
    return parse_inp_text(Path(path).read_text(encoding="utf-8"))


def _element_face_node_ids(element: AbaqusElement) -> list[list[int]]:
    """按单元类型生成可参与压力加载的候选面。

    当前只覆盖常见壳、二维面单元和一阶 C3D4/C3D8 实体。高阶实体
    或特殊单元需要单独定义面节点顺序，否则面积向量方向不可靠。
    """
    element_type = element.element_type.upper()
    nodes = element.node_ids
    if element_type.startswith(("S", "R3D", "M3D", "CPS", "CPE")):
        return [nodes]
    if element_type.startswith("C3D4"):
        return [
            [nodes[0], nodes[2], nodes[1]],
            [nodes[0], nodes[1], nodes[3]],
            [nodes[1], nodes[2], nodes[3]],
            [nodes[2], nodes[0], nodes[3]],
        ]
    if element_type.startswith("C3D8"):
        return [
            [nodes[0], nodes[1], nodes[2], nodes[3]],
            [nodes[4], nodes[7], nodes[6], nodes[5]],
            [nodes[0], nodes[4], nodes[5], nodes[1]],
            [nodes[1], nodes[5], nodes[6], nodes[2]],
            [nodes[2], nodes[6], nodes[7], nodes[3]],
            [nodes[3], nodes[7], nodes[4], nodes[0]],
        ]
    raise ValueError(f"Unsupported element type for surface extraction: {element_type}")


def _area_vector(points: np.ndarray) -> np.ndarray:
    """用三角扇求多边形面的面积向量。

    Args:
        points: 目标面节点坐标，按单元面顺序排列。

    Returns:
        面积向量，模长为面积，方向由节点顺序决定。

    Raises:
        ValueError: 节点数不足或面退化。
    """
    if points.shape[0] < 3:
        raise ValueError("A face must contain at least three nodes.")
    vector = np.zeros(3, dtype=float)
    origin = points[0]
    for index in range(1, points.shape[0] - 1):
        vector += 0.5 * np.cross(points[index] - origin, points[index + 1] - origin)
    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0.0:
        raise ValueError("Degenerate target face encountered.")
    return vector


def select_target_faces(
    model: AbaqusModel,
    target_set: str,
    target_set_type: Literal["nset", "elset"],
) -> TargetFaces:
    """从用户指定的 Nset 或 Elset 生成结构目标面。

    Elset 会先展开为候选单元面，再只保留出现一次的外表面，避免
    内部共享面被加载。Nset 则选择节点全部落在该集合内的单元面。

    Args:
        model: 已解析的 Abaqus 模型。
        target_set: 用户指定的节点集或单元集名称。
        target_set_type: 集合类型，必须为 `nset` 或 `elset`。

    Returns:
        目标面的节点、中心和面积向量。

    Raises:
        ValueError: 目标集合不存在、集合无法生成面或面引用缺失节点。
    """
    normalized_set = target_set.upper()
    selected_faces: list[list[int]] = []

    if target_set_type.lower() == "elset":
        if normalized_set not in model.elsets:
            raise ValueError(f"Target elset '{target_set}' was not found.")
        element_ids = model.elsets[normalized_set]
        face_counts: dict[tuple[int, ...], int] = {}
        candidate_faces: list[list[int]] = []
        for element_id in element_ids:
            element = model.elements.get(element_id)
            if element is None:
                raise ValueError(f"Elset '{target_set}' references missing element {element_id}.")
            for face in _element_face_node_ids(element):
                candidate_faces.append(face)
                face_counts[tuple(sorted(face))] = face_counts.get(tuple(sorted(face)), 0) + 1
        for face in candidate_faces:
            if face_counts[tuple(sorted(face))] == 1:
                selected_faces.append(face)
    elif target_set_type.lower() == "nset":
        if normalized_set not in model.nsets:
            raise ValueError(f"Target nset '{target_set}' was not found.")
        selected_nodes = model.nsets[normalized_set]
        for element in model.elements.values():
            for face in _element_face_node_ids(element):
                if all(node_id in selected_nodes for node_id in face):
                    selected_faces.append(face)
    else:
        raise ValueError("target_set_type must be 'nset' or 'elset'.")

    if not selected_faces:
        raise ValueError(f"Target {target_set_type} '{target_set}' did not produce any faces.")

    centers: list[np.ndarray] = []
    area_vectors: list[np.ndarray] = []
    for face in selected_faces:
        try:
            points = np.vstack([model.nodes[node_id] for node_id in face])
        except KeyError as exc:
            raise ValueError(f"Face references missing node {exc.args[0]}.") from exc
        centers.append(np.mean(points, axis=0))
        area_vectors.append(_area_vector(points))
    return TargetFaces(
        node_ids=selected_faces,
        centers=np.vstack(centers),
        area_vectors=np.vstack(area_vectors),
    )


def apply_coordinate_transform(
    coordinates: np.ndarray,
    scale: float = 1.0,
    translate: np.ndarray | None = None,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """将 CGNS 坐标转换到 INP 坐标系。

    Args:
        coordinates: CGNS 表面面心或节点坐标。
        scale: 坐标统一比例系数。
        translate: 平移向量；为 None 时不平移。
        axis_order: 坐标轴重排顺序，例如 `(2, 0, 1)`。
        axis_sign: 每个重排后坐标轴的方向符号。

    Returns:
        与输入行数一致的转换后坐标。

    Raises:
        ValueError: 输入不是三列坐标，或轴顺序不是合法排列。
    """
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3).")
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError("axis_order must be a permutation of 0, 1, 2.")
    transformed = values[:, axis_order] * np.asarray(axis_sign, dtype=float)
    transformed = transformed * float(scale)
    if translate is not None:
        transformed = transformed + np.asarray(translate, dtype=float)
    return transformed


def _load_ckdtree() -> Any | None:
    """Return scipy.spatial.cKDTree when SciPy is installed, otherwise None."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    return cKDTree


def _nearest_weights_bruteforce(
    target_centers: np.ndarray,
    source_centers: np.ndarray,
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute inverse-distance nearest weights with a NumPy full-distance scan."""
    indices = np.empty((target_centers.shape[0], neighbor_count), dtype=int)
    weights = np.empty((target_centers.shape[0], neighbor_count), dtype=float)
    for row, target in enumerate(target_centers):
        distances = np.linalg.norm(source_centers - target, axis=1)
        nearest = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        nearest = nearest[np.argsort(distances[nearest])]
        nearest_distances = distances[nearest]
        if nearest_distances[0] <= 1.0e-12:
            row_weights = np.zeros(neighbor_count, dtype=float)
            row_weights[0] = 1.0
        else:
            inverse = 1.0 / nearest_distances
            row_weights = inverse / np.sum(inverse)
        indices[row] = nearest
        weights[row] = row_weights
    return indices, weights


def _nearest_weights(
    target_centers: np.ndarray,
    source_centers: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算每个目标面对应的最近源面索引和反距离权重。

    Args:
        target_centers: 结构目标面中心。
        source_centers: CGNS 源表面面心。
        k: 每个目标面选取的最近源面数量。

    Returns:
        `(最近源面索引, 权重)`，二者行数均等于目标面数量。
    """
    source_count = source_centers.shape[0]
    neighbor_count = min(max(int(k), 1), source_count)
    tree_type = _load_ckdtree()
    if tree_type is None:
        return _nearest_weights_bruteforce(
            target_centers,
            source_centers,
            neighbor_count,
        )

    distances, indices = tree_type(source_centers).query(
        target_centers,
        k=neighbor_count,
    )
    distances = np.asarray(distances, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if neighbor_count == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    weights = np.empty_like(distances, dtype=float)
    exact_match_rows = distances[:, 0] <= 1.0e-12
    weights[exact_match_rows, :] = 0.0
    weights[exact_match_rows, 0] = 1.0
    non_exact_rows = ~exact_match_rows
    if np.any(non_exact_rows):
        inverse = 1.0 / distances[non_exact_rows, :]
        weights[non_exact_rows, :] = inverse / np.sum(
            inverse,
            axis=1,
            keepdims=True,
        )
    return indices, weights


def interpolate_pressure_to_faces(
    target_centers: np.ndarray,
    source_centers: np.ndarray,
    source_pressure: np.ndarray,
    k: int = 4,
) -> tuple[np.ndarray, dict[str, float]]:
    """将 CGNS 面压力插值到结构目标面中心。

    采用反距离权重而不是最近邻硬匹配，是为了在两套网格面心不完全
    重合时保持载荷分布连续；若距离接近零，则直接使用重合面压力。

    Args:
        target_centers: 结构目标面中心。
        source_centers: CGNS 源表面面心。
        source_pressure: 与源面心一一对应的压力值，可为实数或复数。
        k: 参与反距离权重的最近源面数量。

    Returns:
        `(目标面压力, 距离统计信息)`。

    Raises:
        ValueError: 坐标或压力数组形状不匹配。
    """
    target = np.asarray(target_centers, dtype=float)
    source = np.asarray(source_centers, dtype=float)
    pressure = np.asarray(source_pressure)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target_centers must have shape (N, 3).")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_centers must have shape (N, 3).")
    if source.shape[0] != pressure.shape[0]:
        raise ValueError("source_pressure length must match source_centers.")
    indices, weights = _nearest_weights(target, source, k)
    interpolated = np.sum(pressure[indices] * weights, axis=1)
    nearest_distances = np.linalg.norm(source[indices[:, 0]] - target, axis=1)
    return interpolated, {
        "max_nearest_distance": float(np.max(nearest_distances)),
        "mean_nearest_distance": float(np.mean(nearest_distances)),
    }


def pressure_faces_to_node_forces(
    face_pressures: np.ndarray,
    target_area_vectors: np.ndarray,
    target_node_ids: list[list[int]],
) -> dict[int, np.ndarray]:
    """将面压力换算为 Abaqus 节点等效集中力。

    压力正方向取为压向结构面，因此等效面力采用
    `force = -pressure * area_vector`。当前实现将面力平均分配给该面
    节点，适合线性壳面和一阶实体外表面的初版载荷映射。

    Args:
        face_pressures: 结构目标面上的压力值。
        target_area_vectors: 结构目标面的面积向量。
        target_node_ids: 每个目标面对应的节点编号。

    Returns:
        节点编号到三向等效力的映射。

    Raises:
        ValueError: 压力数量与目标面数量不一致。
    """
    pressures = np.asarray(face_pressures)
    area_vectors = np.asarray(target_area_vectors, dtype=float)
    if pressures.shape[0] != area_vectors.shape[0]:
        raise ValueError("face_pressures and target_area_vectors must have matching rows.")
    forces: dict[int, np.ndarray] = {}
    for face_index, nodes in enumerate(target_node_ids):
        face_force = -pressures[face_index] * area_vectors[face_index]
        contribution = np.asarray(face_force) / float(len(nodes))
        for node_id in nodes:
            if node_id not in forces:
                forces[node_id] = np.zeros(3, dtype=contribution.dtype)
            forces[node_id] = forces[node_id] + contribution
    return forces


def map_complex_pressure_to_nodes(
    target_centers: np.ndarray,
    target_area_vectors: np.ndarray,
    target_node_ids: list[list[int]],
    source_centers: np.ndarray,
    source_pressure: np.ndarray,
    k: int = 4,
) -> tuple[dict[int, np.ndarray], dict[str, float | int]]:
    """完成频域压力到节点复数力的一步映射。

    Args:
        target_centers: 结构目标面中心。
        target_area_vectors: 结构目标面面积向量。
        target_node_ids: 结构目标面节点编号。
        source_centers: CGNS 源表面面心。
        source_pressure: CGNS 源表面复数压力。
        k: 反距离插值使用的最近邻数量。

    Returns:
        `(节点复数力, 映射统计信息)`。
    """
    face_pressure, distance_stats = interpolate_pressure_to_faces(
        target_centers,
        source_centers,
        source_pressure,
        k=k,
    )
    node_forces = pressure_faces_to_node_forces(
        face_pressure,
        target_area_vectors,
        target_node_ids,
    )
    stats: dict[str, float | int] = {
        "target_face_count": int(len(target_node_ids)),
        "target_node_count": int(len(node_forces)),
        "source_face_count": int(np.asarray(source_centers).shape[0]),
    }
    stats.update(distance_stats)
    return node_forces, stats


def _format_float(value: float) -> str:
    """用紧凑格式写出 Abaqus 输入文件中的浮点数。"""
    if value == 0.0:
        return "0"
    return f"{value:.12g}"


def write_frequency_load_include(
    path: str | Path,
    node_forces: dict[int, np.ndarray],
    frequency_hz: float,
    load_name: str = "CGNS_PRESSURE",
) -> None:
    """写出单个频率的 Abaqus 复数集中力 include。

    Args:
        path: include 文件输出路径。
        node_forces: 节点编号到三向复数力的映射。
        frequency_hz: 本 include 对应的物理频率。
        load_name: 写入注释行的载荷名称。
    """
    include_path = Path(path)
    lines = [
        f"** {load_name} mapped load, frequency_hz={_format_float(frequency_hz)}",
        "** Real part",
        "*CLOAD, REAL",
    ]
    for node_id in sorted(node_forces):
        force = np.asarray(node_forces[node_id])
        for component in range(3):
            value = float(np.real(force[component]))
            if value != 0.0:
                lines.append(f"{node_id}, {component + 1}, {_format_float(value)}")
    lines.extend(["** Imaginary part", "*CLOAD, IMAGINARY"])
    for node_id in sorted(node_forces):
        force = np.asarray(node_forces[node_id])
        for component in range(3):
            value = float(np.imag(force[component]))
            if value != 0.0:
                lines.append(f"{node_id}, {component + 1}, {_format_float(value)}")
    include_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def estimate_time_load_records(
    node_count: int,
    component_count: int,
    sample_count: int,
    max_records: int = 500000,
) -> int:
    """估算时域载荷展开规模并执行安全上限保护。

    Args:
        node_count: 受载节点数量。
        component_count: 每个节点的力分量数量。
        sample_count: 时域采样点数量。
        max_records: 允许写出的最大记录数。

    Returns:
        预计写出的记录数。

    Raises:
        FileSizeLimitError: 预计记录数超过上限。
    """
    records = int(node_count) * int(component_count) * int(sample_count)
    if records > int(max_records):
        raise FileSizeLimitError(
            f"Time-domain load expansion would write {records} records; "
            f"limit is {max_records}."
        )
    return records


def write_time_load_include(
    path: str | Path,
    node_force_series: dict[int, np.ndarray],
    times: np.ndarray,
    max_records: int = 500000,
) -> None:
    """写出瞬态/显式动力学使用的时域集中力 include。

    Args:
        path: include 文件输出路径。
        node_force_series: 节点编号到 `(时间, 三向力)` 数组的映射。
        times: 时刻数组。
        max_records: 时域载荷展开的安全上限。

    Raises:
        FileSizeLimitError: 写出规模超过安全上限。
    """
    active_records = 0
    for force_series in node_force_series.values():
        active_records += int(np.count_nonzero(np.asarray(force_series) != 0.0))
    estimate_time_load_records(
        node_count=max(len(node_force_series), 1),
        component_count=3,
        sample_count=len(times),
        max_records=max_records,
    )
    lines = ["** CGNS_PRESSURE mapped time-domain loads"]
    for node_id in sorted(node_force_series):
        values = np.asarray(node_force_series[node_id], dtype=float)
        for component in range(3):
            series = values[:, component]
            if not np.any(series != 0.0):
                continue
            amp_name = f"CGNS_N{node_id}_D{component + 1}"
            lines.append(f"*Amplitude, name={amp_name}, time=TOTAL TIME")
            pairs = [
                f"{_format_float(float(time_value))}, {_format_float(float(load_value))}"
                for time_value, load_value in zip(times, series)
            ]
            lines.extend(pairs)
            lines.append(f"*CLOAD, amplitude={amp_name}")
            lines.append(f"{node_id}, {component + 1}, 1.")
    if active_records == 0:
        lines.append("** No nonzero time-domain loads were generated.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_frequency_index(
    frequencies_hz: np.ndarray,
    target_frequency_hz: float,
    tolerance_hz: float = 1.0e-6,
) -> int:
    """在提取频率轴上查找与目标频率匹配的索引。

    Args:
        frequencies_hz: CGNS 提取结果中的频率数组。
        target_frequency_hz: 用户请求的目标频率。
        tolerance_hz: 允许的最大频率偏差。

    Returns:
        最接近目标频率的数组索引。

    Raises:
        ValueError: 频率数组为空，或最近频率仍超出容差。
    """
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.size == 0:
        raise ValueError("Extracted frequency array is empty.")
    index = int(np.argmin(np.abs(frequencies - float(target_frequency_hz))))
    delta = abs(float(frequencies[index]) - float(target_frequency_hz))
    if delta > float(tolerance_hz):
        raise ValueError(
            f"No extracted frequency is within {tolerance_hz:g} Hz of "
            f"{target_frequency_hz:g} Hz; closest is {frequencies[index]:g} Hz."
        )
    return index


def _load_surface_geometry(extracted_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """加载提取器导出的表面几何缓存。

    Args:
        extracted_dir: CGNS 提取结果目录。
        name: 表面几何 NPZ 文件名。

    Returns:
        `(面心坐标, 面积向量)`。

    Raises:
        FileNotFoundError: 表面几何文件不存在。
        ValueError: 文件缺少映射所需数组。
    """
    path = extracted_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required surface geometry file was not found: {path}")
    with np.load(path) as saved:
        if "centers" not in saved or "area_vectors" not in saved:
            raise ValueError(f"{path} must contain centers and area_vectors arrays.")
        return (
            np.asarray(saved["centers"], dtype=float),
            np.asarray(saved["area_vectors"], dtype=float),
        )


def _load_complex_spectrum(
    extracted_dir: Path,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """加载全场复数压力谱。

    Args:
        extracted_dir: CGNS 提取结果目录。
        name: 复数压力谱 NPZ 文件名。

    Returns:
        `(频率轴, 复数压力谱)`，压力谱形状为 `(频率, 源面)`。

    Raises:
        FileNotFoundError: 复数压力谱文件不存在。
    """
    path = extracted_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required complex spectrum file was not found: {path}")
    with np.load(path) as saved:
        frequencies = np.asarray(saved["frequencies_hz"], dtype=float)
        pressure = (
            np.asarray(saved["pressure_real"], dtype=float)
            + 1j * np.asarray(saved["pressure_imag"], dtype=float)
        )
    return frequencies, pressure


def _load_time_pressure(
    extracted_dir: Path,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """加载旧版 gzip JSON 中的时域脉动压力。

    Args:
        extracted_dir: CGNS 提取结果目录。
        name: 时域压力 gzip JSON 文件名。

    Returns:
        `(时间轴, 时域压力)`，压力数组形状为 `(时间, 源面)`。

    Raises:
        FileNotFoundError: 时域压力文件不存在。
    """
    path = extracted_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required time pressure file was not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    times = np.asarray(payload["time_s"], dtype=float)
    pressure = np.asarray(payload["pulsating_pressure"], dtype=float)
    return times, pressure


def insert_include_before_step_end(
    lines: list[str],
    step: AbaqusStep,
    include_path: Path,
) -> list[str]:
    """在目标分析步结束前插入 Abaqus include 指令。

    Args:
        lines: 原始 INP 文本行。
        step: 需要接收载荷的分析步。
        include_path: 载荷 include 文件路径。

    Returns:
        已插入 include 指令的新 INP 文本行。
    """
    include_line = f"*INCLUDE, INPUT={include_path.name}"
    return lines[: step.end_line] + [include_line] + lines[step.end_line :]


def _write_mapping_report(
    report_path: Path,
    payload: dict[str, Any],
) -> None:
    """写出映射报告，保留坐标变换和频率匹配等溯源信息。"""
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _mapping_assumptions() -> dict[str, str]:
    """Describe the physical assumptions used by the pressure mapping."""
    return {
        "pressure_sign": "-pressure * area_vector",
        "node_force_distribution": "equal_share_per_face_node",
        "interpolation": "inverse_distance_weighting_on_face_centers",
        "supported_element_scope": "linear shell/surface elements and first-order C3D4/C3D8 exterior faces",
    }


def _default_output_path(inp_path: Path) -> Path:
    """根据原始 INP 路径生成默认输出路径。"""
    return inp_path.with_name(f"{inp_path.stem}_mapped{inp_path.suffix}")


def _frequency_suffix(frequency_hz: float) -> str:
    """将频率值转换为可用于文件名的后缀。"""
    text = f"{float(frequency_hz):g}".replace("-", "m").replace(".", "p")
    return f"{text}Hz"


def _frequency_output_path(base_output: Path, frequency_hz: float) -> Path:
    """为多频率稳态映射生成单频率输出 INP 路径。"""
    suffix = _frequency_suffix(frequency_hz)
    return base_output.with_name(f"{base_output.stem}_{suffix}{base_output.suffix}")


def _load_model_step(model: AbaqusModel, requested_step: str | None) -> AbaqusStep:
    """选择接收载荷 include 的 Abaqus 分析步。

    Args:
        model: 已解析的 INP 模型。
        requested_step: 用户指定的 step 名称；为 None 时使用首个 step。

    Returns:
        匹配的分析步。

    Raises:
        ValueError: 模型没有分析步，或指定名称不存在。
    """
    if not model.steps:
        raise ValueError("The INP file does not contain an Abaqus *Step.")
    if requested_step is None:
        return model.steps[0]
    for step in model.steps:
        if step.name.upper() == requested_step.upper():
            return step
    raise ValueError(f"Step '{requested_step}' was not found.")


def run_mapping(
    *,
    inp_path: str | Path,
    extracted_dir: str | Path,
    target_set: str,
    target_set_type: Literal["nset", "elset"],
    frequencies: list[float] | None = None,
    output_path: str | Path | None = None,
    step_name: str | None = None,
    frequency_tolerance_hz: float = 1.0e-6,
    k: int = 4,
    scale: float = 1.0,
    translate: np.ndarray | None = None,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_time_records: int = 500000,
    surface_geometry_name: str = "surface_geometry.npz",
    complex_spectrum_name: str = "pressure_complex_spectrum.npz",
    pressure_time_name: str = "pressure_time.json.gz",
    report_name: str = "mapping_report.json",
) -> MappingResult:
    """执行完整压力映射流程并写出新 INP。

    稳态动力学按指定频率读取复数压力谱；若给出多个频率，则每个
    频率生成一个独立 INP 和 include。瞬态/显式动力学读取时域压力，
    并在写 include 前检查展开规模。

    Args:
        inp_path: 原始 Abaqus INP 文件路径。
        extracted_dir: CGNS 提取结果目录。
        target_set: 受载 Nset 或 Elset 名称。
        target_set_type: 目标集合类型。
        frequencies: 稳态动力学目标频率列表。
        output_path: 输出 INP 基础路径；为 None 时自动加 `_mapped`。
        step_name: 指定接收载荷的分析步名称；为 None 时使用首个 step。
        frequency_tolerance_hz: 频率匹配容差。
        k: 反距离插值使用的最近邻数量。
        scale: CGNS 到 INP 坐标的比例系数。
        translate: CGNS 到 INP 坐标的平移向量。
        axis_order: CGNS 到 INP 坐标的轴顺序。
        axis_sign: CGNS 到 INP 坐标的轴方向符号。
        max_time_records: 时域 include 允许写出的最大记录数。
        surface_geometry_name: 表面几何 NPZ 文件名。
        complex_spectrum_name: 复数压力谱 NPZ 文件名。
        pressure_time_name: 时域压力 gzip JSON 文件名。
        report_name: 映射报告文件名。

    Returns:
        输出 INP、include 和报告路径。

    Raises:
        ValueError: 输入配置不完整、步类型不支持或会覆盖原 INP。
        FileNotFoundError: 必要的提取结果文件不存在。
        FileSizeLimitError: 时域载荷输出规模超过上限。
    """
    inp = Path(inp_path)
    extracted = Path(extracted_dir)
    output = Path(output_path) if output_path is not None else _default_output_path(inp)
    if output.resolve() == inp.resolve():
        raise ValueError("Output INP must not overwrite the original INP.")
    if not target_set:
        raise ValueError("target_set is required; mapping all nodes is not supported.")

    model = parse_inp_file(inp)
    step = _load_model_step(model, step_name)
    target_faces = select_target_faces(model, target_set, target_set_type)
    source_centers, _source_area_vectors = _load_surface_geometry(extracted, surface_geometry_name)
    transformed_source_centers = apply_coordinate_transform(
        source_centers,
        scale=scale,
        translate=translate,
        axis_order=axis_order,
        axis_sign=axis_sign,
    )

    include_paths: list[Path] = []
    output_paths: list[Path] = []
    report: dict[str, Any] = {
        "input_inp": str(inp),
        "output_inp": str(output),
        "target_set": target_set,
        "target_set_type": target_set_type,
        "step_name": step.name,
        "step_kind": step.kind,
        "k": int(k),
        "scale": float(scale),
        "axis_order": list(axis_order),
        "axis_sign": list(axis_sign),
        "translate": None if translate is None else np.asarray(translate, dtype=float).tolist(),
        "mapping_assumptions": _mapping_assumptions(),
    }

    if step.kind == "steady_state":
        if not frequencies:
            raise ValueError("At least one --frequency is required for steady-state mapping.")
        extracted_frequencies, pressure_spectrum = _load_complex_spectrum(
            extracted,
            complex_spectrum_name,
        )
        per_frequency_stats: list[dict[str, Any]] = []
        mapped_frequencies: list[float] = []
        for requested_frequency in frequencies:
            output_for_frequency = (
                output
                if len(frequencies) == 1
                else _frequency_output_path(output, float(requested_frequency))
            )
            include_path = output_for_frequency.with_name(
                f"{output_for_frequency.stem}_loads.inc"
            )
            spectrum_index = find_frequency_index(
                extracted_frequencies,
                float(requested_frequency),
                tolerance_hz=frequency_tolerance_hz,
            )
            node_forces, stats = map_complex_pressure_to_nodes(
                target_faces.centers,
                target_faces.area_vectors,
                target_faces.node_ids,
                transformed_source_centers,
                pressure_spectrum[spectrum_index],
                k=k,
            )
            write_frequency_load_include(
                include_path,
                node_forces,
                frequency_hz=float(extracted_frequencies[spectrum_index]),
            )
            include_paths.append(include_path)
            output_paths.append(output_for_frequency)
            mapped_frequencies.append(float(extracted_frequencies[spectrum_index]))
            per_frequency_stats.append(
                {
                    "requested_frequency_hz": float(requested_frequency),
                    "mapped_frequency_hz": float(extracted_frequencies[spectrum_index]),
                    **stats,
                }
            )
        report.update(
            {
                "frequencies_hz": mapped_frequencies,
                "mapping_stats": per_frequency_stats[0]
                if len(per_frequency_stats) == 1
                else per_frequency_stats,
                "output_inps": [str(path) for path in output_paths],
                "include_files": [str(path) for path in include_paths],
            }
        )
    elif step.kind in {"dynamic", "dynamic_explicit"}:
        times, pressure_time = _load_time_pressure(extracted, pressure_time_name)
        node_series: dict[int, np.ndarray] = {}
        last_stats: dict[str, float | int] = {}
        for time_index in range(pressure_time.shape[0]):
            node_forces, last_stats = map_complex_pressure_to_nodes(
                target_faces.centers,
                target_faces.area_vectors,
                target_faces.node_ids,
                transformed_source_centers,
                pressure_time[time_index],
                k=k,
            )
            for node_id, force in node_forces.items():
                if node_id not in node_series:
                    node_series[node_id] = np.zeros((pressure_time.shape[0], 3), dtype=float)
                node_series[node_id][time_index] = np.real(force)
        include_path = output.with_name(f"{output.stem}_loads.inc")
        write_time_load_include(
            include_path,
            node_series,
            times,
            max_records=max_time_records,
        )
        include_paths.append(include_path)
        output_paths.append(output)
        report.update(
            {
                "time_sample_count": int(len(times)),
                "mapping_stats": last_stats,
                "output_inps": [str(output)],
                "include_files": [str(include_path)],
            }
        )
    else:
        raise ValueError(
            "Only *STEADY STATE DYNAMICS and *DYNAMIC steps are supported for mapping."
        )

    for output_file, include_file in zip(output_paths, include_paths):
        new_lines = insert_include_before_step_end(model.lines, step, include_file)
        output_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    report_path = output.with_name(report_name)
    _write_mapping_report(report_path, report)
    return MappingResult(
        output_inp_path=output_paths[0],
        output_inp_paths=output_paths,
        include_paths=include_paths,
        report_path=report_path,
    )


def _parse_triplet(text: str, *, cast=float) -> tuple[Any, Any, Any]:
    """解析 CLI 中的三元组参数，例如坐标平移或轴顺序。"""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values.")
    return tuple(cast(part) for part in parts)  # type: ignore[return-value]


def _expand_frequency_range(text: str) -> list[float]:
    """Expand an inclusive frequency range written as start:end:step."""
    try:
        start, end, step = (float(part.strip()) for part in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected frequency range as start:end:step, for example 1:800:1."
        ) from exc
    if step <= 0.0:
        raise argparse.ArgumentTypeError("Frequency range step must be positive.")
    if end < start:
        raise argparse.ArgumentTypeError("Frequency range end must be >= start.")

    values: list[float] = []
    current = start
    tolerance = abs(step) * 1.0e-9
    while current <= end + tolerance:
        values.append(float(round(current, 12)))
        current += step
    return values


def resolve_requested_frequencies(args: argparse.Namespace) -> list[float] | None:
    """Combine repeated --frequency values with inclusive --frequency-range values."""
    frequencies = list(getattr(args, "frequency", None) or [])
    for frequency_range in getattr(args, "frequency_range", None) or []:
        frequencies.extend(_expand_frequency_range(frequency_range))
    return frequencies or None


def parse_frequency_text(text: str) -> list[float] | None:
    """Parse GUI frequency text containing comma-separated values or ranges."""
    frequencies: list[float] = []
    for item in text.split(","):
        value = item.strip()
        if not value:
            continue
        if ":" in value:
            frequencies.extend(_expand_frequency_range(value))
        else:
            frequencies.append(float(value))
    return frequencies or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 命令行参数列表；为 None 时由 argparse 使用系统参数。

    Returns:
        argparse 解析后的命名空间。
    """
    parser = argparse.ArgumentParser(
        description="Map extracted CGNS pressure data onto an Abaqus INP model.",
    )
    parser.add_argument("--inp", required=True, help="Input Abaqus INP path.")
    parser.add_argument(
        "--extracted",
        required=True,
        help="Directory containing extracted CGNS outputs.",
    )
    parser.add_argument("--target-set", required=True, help="Target Nset or Elset name.")
    parser.add_argument(
        "--target-set-type",
        choices=("nset", "elset"),
        required=True,
        help="Whether --target-set names an Nset or Elset.",
    )
    parser.add_argument(
        "--frequency",
        action="append",
        type=float,
        help="Target frequency in Hz. Repeat for multiple runs; v1 maps one per output.",
    )
    parser.add_argument(
        "--frequency-range",
        action="append",
        help="Inclusive frequency range as start:end:step in Hz, e.g. 1:800:1.",
    )
    parser.add_argument("--output", help="Output INP path.")
    parser.add_argument("--step-name", help="Step name to receive the load include.")
    parser.add_argument(
        "--frequency-tolerance-hz",
        type=float,
        default=1.0e-6,
        help="Maximum allowed difference between requested and extracted frequency.",
    )
    parser.add_argument("--k", type=int, default=4, help="Nearest source faces for IDW.")
    parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scale factor.")
    parser.add_argument(
        "--translate",
        type=lambda text: np.array(_parse_triplet(text, cast=float), dtype=float),
        help="Coordinate translation as dx,dy,dz.",
    )
    parser.add_argument(
        "--axis-order",
        type=lambda text: _parse_triplet(text, cast=int),
        default=(0, 1, 2),
        help="Coordinate axis order, e.g. 0,1,2 or 2,0,1.",
    )
    parser.add_argument(
        "--axis-sign",
        type=lambda text: _parse_triplet(text, cast=float),
        default=(1.0, 1.0, 1.0),
        help="Coordinate axis signs, e.g. 1,-1,1.",
    )
    parser.add_argument(
        "--max-time-records",
        type=int,
        default=500000,
        help="Safety limit for expanded time-domain load records.",
    )
    return parser.parse_args(argv)


def run_cli(argv: list[str] | None = None) -> int:
    """执行命令行映射流程。

    Args:
        argv: 命令行参数列表；为 None 时读取系统参数。

    Returns:
        进程退出码，0 表示成功。
    """
    args = parse_args(argv)
    try:
        result = run_mapping(
            inp_path=args.inp,
            extracted_dir=args.extracted,
            target_set=args.target_set,
            target_set_type=args.target_set_type,
            frequencies=resolve_requested_frequencies(args),
            output_path=args.output,
            step_name=args.step_name,
            frequency_tolerance_hz=args.frequency_tolerance_hz,
            k=args.k,
            scale=args.scale,
            translate=args.translate,
            axis_order=args.axis_order,
            axis_sign=args.axis_sign,
            max_time_records=args.max_time_records,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for output_path in result.output_inp_paths:
        print(f"Wrote: {output_path}")
    for include_path in result.include_paths:
        print(f"Wrote: {include_path}")
    print(f"Wrote: {result.report_path}")
    return 0


def run_gui() -> int:
    """启动基于文件选择器的 Tkinter 映射界面。

    Returns:
        进程退出码；GUI 正常关闭返回 0。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        print(f"Error: Tkinter is required for GUI mode: {exc}", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("CGNS Pressure to INP Mapper")
    root.geometry("780x450")
    root.resizable(False, False)

    inp_var = tk.StringVar(value="")
    extracted_var = tk.StringVar(value="cgns_pressure_output")
    output_var = tk.StringVar(value="")
    target_set_var = tk.StringVar(value="")
    target_type_var = tk.StringVar(value="elset")
    frequency_var = tk.StringVar(value="1:800:1")
    status_var = tk.StringVar(value="Ready")

    frame = ttk.Frame(root, padding=16)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    def browse_inp() -> None:
        path = filedialog.askopenfilename(
            title="Select Abaqus INP",
            filetypes=[("Abaqus INP", "*.inp"), ("All files", "*.*")],
        )
        if path:
            inp_var.set(path)
            if not output_var.get().strip():
                input_path = Path(path)
                output_var.set(str(_default_output_path(input_path)))

    def browse_extracted() -> None:
        path = filedialog.askdirectory(title="Select extracted CGNS output directory")
        if path:
            extracted_var.set(path)

    def browse_output() -> None:
        path = filedialog.asksaveasfilename(
            title="Save mapped INP",
            defaultextension=".inp",
            filetypes=[("Abaqus INP", "*.inp"), ("All files", "*.*")],
        )
        if path:
            output_var.set(path)

    def run_job() -> None:
        try:
            frequencies = parse_frequency_text(frequency_var.get())
            result = run_mapping(
                inp_path=inp_var.get().strip(),
                extracted_dir=extracted_var.get().strip(),
                target_set=target_set_var.get().strip(),
                target_set_type=target_type_var.get().strip(),  # type: ignore[arg-type]
                frequencies=frequencies,
                output_path=output_var.get().strip() or None,
            )
        except Exception as exc:
            status_var.set(f"Error: {exc}")
            messagebox.showerror("Mapping failed", str(exc))
            return
        status_var.set(f"Wrote {result.output_inp_path}")
        messagebox.showinfo("Done", f"Wrote:\n{result.output_inp_path}")

    ttk.Label(frame, text="INP file").grid(row=0, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=inp_var, width=72).grid(row=0, column=1, sticky="ew", pady=6)
    ttk.Button(frame, text="Browse", command=browse_inp).grid(row=0, column=2, padx=(8, 0))

    ttk.Label(frame, text="Extracted dir").grid(row=1, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=extracted_var, width=72).grid(row=1, column=1, sticky="ew", pady=6)
    ttk.Button(frame, text="Browse", command=browse_extracted).grid(row=1, column=2, padx=(8, 0))

    ttk.Label(frame, text="Output INP").grid(row=2, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=output_var, width=72).grid(row=2, column=1, sticky="ew", pady=6)
    ttk.Button(frame, text="Browse", command=browse_output).grid(row=2, column=2, padx=(8, 0))

    ttk.Label(frame, text="Target set").grid(row=3, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=target_set_var, width=24).grid(row=3, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="Set type").grid(row=4, column=0, sticky="w", pady=6)
    ttk.Combobox(
        frame,
        textvariable=target_type_var,
        values=("elset", "nset"),
        state="readonly",
        width=12,
    ).grid(row=4, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="Frequency Hz or range").grid(row=5, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=frequency_var, width=32).grid(row=5, column=1, sticky="w", pady=6)

    ttk.Button(frame, text="Run mapping", command=run_job).grid(row=6, column=1, sticky="w", pady=(16, 8))
    ttk.Label(frame, textvariable=status_var, wraplength=680).grid(
        row=7,
        column=0,
        columnspan=3,
        sticky="w",
        pady=(12, 0),
    )

    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    """根据是否传入参数选择 GUI 或 CLI 入口。

    Args:
        argv: 命令行参数；为 None 时读取 `sys.argv[1:]`。

    Returns:
        进程退出码。
    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return run_gui()
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
