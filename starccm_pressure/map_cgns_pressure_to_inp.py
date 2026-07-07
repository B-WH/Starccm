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
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np

from starccm_pressure.optional_deps import load_ckdtree


StepKind = Literal["steady_state", "dynamic_explicit", "dynamic", "unknown"]
FrequencyGroupMode = Literal["none", "groups", "bandwidth"]
ProgressCallback = Callable[[dict[str, Any]], None]
PREPRINT_NO_ECHO_LINE = "*Preprint, echo=NO, model=NO, history=NO"


def _report_mapping_progress(
    progress_callback: ProgressCallback | None,
    current: int,
    total: int,
    message: str,
) -> None:
    """把映射进度统一转成 GUI/CLI 可消费的字典事件。"""
    if progress_callback is not None:
        progress_callback({"current": current, "total": total, "message": message})


def _resolve_worker_count(num_workers: int) -> int:
    """解析频率批次线程数；0 表示使用标准库默认上限策略。"""
    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers 必须大于或等于 0。")
    if workers == 0:
        return min(32, (os.cpu_count() or 1) + 4)
    return workers


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
        nodes_by_part: Part 内部节点坐标，用于装配体实例集合定位。
        elements_by_part: Part 内部单元连接，用于装配体实例集合定位。
        nsets: 节点集名称到节点编号集合的映射。
        elsets: 单元集名称到单元编号集合的映射。
        set_instances: 集合名称到实例名称的映射；多实例共名时标记为 None。
        instance_parts: 实例名称到 Part 名称的映射。
        steps: 文件中识别到的 Abaqus 分析步。
    """

    lines: list[str]
    nodes: dict[int, np.ndarray]
    elements: dict[int, AbaqusElement]
    nodes_by_part: dict[str, dict[int, np.ndarray]]
    elements_by_part: dict[str, dict[int, AbaqusElement]]
    nsets: dict[str, set[int]]
    elsets: dict[str, set[int]]
    set_instances: dict[tuple[str, str], str | None]
    instance_parts: dict[str, str]
    steps: list[AbaqusStep]


@dataclass(frozen=True)
class TargetFaces:
    """结构模型中实际承受流体压力的目标面集合。

    Attributes:
        node_ids: 每个目标面的节点编号。
        face_points: 每个目标面的节点坐标，顺序与 node_ids 一致。
        centers: 目标面中心坐标。
        area_vectors: 目标面面积向量，方向用于确定压力等效力方向。
    """

    node_ids: list[list[int]]
    face_points: list[np.ndarray]
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


@dataclass(frozen=True)
class AlignmentPreview:
    """CGNS 与 INP 目标区域的位置预览数据。"""

    source_centers: np.ndarray
    target_centers: np.ndarray
    target_nodes: np.ndarray
    source_count: int
    target_face_count: int
    target_node_count: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    nearest_distance_min: float
    nearest_distance_mean: float
    nearest_distance_max: float


@dataclass(frozen=True)
class ConsistentForcePlan:
    """一致节点力映射的预计算计划。

    Attributes:
        indices: 每个积分点对应的近邻 CGNS 面索引。
        weights: 近邻插值权重，顺序与 indices 一致。
        distance_stats: 近邻距离统计，用于写入映射报告。
        source_count: CGNS 源面数量，用于校验压力行长度。
        node_ids: 参与受载的全局节点编号。
        scatter_point_indices: 积分点到散布项的索引。
        scatter_node_indices: 散布项对应的目标节点索引。
        scatter_shape_values: 散布项的形函数权重。
        scatter_area_vectors: 散布项对应的积分面面积向量。
    """

    indices: np.ndarray
    weights: np.ndarray
    distance_stats: dict[str, float]
    source_count: int
    node_ids: np.ndarray
    scatter_point_indices: np.ndarray
    scatter_node_indices: np.ndarray
    scatter_shape_values: np.ndarray
    scatter_area_vectors: np.ndarray

    @property
    def integration_point_count(self) -> int:
        """返回预计算计划中的积分点数量。"""
        return int(self.indices.shape[0])


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
                raise ValueError("generate 集合行必须包含 start、end、increment 三个值。")
            values.update(range(numbers[0], numbers[1] + 1, numbers[2]))
        else:
            values.update(numbers)
        index += 1
    return values, index


def _remember_set_instance(
    set_instances: dict[tuple[str, str], str | None],
    set_type: str,
    set_name: str,
    instance_name: str,
) -> None:
    """记录集合所属实例；同名集合跨实例出现时禁用实例前缀。"""
    key = (set_type.lower(), set_name.upper())
    previous = set_instances.get(key)
    if previous is None and key in set_instances:
        return
    set_instances[key] = instance_name if previous in {None, instance_name} else None


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
    nodes_by_part: dict[str, dict[int, np.ndarray]] = {}
    elements_by_part: dict[str, dict[int, AbaqusElement]] = {}
    nsets: dict[str, set[int]] = {}
    elsets: dict[str, set[int]] = {}
    set_instances: dict[tuple[str, str], str | None] = {}
    instance_parts: dict[str, str] = {}
    raw_steps: list[tuple[str, int, int, list[str]]] = []
    current_part: str | None = None

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        lower = stripped.lower()
        if not stripped or stripped.startswith("**"):
            index += 1
            continue
        if lower.startswith("*include"):
            raise ValueError("暂不支持嵌套 *INCLUDE 文件。")
        if not stripped.startswith("*"):
            index += 1
            continue

        keyword = _keyword_name(stripped)
        params = _parse_keyword_params(stripped)

        if keyword == "*part":
            current_part = str(params.get("name", "")).strip() or None
            index += 1
            continue

        if keyword == "*end part":
            current_part = None
            index += 1
            continue

        if keyword == "*instance":
            instance_name = str(params.get("name", "")).strip()
            part_name = str(params.get("part", "")).strip()
            if instance_name and part_name:
                instance_parts[instance_name] = part_name
            index += 1
            continue

        if keyword == "*node":
            index += 1
            while index < len(lines) and not lines[index].lstrip().startswith("*"):
                row = lines[index].strip()
                if row and not row.startswith("**"):
                    parts = [part.strip() for part in row.split(",")]
                    if len(parts) < 4:
                        raise ValueError(f"无效的 *Node 行：{row}")
                    nodes[int(parts[0])] = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=float,
                    )
                    if current_part:
                        nodes_by_part.setdefault(current_part, {})[int(parts[0])] = nodes[
                            int(parts[0])
                        ]
                index += 1
            continue

        if keyword == "*element":
            element_type = str(params.get("type", "")).upper()
            header_elset = str(params.get("elset", "")).strip()
            if not element_type:
                raise ValueError("*Element 关键字必须包含 type=。")
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
                    if current_part:
                        elements_by_part.setdefault(current_part, {})[
                            element_id
                        ] = elements[element_id]
                    if header_elset:
                        elsets.setdefault(header_elset.upper(), set()).add(element_id)
                index += 1
            continue

        if keyword in {"*nset", "*elset"}:
            name_key = "nset" if keyword == "*nset" else "elset"
            set_name = str(params.get(name_key, "")).strip()
            if not set_name:
                raise ValueError(f"{keyword} 必须包含 {name_key}=。")
            generated = bool(params.get("generate", False))
            values, index = _parse_set_values(lines, index + 1, generated)
            target = nsets if keyword == "*nset" else elsets
            target.setdefault(set_name.upper(), set()).update(values)
            instance_name = str(params.get("instance", "")).strip()
            if instance_name:
                _remember_set_instance(
                    set_instances,
                    name_key,
                    set_name,
                    instance_name,
                )
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
        nodes_by_part=nodes_by_part,
        elements_by_part=elements_by_part,
        nsets=nsets,
        elsets=elsets,
        set_instances=set_instances,
        instance_parts=instance_parts,
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
    raise ValueError(f"不支持从该单元类型提取表面：{element_type}")


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
        raise ValueError("面至少需要三个节点。")
    vector = np.zeros(3, dtype=float)
    origin = points[0]
    for index in range(1, points.shape[0] - 1):
        vector += 0.5 * np.cross(points[index] - origin, points[index + 1] - origin)
    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0.0:
        raise ValueError("遇到退化的目标面。")
    return vector


def _lookup_case_insensitive(mapping: dict[str, Any], name: str) -> Any | None:
    """按 Abaqus 名称习惯执行大小写不敏感查找。"""
    for key, value in mapping.items():
        if key.upper() == name.upper():
            return value
    return None


def _target_instance_name(
    model: AbaqusModel,
    target_set: str,
    target_set_type: Literal["nset", "elset"],
) -> str | None:
    """返回目标集合显式绑定的实例名。"""
    return model.set_instances.get((target_set_type.lower(), target_set.upper()))


def _target_part_name(
    model: AbaqusModel,
    target_set: str,
    target_set_type: Literal["nset", "elset"],
) -> str | None:
    """根据目标集合的实例名找到所属 Part。"""
    instance_name = _target_instance_name(model, target_set, target_set_type)
    if not instance_name:
        return None
    part_name = _lookup_case_insensitive(model.instance_parts, instance_name)
    if not part_name:
        raise ValueError(
            f"目标 {target_set_type} '{target_set}' 引用了实例 "
            f"'{instance_name}'，但找不到匹配的 *Instance 定义。"
        )
    return str(part_name)


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
    part_name = _target_part_name(model, target_set, target_set_type)
    nodes = model.nodes
    elements = model.elements
    if part_name is not None:
        nodes = _lookup_case_insensitive(model.nodes_by_part, part_name)
        elements = _lookup_case_insensitive(model.elements_by_part, part_name)
        if nodes is None or elements is None:
            raise ValueError(
                f"目标 {target_set_type} '{target_set}' 引用了 Part "
                f"'{part_name}'，但未解析到该 Part 的网格。"
            )

    if target_set_type.lower() == "elset":
        if normalized_set not in model.elsets:
            raise ValueError(f"找不到目标 elset '{target_set}'。")
        element_ids = model.elsets[normalized_set]
        face_counts: dict[tuple[int, ...], int] = {}
        candidate_faces: list[list[int]] = []
        for element_id in element_ids:
            element = elements.get(element_id)
            if element is None:
                raise ValueError(f"Elset '{target_set}' 引用了缺失的单元 {element_id}。")
            for face in _element_face_node_ids(element):
                candidate_faces.append(face)
                face_counts[tuple(sorted(face))] = face_counts.get(tuple(sorted(face)), 0) + 1
        for face in candidate_faces:
            if face_counts[tuple(sorted(face))] == 1:
                selected_faces.append(face)
    elif target_set_type.lower() == "nset":
        if normalized_set not in model.nsets:
            raise ValueError(f"找不到目标 nset '{target_set}'。")
        selected_nodes = model.nsets[normalized_set]
        # 建立节点到单元的反向索引，只检查至少引用一个目标节点的单元。
        # 若每次都扫描全模型，大型 INP 中的小受湿面会成为明显瓶颈。
        node_to_element_ids: dict[int, set[int]] = {}
        for element_id, element in elements.items():
            for node_id in element.node_ids:
                node_to_element_ids.setdefault(node_id, set()).add(element_id)
        candidate_ids: set[int] = set()
        for node_id in selected_nodes:
            candidate_ids.update(node_to_element_ids.get(node_id, ()))
        for element_id in candidate_ids:
            element = elements[element_id]
            for face in _element_face_node_ids(element):
                if all(node_id in selected_nodes for node_id in face):
                    selected_faces.append(face)
    else:
        raise ValueError("target_set_type 必须是 'nset' 或 'elset'。")

    if not selected_faces:
        raise ValueError(f"目标 {target_set_type} '{target_set}' 没有生成任何受载面。")

    centers: list[np.ndarray] = []
    area_vectors: list[np.ndarray] = []
    face_points: list[np.ndarray] = []
    for face in selected_faces:
        try:
            points = np.vstack([nodes[node_id] for node_id in face])
        except KeyError as exc:
            raise ValueError(f"面引用了缺失的节点 {exc.args[0]}。") from exc
        face_points.append(points)
        centers.append(np.mean(points, axis=0))
        area_vectors.append(_area_vector(points))
    return TargetFaces(
        node_ids=selected_faces,
        face_points=face_points,
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
        raise ValueError("coordinates 必须为形状 (N, 3) 的数组。")
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError("axis_order 必须是 0、1、2 的排列。")
    transformed = values[:, axis_order] * np.asarray(axis_sign, dtype=float)
    transformed = transformed * float(scale)
    if translate is not None:
        transformed = transformed + np.asarray(translate, dtype=float)
    return transformed


def _coordinate_transform_matrix(
    scale: float = 1.0,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """构造坐标轴重排、符号翻转和尺度缩放对应的线性矩阵。"""
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError("axis_order 必须是 0、1、2 的排列。")
    matrix = np.zeros((3, 3), dtype=float)
    for row, source_axis in enumerate(axis_order):
        matrix[row, source_axis] = float(axis_sign[row]) * float(scale)
    return matrix


def transform_area_vectors(
    area_vectors: np.ndarray,
    scale: float = 1.0,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """将 CGNS 有向面积矢量转换到 INP 坐标系。"""
    values = np.asarray(area_vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("area_vectors 必须为形状 (N, 3) 的数组。")
    matrix = _coordinate_transform_matrix(
        scale=scale,
        axis_order=axis_order,
        axis_sign=axis_sign,
    )
    determinant = float(np.linalg.det(matrix))
    if determinant == 0.0:
        raise ValueError("转换面积向量时 scale 不能为 0。")
    area_transform = determinant * np.linalg.inv(matrix).T
    return values @ area_transform.T


def _load_ckdtree() -> Any | None:
    """在安装 SciPy 时返回 scipy.spatial.cKDTree，否则返回 None。"""
    return load_ckdtree()


def _query_ckdtree(tree: Any, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """兼容不同 SciPy 版本的 cKDTree.query 并行参数。"""
    try:
        return tree.query(points, k=k, workers=-1)
    except TypeError:
        return tree.query(points, k=k)


def _nearest_weights_bruteforce(
    target_centers: np.ndarray,
    source_centers: np.ndarray,
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用分块向量化扫描计算反距离近邻权重。

    目标点按内存上限分块处理，避免逐行 Python 循环开销，也避免一次性构造
    O(N_targets * N_sources) 完整距离矩阵造成的峰值内存占用。
    """
    n_targets = target_centers.shape[0]
    n_sources = source_centers.shape[0]
    indices = np.empty((n_targets, neighbor_count), dtype=int)
    weights = np.empty((n_targets, neighbor_count), dtype=float)

    max_pairs_per_chunk = 2_000_000
    chunk_size = max(1, min(n_targets, max_pairs_per_chunk // max(n_sources, 1)))

    for start in range(0, n_targets, chunk_size):
        stop = min(start + chunk_size, n_targets)
        chunk = target_centers[start:stop]  # (C, 3)
        # (C, S) 距离矩阵：每个分块只做一次向量化计算。
        delta = chunk[:, np.newaxis, :] - source_centers[np.newaxis, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))

        # 每行取最近 k 个并排序
        col_idx = np.argpartition(distances, neighbor_count - 1, axis=1)[
            :, :neighbor_count
        ]
        row_idx = np.arange(col_idx.shape[0])[:, np.newaxis]
        sorted_order = np.argsort(distances[row_idx, col_idx], axis=1)
        nearest = col_idx[row_idx, sorted_order]  # (C, k)
        nearest_distances = distances[row_idx, nearest]  # (C, k)

        # 反距离权重
        exact = nearest_distances[:, 0] <= 1.0e-12
        inv = 1.0 / np.maximum(nearest_distances, np.finfo(float).tiny)
        w = inv / np.sum(inv, axis=1, keepdims=True)
        w[exact, :] = 0.0
        w[exact, 0] = 1.0

        indices[start:stop] = nearest
        weights[start:stop] = w

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

    distances, indices = _query_ckdtree(tree_type(source_centers), target_centers, neighbor_count)
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


def _nearest_distances(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """返回每个源点到最近目标点的距离。"""
    sources = np.asarray(source_points, dtype=float)
    targets = np.asarray(target_points, dtype=float)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("source_points 必须为形状 (N, 3) 的数组。")
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target_points 必须为形状 (N, 3) 的数组。")
    if sources.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("source_points 和 target_points 不能为空。")

    tree_type = _load_ckdtree()
    if tree_type is not None:
        distances, _indices = _query_ckdtree(tree_type(targets), sources, 1)
        return np.asarray(distances, dtype=float)

    distances = np.empty(sources.shape[0], dtype=float)
    max_pairs_per_chunk = 2_000_000
    chunk_size = max(1, min(sources.shape[0], max_pairs_per_chunk // targets.shape[0]))
    for start in range(0, sources.shape[0], chunk_size):
        stop = min(start + chunk_size, sources.shape[0])
        delta = sources[start:stop, np.newaxis, :] - targets[np.newaxis, :, :]
        distance_squared = np.sum(delta * delta, axis=2)
        distances[start:stop] = np.sqrt(np.min(distance_squared, axis=1))
    return distances


def _sample_rows_evenly(points: np.ndarray, limit: int) -> np.ndarray:
    """等间隔抽取不超过 limit 行；小数组直接原样返回。"""
    if limit <= 0 or points.shape[0] <= limit:
        return points
    indices = np.linspace(0, points.shape[0] - 1, limit).astype(int)
    return points[indices]


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
        raise ValueError("target_centers 必须为形状 (N, 3) 的数组。")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_centers 必须为形状 (N, 3) 的数组。")
    if source.shape[0] != pressure.shape[0]:
        raise ValueError("source_pressure 的长度必须与 source_centers 匹配。")
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
        raise ValueError("face_pressures 和 target_area_vectors 的行数必须一致。")
    forces: dict[int, np.ndarray] = {}
    for face_index, nodes in enumerate(target_node_ids):
        face_force = -pressures[face_index] * area_vectors[face_index]
        contribution = np.asarray(face_force) / float(len(nodes))
        for node_id in nodes:
            if node_id not in forces:
                forces[node_id] = np.zeros(3, dtype=contribution.dtype)
            forces[node_id] = forces[node_id] + contribution
    return forces


def _tri3_quadrature(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """返回 TRI3 面的三点积分规则。"""
    area_vector = 0.5 * np.cross(points[1] - points[0], points[2] - points[0])
    barycentric_points = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=float,
    )
    return [
        (shape, shape @ points, area_vector / 3.0)
        for shape in barycentric_points
    ]


def _quad4_shape_functions(xi: float, eta: float) -> np.ndarray:
    """计算四节点四边形在自然坐标下的形函数。"""
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def _quad4_shape_derivatives(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """计算 QUAD4 形函数对自然坐标 xi/eta 的导数。"""
    dxi = 0.25 * np.array(
        [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)],
        dtype=float,
    )
    deta = 0.25 * np.array(
        [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi],
        dtype=float,
    )
    return dxi, deta


def _quad4_quadrature(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """返回 QUAD4 面的 2x2 高斯积分点和面积权重。"""
    gauss = 1.0 / np.sqrt(3.0)
    result: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for xi, eta in [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]:
        shape = _quad4_shape_functions(xi, eta)
        dxi, deta = _quad4_shape_derivatives(xi, eta)
        dx_dxi = dxi @ points
        dx_deta = deta @ points
        area_vector_weight = np.cross(dx_dxi, dx_deta)
        result.append((shape, shape @ points, area_vector_weight))
    return result


def _face_quadrature(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """按面节点数选择 TRI3 或 QUAD4 积分规则。"""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("目标面节点坐标必须为形状 (N, 3) 的数组。")
    if points.shape[0] == 3:
        return _tri3_quadrature(points)
    if points.shape[0] == 4:
        return _quad4_quadrature(points)
    raise ValueError("一致节点力积分只支持 3 节点或 4 节点面。")


def pressure_faces_to_consistent_node_forces(
    target_face_points: list[np.ndarray],
    target_node_ids: list[list[int]],
    source_centers: np.ndarray,
    source_pressure: np.ndarray,
    k: int = 4,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """将面压力通过形函数积分为一致等效节点力。

    压力先在每个目标面的积分点上由 CGNS 源面心插值得到，再按
    `F_i = -∫ N_i p n dA` 累积到该面的节点。
    """
    plan = build_consistent_force_plan(
        target_face_points,
        target_node_ids,
        source_centers,
        k=k,
    )
    return apply_consistent_force_plan(plan, source_pressure)


def build_consistent_force_plan(
    target_face_points: list[np.ndarray],
    target_node_ids: list[list[int]],
    source_centers: np.ndarray,
    k: int = 4,
) -> ConsistentForcePlan:
    """构建一致节点力映射所需的纯几何插值数据。"""
    if len(target_face_points) != len(target_node_ids):
        raise ValueError("target_face_points 的长度必须与 target_node_ids 匹配。")

    integration_points: list[np.ndarray] = []
    plan_node_ids: list[int] = []
    node_index_by_id: dict[int, int] = {}
    scatter_point_indices: list[int] = []
    scatter_node_indices: list[int] = []
    scatter_shape_values: list[float] = []
    scatter_area_vectors: list[np.ndarray] = []
    for points, node_ids in zip(target_face_points, target_node_ids):
        point_array = np.asarray(points, dtype=float)
        if point_array.shape[0] != len(node_ids):
            raise ValueError("每个目标面的坐标点数量必须与节点数量一致。")
        for shape, location, area_vector_weight in _face_quadrature(point_array):
            point_index = len(integration_points)
            integration_points.append(location)
            for local_index, node_id in enumerate(node_ids):
                if node_id not in node_index_by_id:
                    node_index_by_id[node_id] = len(plan_node_ids)
                    plan_node_ids.append(node_id)
                scatter_point_indices.append(point_index)
                scatter_node_indices.append(node_index_by_id[node_id])
                scatter_shape_values.append(float(shape[local_index]))
                scatter_area_vectors.append(area_vector_weight)

    integration_point_array = np.vstack(integration_points)
    source = np.asarray(source_centers, dtype=float)
    indices, weights = _nearest_weights(integration_point_array, source, k)
    nearest_distances = np.linalg.norm(source[indices[:, 0]] - integration_point_array, axis=1)
    return ConsistentForcePlan(
        indices=indices,
        weights=weights,
        distance_stats={
            "max_nearest_distance": float(np.max(nearest_distances)),
            "mean_nearest_distance": float(np.mean(nearest_distances)),
        },
        source_count=int(source.shape[0]),
        node_ids=np.asarray(plan_node_ids, dtype=int),
        scatter_point_indices=np.asarray(scatter_point_indices, dtype=int),
        scatter_node_indices=np.asarray(scatter_node_indices, dtype=int),
        scatter_shape_values=np.asarray(scatter_shape_values, dtype=float),
        scatter_area_vectors=np.vstack(scatter_area_vectors),
    )


def _node_force_dict(node_ids: np.ndarray, node_force_array: np.ndarray) -> dict[int, np.ndarray]:
    """把按数组存储的节点力转换回节点编号字典。"""
    return {
        int(node_id): np.asarray(node_force_array[index]).copy()
        for index, node_id in enumerate(node_ids)
    }


def apply_consistent_force_plan_batch(
    plan: ConsistentForcePlan,
    source_pressures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """一次性计算每个频率行的一致节点力。

    热点路径是按形函数权重散布累加压力贡献。这里按分量计算，使最大临时数组
    保持为 ``(n_freqs, n_scatter)``，避免生成
    ``(n_freqs, n_scatter, 3)``，在积分点很多时可降低散布数组峰值内存。
    """
    pressures = np.asarray(source_pressures)
    if pressures.ndim != 2:
        raise ValueError("source_pressures 必须为形状 (frequency_count, source_count) 的数组。")
    if pressures.shape[1] != plan.source_count:
        raise ValueError("source_pressures 的第二维必须与 source_centers 匹配。")

    # (n_freqs, n_integration_points)：把源面压力插值到每个目标面积分点。
    pressure_at_points = np.sum(
        pressures[:, plan.indices] * plan.weights[np.newaxis, :, :],
        axis=2,
    )

    n_freqs = pressures.shape[0]
    n_nodes = plan.node_ids.shape[0]
    n_scatter = plan.scatter_point_indices.shape[0]

    # 预先展开形函数值和节点索引，分量循环中只保留一次乘法和 add.at。
    scattered_pressure = np.empty((n_freqs, n_scatter), dtype=pressure_at_points.dtype)
    # 取出每个散布位置对应的积分点压力。
    scattered_pressure[:, :] = pressure_at_points[:, plan.scatter_point_indices]

    forces = np.zeros((n_freqs, n_nodes, 3), dtype=pressure_at_points.dtype)
    rows = np.arange(n_freqs)[:, np.newaxis]
    columns = plan.scatter_node_indices[np.newaxis, :]

    for comp in range(3):
        contrib = (
            -scattered_pressure
            * plan.scatter_shape_values[np.newaxis, :]
            * plan.scatter_area_vectors[np.newaxis, :, comp]
        )
        np.add.at(forces[:, :, comp], (rows, columns), contrib)

    return forces, plan.node_ids, plan.distance_stats


def apply_consistent_force_plan(
    plan: ConsistentForcePlan,
    source_pressure: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """用预计算计划映射单个频率的复压力行。"""
    pressure = np.asarray(source_pressure)
    if pressure.shape[0] != plan.source_count:
        raise ValueError("source_pressure 的长度必须与 source_centers 匹配。")
    force_array, node_ids, stats = apply_consistent_force_plan_batch(
        plan,
        pressure[np.newaxis, :],
    )
    return _node_force_dict(node_ids, force_array[0]), stats


def compute_face_force_moment(
    centers: np.ndarray,
    area_vectors: np.ndarray,
    pressures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """由面压力计算总力和总力矩。

    力的方向约定与映射载荷一致：`face_force = -pressure * area_vector`。
    """
    center_array = np.asarray(centers, dtype=float)
    area_array = np.asarray(area_vectors, dtype=float)
    pressure_array = np.asarray(pressures)
    if center_array.ndim != 2 or center_array.shape[1] != 3:
        raise ValueError("centers 必须为形状 (N, 3) 的数组。")
    if area_array.shape != center_array.shape:
        raise ValueError("area_vectors 的形状必须与 centers 一致。")
    if pressure_array.shape[0] != center_array.shape[0]:
        raise ValueError("pressures 的长度必须与 centers 匹配。")
    face_forces = -pressure_array[:, np.newaxis] * area_array
    return (
        np.sum(face_forces, axis=0),
        np.sum(np.cross(center_array, face_forces), axis=0),
    )


def _compute_face_force_moment_batch(
    centers: np.ndarray,
    area_vectors: np.ndarray,
    pressures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """批量计算每个频率的总力和总力矩。"""
    center_array = np.asarray(centers, dtype=float)
    area_array = np.asarray(area_vectors, dtype=float)
    pressure_array = np.asarray(pressures)
    if pressure_array.ndim != 2:
        raise ValueError("pressures 必须为形状 (frequency_count, source_count) 的数组。")
    if center_array.ndim != 2 or center_array.shape[1] != 3:
        raise ValueError("centers 必须为形状 (N, 3) 的数组。")
    if area_array.shape != center_array.shape:
        raise ValueError("area_vectors 的形状必须与 centers 一致。")
    if pressure_array.shape[1] != center_array.shape[0]:
        raise ValueError("pressures 的第二维必须与 centers 匹配。")
    face_forces = -pressure_array[:, :, np.newaxis] * area_array[np.newaxis, :, :]
    return (
        np.sum(face_forces, axis=1),
        np.sum(np.cross(center_array[np.newaxis, :, :], face_forces), axis=1),
    )


def compute_node_force_moment(
    node_forces: dict[int, np.ndarray],
    node_coordinates: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """由节点集中力计算总力和总力矩。"""
    if not node_forces:
        raise ValueError("node_forces 不能为空。")
    dtype = np.result_type(*[np.asarray(force).dtype for force in node_forces.values()])
    total_force = np.zeros(3, dtype=dtype)
    total_moment = np.zeros(3, dtype=dtype)
    for node_id, force in node_forces.items():
        if node_id not in node_coordinates:
            raise ValueError(f"缺少节点 {node_id} 的坐标。")
        force_array = np.asarray(force, dtype=dtype)
        coordinate = np.asarray(node_coordinates[node_id], dtype=float)
        if force_array.shape != (3,) or coordinate.shape != (3,):
            raise ValueError("节点力和坐标条目都必须是长度为 3 的向量。")
        total_force += force_array
        total_moment += np.cross(coordinate, force_array)
    return total_force, total_moment


def _cross_matrix(vector: np.ndarray) -> np.ndarray:
    """把叉乘向量写成矩阵形式，便于组装力矩约束。"""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _node_coordinates_from_faces(
    target_node_ids: list[list[int]],
    target_face_points: list[np.ndarray],
) -> dict[int, np.ndarray]:
    """从目标面节点列表恢复节点编号到坐标的唯一映射。"""
    if len(target_node_ids) != len(target_face_points):
        raise ValueError("target_node_ids 和 target_face_points 的长度必须一致。")
    coordinates: dict[int, np.ndarray] = {}
    for node_ids, points in zip(target_node_ids, target_face_points):
        point_array = np.asarray(points, dtype=float)
        if point_array.shape[0] != len(node_ids) or point_array.shape[1] != 3:
            raise ValueError("每个目标面的节点编号和三维坐标点必须匹配。")
        for node_id, coordinate in zip(node_ids, point_array):
            if node_id in coordinates and not np.allclose(coordinates[node_id], coordinate):
                raise ValueError(f"节点 {node_id} 在不同面中的坐标不一致。")
            coordinates[node_id] = coordinate
    return coordinates


def apply_global_conservation_correction(
    node_forces: dict[int, np.ndarray],
    node_coordinates: dict[int, np.ndarray],
    desired_force: np.ndarray,
    desired_moment: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """施加最小范数修正，使总力和总力矩守恒。"""
    if not node_forces:
        raise ValueError("node_forces 不能为空。")
    node_ids = sorted(node_forces)
    dtype = np.result_type(
        np.asarray(desired_force).dtype,
        np.asarray(desired_moment).dtype,
        *[np.asarray(node_forces[node_id]).dtype for node_id in node_ids],
    )
    current_force, current_moment = compute_node_force_moment(node_forces, node_coordinates)
    desired_force_array = np.asarray(desired_force, dtype=dtype)
    desired_moment_array = np.asarray(desired_moment, dtype=dtype)
    residual = np.concatenate(
        [
            desired_force_array - current_force,
            desired_moment_array - current_moment,
        ]
    )

    constraint = np.zeros((6, 3 * len(node_ids)), dtype=float)
    for index, node_id in enumerate(node_ids):
        coordinate = np.asarray(node_coordinates[node_id], dtype=float)
        block = slice(3 * index, 3 * index + 3)
        constraint[0:3, block] = np.eye(3)
        constraint[3:6, block] = _cross_matrix(coordinate)

    gram = constraint @ constraint.T
    correction_flat = constraint.T @ (np.linalg.pinv(gram) @ residual)
    corrected = {
        node_id: np.asarray(node_forces[node_id], dtype=dtype).copy()
        for node_id in node_ids
    }
    for index, node_id in enumerate(node_ids):
        corrected[node_id] = corrected[node_id] + correction_flat[3 * index : 3 * index + 3]

    corrected_force, corrected_moment = compute_node_force_moment(corrected, node_coordinates)
    force_residual_before = desired_force_array - current_force
    moment_residual_before = desired_moment_array - current_moment
    force_residual_after = desired_force_array - corrected_force
    moment_residual_after = desired_moment_array - corrected_moment
    return corrected, {
        "force_residual_norm_before": float(np.linalg.norm(force_residual_before)),
        "moment_residual_norm_before": float(np.linalg.norm(moment_residual_before)),
        "force_residual_norm_after": float(np.linalg.norm(force_residual_after)),
        "moment_residual_norm_after": float(np.linalg.norm(moment_residual_after)),
        "constraint_rank": int(np.linalg.matrix_rank(constraint)),
    }


def _apply_global_conservation_correction_batch(
    node_ids: np.ndarray,
    node_forces: np.ndarray,
    node_coordinates: dict[int, np.ndarray],
    desired_force: np.ndarray,
    desired_moment: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """批量施加总力和总力矩守恒修正。"""
    if node_forces.ndim != 3 or node_forces.shape[2] != 3:
        raise ValueError("node_forces 必须为形状 (frequency_count, node_count, 3) 的数组。")
    if node_forces.shape[1] != len(node_ids):
        raise ValueError("node_forces 的节点维度必须与 node_ids 匹配。")

    coordinates = np.vstack(
        [np.asarray(node_coordinates[int(node_id)], dtype=float) for node_id in node_ids]
    )
    dtype = np.result_type(node_forces.dtype, np.asarray(desired_force).dtype, np.asarray(desired_moment).dtype)
    forces = np.asarray(node_forces, dtype=dtype)
    desired_force_array = np.asarray(desired_force, dtype=dtype)
    desired_moment_array = np.asarray(desired_moment, dtype=dtype)

    current_force = np.sum(forces, axis=1)
    current_moment = np.sum(np.cross(coordinates[np.newaxis, :, :], forces), axis=1)
    residual = np.concatenate(
        [
            desired_force_array - current_force,
            desired_moment_array - current_moment,
        ],
        axis=1,
    )

    constraint = np.zeros((6, 3 * len(node_ids)), dtype=float)
    for index, coordinate in enumerate(coordinates):
        block = slice(3 * index, 3 * index + 3)
        constraint[0:3, block] = np.eye(3)
        constraint[3:6, block] = _cross_matrix(coordinate)

    gram = constraint @ constraint.T
    correction_matrix = constraint.T @ np.linalg.pinv(gram)
    correction = (residual @ correction_matrix.T).reshape(forces.shape)
    corrected = forces + correction

    corrected_force = np.sum(corrected, axis=1)
    corrected_moment = np.sum(np.cross(coordinates[np.newaxis, :, :], corrected), axis=1)
    force_residual_before = desired_force_array - current_force
    moment_residual_before = desired_moment_array - current_moment
    force_residual_after = desired_force_array - corrected_force
    moment_residual_after = desired_moment_array - corrected_moment
    rank = int(np.linalg.matrix_rank(constraint))
    stats = [
        {
            "force_residual_norm_before": float(np.linalg.norm(force_residual_before[index])),
            "moment_residual_norm_before": float(np.linalg.norm(moment_residual_before[index])),
            "force_residual_norm_after": float(np.linalg.norm(force_residual_after[index])),
            "moment_residual_norm_after": float(np.linalg.norm(moment_residual_after[index])),
            "constraint_rank": rank,
        }
        for index in range(forces.shape[0])
    ]
    return corrected, stats


def _map_complex_pressure_batch_to_force_array(
    target_node_ids: list[list[int]],
    target_face_points: list[np.ndarray],
    source_centers: np.ndarray,
    source_area_vectors: np.ndarray,
    source_pressures: np.ndarray,
    *,
    apply_conservation: bool,
    consistent_force_plan: ConsistentForcePlan,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """批量完成频率行到 Abaqus 节点集中力的映射。"""
    force_array, node_ids, distance_stats = apply_consistent_force_plan_batch(
        consistent_force_plan,
        source_pressures,
    )
    base_stats: dict[str, float | int] = {
        "target_face_count": int(len(target_node_ids)),
        "target_node_count": int(len(node_ids)),
        "source_face_count": int(np.asarray(source_centers).shape[0]),
        "integration_point_count": consistent_force_plan.integration_point_count,
    }
    stats = [{**base_stats, **distance_stats} for _ in range(force_array.shape[0])]
    if apply_conservation:
        desired_force, desired_moment = _compute_face_force_moment_batch(
            source_centers,
            source_area_vectors,
            source_pressures,
        )
        node_coordinates = _node_coordinates_from_faces(target_node_ids, target_face_points)
        force_array, conservation_stats = _apply_global_conservation_correction_batch(
            node_ids,
            force_array,
            node_coordinates,
            desired_force,
            desired_moment,
        )
        for item, conservation in zip(stats, conservation_stats):
            item["conservation_enabled"] = 1
            item.update(conservation)
    else:
        for item in stats:
            item["conservation_enabled"] = 0
    return force_array, node_ids, stats


def _map_complex_pressure_batch_to_nodes(
    target_node_ids: list[list[int]],
    target_face_points: list[np.ndarray],
    source_centers: np.ndarray,
    source_area_vectors: np.ndarray,
    source_pressures: np.ndarray,
    *,
    apply_conservation: bool,
    consistent_force_plan: ConsistentForcePlan,
) -> tuple[list[dict[int, np.ndarray]], list[dict[str, float | int]]]:
    """批量完成频率行到 Abaqus 节点集中力的映射。"""
    force_array, node_ids, stats = _map_complex_pressure_batch_to_force_array(
        target_node_ids,
        target_face_points,
        source_centers,
        source_area_vectors,
        source_pressures,
        apply_conservation=apply_conservation,
        consistent_force_plan=consistent_force_plan,
    )
    return (
        [_node_force_dict(node_ids, force_array[index]) for index in range(force_array.shape[0])],
        stats,
    )


def map_complex_pressure_to_nodes(
    target_centers: np.ndarray,
    target_area_vectors: np.ndarray,
    target_node_ids: list[list[int]],
    source_centers: np.ndarray,
    source_pressure: np.ndarray,
    k: int = 4,
    target_face_points: list[np.ndarray] | None = None,
    source_area_vectors: np.ndarray | None = None,
    apply_conservation: bool = True,
    consistent_force_plan: ConsistentForcePlan | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, float | int]]:
    """完成频域压力到节点复数力的一步映射。

    Args:
        target_centers: 结构目标面中心。
        target_area_vectors: 结构目标面面积向量。
        target_node_ids: 结构目标面节点编号。
        source_centers: CGNS 源表面面心。
        source_pressure: CGNS 源表面复数压力。
        k: 反距离插值使用的最近邻数量。
        target_face_points: 结构目标面节点坐标；提供时使用一致等效节点力积分。
        source_area_vectors: CGNS 源表面面积矢量；提供时可做全局保守修正。
        apply_conservation: 是否启用总力和总力矩保守修正。

    Returns:
        `(节点复数力, 映射统计信息)`。
    """
    if target_face_points is None:
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
        integration_point_count = int(len(target_node_ids))
    else:
        if consistent_force_plan is None:
            consistent_force_plan = build_consistent_force_plan(
                target_face_points,
                target_node_ids,
                source_centers,
                k=k,
            )
        node_forces, distance_stats = apply_consistent_force_plan(
            consistent_force_plan,
            source_pressure,
        )
        integration_point_count = consistent_force_plan.integration_point_count
    stats: dict[str, float | int] = {
        "target_face_count": int(len(target_node_ids)),
        "target_node_count": int(len(node_forces)),
        "source_face_count": int(np.asarray(source_centers).shape[0]),
        "integration_point_count": integration_point_count,
    }
    if apply_conservation and source_area_vectors is not None and target_face_points is not None:
        source_force, source_moment = compute_face_force_moment(
            source_centers,
            source_area_vectors,
            source_pressure,
        )
        node_coordinates = _node_coordinates_from_faces(target_node_ids, target_face_points)
        node_forces, conservation_stats = apply_global_conservation_correction(
            node_forces,
            node_coordinates,
            source_force,
            source_moment,
        )
        stats["conservation_enabled"] = 1
        stats.update(conservation_stats)
    else:
        stats["conservation_enabled"] = 0
    stats.update(distance_stats)
    return node_forces, stats


def _format_float(value: float) -> str:
    """用紧凑格式写出 Abaqus 输入文件中的浮点数。"""
    if value == 0.0:
        return "0"
    return f"{value:.8g}"


def write_frequency_load_include(
    path: str | Path,
    node_forces: dict[int, np.ndarray],
    frequency_hz: float,
    load_name: str = "CGNS_PRESSURE",
    node_label_prefix: str = "",
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
        "** 实部",
        "*CLOAD, REAL",
    ]
    for node_id in sorted(node_forces):
        force = np.asarray(node_forces[node_id])
        for component in range(3):
            value = float(np.real(force[component]))
            if value != 0.0:
                lines.append(
                    f"{_format_cload_node(node_id, node_label_prefix)}, "
                    f"{component + 1}, {_format_float(value)}"
                )
    lines.extend(["** 虚部", "*CLOAD, IMAGINARY"])
    for node_id in sorted(node_forces):
        force = np.asarray(node_forces[node_id])
        for component in range(3):
            value = float(np.imag(force[component]))
            if value != 0.0:
                lines.append(
                    f"{_format_cload_node(node_id, node_label_prefix)}, "
                    f"{component + 1}, {_format_float(value)}"
                )
    include_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_frequency_table_load_include(
    path: str | Path,
    node_force_maps: list[dict[int, np.ndarray]],
    frequencies_hz: list[float],
    load_name: str = "CGNS_PRESSURE",
    relative_zero_tolerance: float = 1.0e-12,
    node_label_prefix: str = "",
) -> dict[str, int | float]:
    """写出稳态动力学频率相关载荷 include 文件。

    对每个节点/方向/实部或虚部序列，若 ``max(abs(series))`` 不超过
    ``global_max_abs_force * relative_zero_tolerance`` 则跳过对应
    amplitude 和 *CLOAD，以减少近零载荷噪音。

    采用流式写入避免在内存中构建巨大的字符串列表；对于数千节点 ×
    数百频率的场景，可节省数百 MB 内存峰值。

    Args:
        path: include 文件输出路径。
        node_force_maps: 每个频率对应的节点复数力映射。
        frequencies_hz: 与 node_force_maps 对应的频率列表。
        load_name: 写入注释行的载荷名称。
        relative_zero_tolerance: 相对全局最大力幅值的近零过滤阈值。

    Returns:
        输出规模统计字典，包含频率数、载荷表数、有效 *CLOAD 数、
        近零分量跳过数、全局最大力幅值和近零阈值。
    """
    if len(node_force_maps) != len(frequencies_hz):
        raise ValueError("node_force_maps 和 frequencies_hz 的长度必须一致。")

    n_freqs = len(frequencies_hz)

    # 只保留当前节点/方向的短序列，避免重复构建完整力矩阵。
    all_node_ids = sorted(
        {node_id for node_forces in node_force_maps for node_id in node_forces}
    )
    global_max_abs_force = 0.0
    for node_forces in node_force_maps:
        for force in node_forces.values():
            force_max = float(np.max(np.abs(np.asarray(force, dtype=complex))))
            global_max_abs_force = max(global_max_abs_force, force_max)

    # ── 全局最大力幅值 ────────────────────────────────────────────
    threshold = global_max_abs_force * float(relative_zero_tolerance)

    # ── 预格式化频率字符串，避免循环内重复转换 ─────────────────
    freq_strings = [_format_float(float(f)) for f in frequencies_hz]

    include_path = Path(path)
    load_table_count = 0
    active_cload_count = 0
    skipped_near_zero = 0

    with include_path.open("w", encoding="utf-8") as fh:
        fh.write(f"** {load_name} 映射后的频率相关载荷\n")

        if global_max_abs_force == 0.0:
            fh.write("** 未生成非零稳态载荷。\n")
        else:
            for node_id in all_node_ids:
                for component in range(3):
                    component_series = [
                        complex(np.asarray(node_forces[node_id], dtype=complex)[component])
                        if node_id in node_forces
                        else 0.0 + 0.0j
                        for node_forces in node_force_maps
                    ]
                    real_series = [float(value.real) for value in component_series]
                    imag_series = [float(value.imag) for value in component_series]

                    real_max = max((abs(value) for value in real_series), default=0.0)
                    imag_max = max((abs(value) for value in imag_series), default=0.0)

                    if real_max > threshold:
                        amp_name = f"CGNS_R_N{node_id}_D{component + 1}"
                        fh.write(
                            f"*Amplitude, name={amp_name}, definition=TABULAR\n"
                        )
                        for freq_str, value in zip(freq_strings, real_series):
                            fh.write(
                                f"{freq_str}, {_format_float(float(value))}\n"
                            )
                        fh.write(f"*CLOAD, REAL, amplitude={amp_name}\n")
                        fh.write(
                            f"{_format_cload_node(node_id, node_label_prefix)}, "
                            f"{component + 1}, 1.\n"
                        )
                        load_table_count += 1
                        active_cload_count += 1
                    else:
                        skipped_near_zero += 1

                    if imag_max > threshold:
                        amp_name = f"CGNS_I_N{node_id}_D{component + 1}"
                        fh.write(
                            f"*Amplitude, name={amp_name}, definition=TABULAR\n"
                        )
                        for freq_str, value in zip(freq_strings, imag_series):
                            fh.write(
                                f"{freq_str}, {_format_float(float(value))}\n"
                            )
                        fh.write(f"*CLOAD, IMAGINARY, amplitude={amp_name}\n")
                        fh.write(
                            f"{_format_cload_node(node_id, node_label_prefix)}, "
                            f"{component + 1}, 1.\n"
                        )
                        load_table_count += 1
                        active_cload_count += 1
                    else:
                        skipped_near_zero += 1

    return {
        "frequency_count": n_freqs,
        "load_table_count": load_table_count,
        "active_cload_count": active_cload_count,
        "skipped_near_zero_components": skipped_near_zero,
        "global_max_abs_force": global_max_abs_force,
        "relative_zero_tolerance": float(relative_zero_tolerance),
    }


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
            f"时域载荷展开将写出 {records} 条记录，当前上限为 {max_records}。"
        )
    return records


def write_time_load_include(
    path: str | Path,
    node_force_series: dict[int, np.ndarray],
    times: np.ndarray,
    max_records: int = 500000,
    node_label_prefix: str = "",
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
    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1:
        raise ValueError("times 必须是一维数组。")
    active_records = 0
    force_series_by_node: dict[int, np.ndarray] = {}
    for node_id, force_series in node_force_series.items():
        values = np.asarray(force_series, dtype=float)
        if values.shape != (len(time_values), 3):
            raise ValueError("times 的长度必须与每个节点三向力序列匹配。")
        force_series_by_node[node_id] = values
        active_records += int(np.count_nonzero(values != 0.0))
    estimate_time_load_records(
        node_count=max(len(force_series_by_node), 1),
        component_count=3,
        sample_count=len(time_values),
        max_records=max_records,
    )
    lines = ["** CGNS_PRESSURE 映射后的时域载荷"]
    for node_id in sorted(force_series_by_node):
        values = force_series_by_node[node_id]
        for component in range(3):
            series = values[:, component]
            if not np.any(series != 0.0):
                continue
            amp_name = f"CGNS_N{node_id}_D{component + 1}"
            lines.append(f"*Amplitude, name={amp_name}, time=TOTAL TIME")
            pairs = [
                f"{_format_float(float(time_value))}, {_format_float(float(load_value))}"
                for time_value, load_value in zip(time_values, series)
            ]
            lines.extend(pairs)
            lines.append(f"*CLOAD, amplitude={amp_name}")
            lines.append(
                f"{_format_cload_node(node_id, node_label_prefix)}, "
                f"{component + 1}, 1."
            )
    if active_records == 0:
        lines.append("** 未生成非零时域载荷。")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_time_load_include_from_force_maps(
    path: str | Path,
    node_force_maps: Iterable[dict[int, np.ndarray]],
    times: np.ndarray,
    max_records: int = 500000,
    node_label_prefix: str = "",
) -> None:
    """按时间步节点力映射直接写出时域 include，避免额外堆叠节点时间序列。"""
    force_maps = list(node_force_maps)
    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1:
        raise ValueError("times 必须是一维数组。")
    if len(force_maps) != len(time_values):
        raise ValueError("times 的长度必须与节点力映射数量匹配。")
    node_ids = sorted({node_id for force_map in force_maps for node_id in force_map})
    estimate_time_load_records(
        node_count=max(len(node_ids), 1),
        component_count=3,
        sample_count=len(time_values),
        max_records=max_records,
    )

    lines = ["** CGNS_PRESSURE 映射后的时域载荷"]
    active_records = 0
    for node_id in node_ids:
        for component in range(3):
            pairs: list[str] = []
            has_nonzero = False
            for time_value, force_map in zip(time_values, force_maps):
                force = force_map.get(node_id)
                if force is None:
                    load_value = 0.0
                else:
                    force_values = np.asarray(force, dtype=float)
                    if force_values.shape != (3,):
                        raise ValueError("每个节点力条目必须包含三个分量。")
                    load_value = float(np.real(force_values[component]))
                if load_value != 0.0:
                    has_nonzero = True
                    active_records += 1
                pairs.append(f"{_format_float(float(time_value))}, {_format_float(load_value)}")
            if not has_nonzero:
                continue
            amp_name = f"CGNS_N{node_id}_D{component + 1}"
            lines.append(f"*Amplitude, name={amp_name}, time=TOTAL TIME")
            lines.extend(pairs)
            lines.append(f"*CLOAD, amplitude={amp_name}")
            lines.append(
                f"{_format_cload_node(node_id, node_label_prefix)}, "
                f"{component + 1}, 1."
            )
    if active_records == 0:
        lines.append("** 未生成非零时域载荷。")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_time_load_include_from_force_array(
    path: str | Path,
    node_ids: np.ndarray,
    node_force_array: np.ndarray,
    times: np.ndarray,
    max_records: int = 500000,
    node_label_prefix: str = "",
) -> None:
    """从 `(time, node, component)` 节点力数组写出时域 include，避免 dict/字符串堆叠。"""
    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1:
        raise ValueError("times 必须是一维数组。")
    ids = np.asarray(node_ids, dtype=int)
    forces = np.asarray(node_force_array)
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise ValueError("node_force_array 必须为形状 (time_count, node_count, 3) 的数组。")
    if forces.shape[0] != len(time_values):
        raise ValueError("times 的长度必须与 node_force_array 的时间维度匹配。")
    if forces.shape[1] != len(ids):
        raise ValueError("node_force_array 的节点维度必须与 node_ids 匹配。")
    estimate_time_load_records(
        node_count=max(len(ids), 1),
        component_count=3,
        sample_count=len(time_values),
        max_records=max_records,
    )

    active_records = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("** CGNS_PRESSURE 映射后的时域载荷\n")
        for node_index, node_id in enumerate(ids):
            for component in range(3):
                series = np.real(forces[:, node_index, component])
                if not np.any(series != 0.0):
                    continue
                active_records += int(np.count_nonzero(series != 0.0))
                amp_name = f"CGNS_N{int(node_id)}_D{component + 1}"
                handle.write(f"*Amplitude, name={amp_name}, time=TOTAL TIME\n")
                for time_value, load_value in zip(time_values, series):
                    handle.write(
                        f"{_format_float(float(time_value))}, "
                        f"{_format_float(float(load_value))}\n"
                    )
                handle.write(f"*CLOAD, amplitude={amp_name}\n")
                handle.write(
                    f"{_format_cload_node(int(node_id), node_label_prefix)}, "
                    f"{component + 1}, 1.\n"
                )
        if active_records == 0:
            handle.write("** 未生成非零时域载荷。\n")


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
        raise ValueError("提取结果中的频率数组为空。")
    index = int(np.argmin(np.abs(frequencies - float(target_frequency_hz))))
    delta = abs(float(frequencies[index]) - float(target_frequency_hz))
    if delta > float(tolerance_hz):
        raise ValueError(
            f"没有提取频率落在 {target_frequency_hz:g} Hz 的 "
            f"{tolerance_hz:g} Hz 容差内；最接近的是 {frequencies[index]:g} Hz。"
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
        raise FileNotFoundError(f"找不到必需的表面几何文件：{path}")
    with np.load(path) as saved:
        if "centers" not in saved or "area_vectors" not in saved:
            raise ValueError(f"{path} 必须包含 centers 和 area_vectors 数组。")
        return (
            np.asarray(saved["centers"], dtype=float),
            np.asarray(saved["area_vectors"], dtype=float),
        )


def _unique_target_nodes(target_faces: TargetFaces) -> np.ndarray:
    """按面遍历顺序收集不重复的目标节点坐标。"""
    seen: set[int] = set()
    points: list[np.ndarray] = []
    for node_ids, face_points in zip(target_faces.node_ids, target_faces.face_points):
        for node_id, point in zip(node_ids, face_points):
            if node_id in seen:
                continue
            seen.add(node_id)
            points.append(np.asarray(point, dtype=float))
    if not points:
        raise ValueError("目标面不包含任何节点。")
    return np.vstack(points)


def build_alignment_preview(
    inp_path: str | Path,
    extracted_dir: str | Path,
    target_set: str,
    target_set_type: Literal["nset", "elset"],
    scale: float = 1.0,
    translate: np.ndarray | None = None,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
    surface_geometry_name: str = "surface_geometry.npz",
    distance_sample_limit: int = 3000,
) -> AlignmentPreview:
    """为 GUI 对齐预览准备不改写原数据的 CGNS/INP 坐标。"""
    model = parse_inp_file(inp_path)
    target_faces = select_target_faces(model, target_set, target_set_type)
    source_centers, _source_area_vectors = _load_surface_geometry(
        Path(extracted_dir),
        surface_geometry_name,
    )
    transformed_source_centers = apply_coordinate_transform(
        source_centers,
        scale=scale,
        translate=translate,
        axis_order=axis_order,
        axis_sign=axis_sign,
    )
    target_nodes = _unique_target_nodes(target_faces)
    distance_sources = _sample_rows_evenly(transformed_source_centers, distance_sample_limit)
    distance_targets = _sample_rows_evenly(target_faces.centers, distance_sample_limit)
    nearest_distances = _nearest_distances(distance_sources, distance_targets)
    bounds_min = np.minimum.reduce(
        [
            np.min(transformed_source_centers, axis=0),
            np.min(target_faces.centers, axis=0),
            np.min(target_nodes, axis=0),
        ],
    )
    bounds_max = np.maximum.reduce(
        [
            np.max(transformed_source_centers, axis=0),
            np.max(target_faces.centers, axis=0),
            np.max(target_nodes, axis=0),
        ],
    )
    return AlignmentPreview(
        source_centers=transformed_source_centers,
        target_centers=target_faces.centers,
        target_nodes=target_nodes,
        source_count=int(transformed_source_centers.shape[0]),
        target_face_count=int(target_faces.centers.shape[0]),
        target_node_count=int(target_nodes.shape[0]),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        nearest_distance_min=float(np.min(nearest_distances)),
        nearest_distance_mean=float(np.mean(nearest_distances)),
        nearest_distance_max=float(np.max(nearest_distances)),
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
        raise FileNotFoundError(f"找不到必需的复数压力谱文件：{path}")
    with np.load(path) as saved:
        frequencies = np.asarray(saved["frequencies_hz"], dtype=float)
        pressure = (
            np.asarray(saved["pressure_real"], dtype=float)
            + 1j * np.asarray(saved["pressure_imag"], dtype=float)
        )
    return frequencies, pressure


def _discard_bytes(reader: Any, byte_count: int) -> None:
    """从压缩包成员流中丢弃指定字节数，避免读取整块数组。"""
    remaining = int(byte_count)
    while remaining > 0:
        chunk = reader.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("跳过行时 npz 数组意外结束。")
        remaining -= len(chunk)


def _read_npy_header(reader: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    """读取 npz 成员内嵌的 npy 头信息。"""
    version = np.lib.format.read_magic(reader)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(reader)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(reader)
    else:
        raise ValueError(f"npz 成员中的 npy 版本不受支持：{version}")
    return tuple(int(value) for value in shape), bool(fortran_order), np.dtype(dtype)


class _NpzArrayRowStream:
    """按行顺序读取 npz 内二维 npy 数组，降低大频谱内存占用。"""

    def __init__(self, archive: zipfile.ZipFile, member_name: str):
        """打开指定 npz 成员并记录数组布局。"""
        self._reader = archive.open(member_name)
        shape, fortran_order, dtype = _read_npy_header(self._reader)
        if fortran_order:
            raise ValueError(f"{member_name} 必须为 C 连续布局。")
        if len(shape) != 2:
            raise ValueError(f"{member_name} 必须是二维数组。")
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self._row_bytes = self.shape[1] * self.dtype.itemsize
        self._current_row = 0
        self._last_row_index: int | None = None
        self._last_row: np.ndarray | None = None

    def close(self) -> None:
        """关闭当前成员流。"""
        self._reader.close()

    def read_rows(self, row_indices: np.ndarray) -> np.ndarray:
        """按非递减行号读取数组行，允许重复读取上一行。"""
        rows = np.asarray(row_indices, dtype=int)
        result = np.empty((rows.size, self.shape[1]), dtype=self.dtype)
        for output_index, row_index in enumerate(rows):
            if row_index < 0 or row_index >= self.shape[0]:
                raise IndexError("压力谱行索引越界。")
            if row_index == self._last_row_index and self._last_row is not None:
                result[output_index] = self._last_row
                continue
            if row_index < self._current_row:
                raise ValueError(
                    "流式压力谱行必须按升序读取。"
                )
            _discard_bytes(self._reader, (row_index - self._current_row) * self._row_bytes)
            raw = self._reader.read(self._row_bytes)
            if len(raw) != self._row_bytes:
                raise ValueError("读取行时 npz 数组意外结束。")
            row = np.frombuffer(raw, dtype=self.dtype, count=self.shape[1]).copy()
            result[output_index] = row
            self._current_row = row_index + 1
            self._last_row_index = row_index
            self._last_row = row
        return np.asarray(result, dtype=float)


class _ComplexSpectrumRowReader:
    """成对流式读取 pressure_real/pressure_imag 频谱行。"""

    def __init__(self, path: Path):
        """记录频谱文件路径并预读取频率轴。"""
        if not path.exists():
            raise FileNotFoundError(f"找不到必需的复数压力谱文件：{path}")
        self.path = path
        self.frequencies = self._load_frequencies(path)
        self._archive: zipfile.ZipFile | None = None
        self._real: _NpzArrayRowStream | None = None
        self._imag: _NpzArrayRowStream | None = None
        self.source_count = 0

    @staticmethod
    def _load_frequencies(path: Path) -> np.ndarray:
        """仅读取频率轴，避免提前载入完整压力矩阵。"""
        with np.load(path) as saved:
            return np.asarray(saved["frequencies_hz"], dtype=float)

    def __enter__(self) -> "_ComplexSpectrumRowReader":
        """打开 npz 包和实部/虚部成员流。"""
        self._archive = zipfile.ZipFile(self.path)
        try:
            self._real = _NpzArrayRowStream(self._archive, "pressure_real.npy")
            self._imag = _NpzArrayRowStream(self._archive, "pressure_imag.npy")
            if self._real.shape != self._imag.shape:
                raise ValueError("pressure_real 和 pressure_imag 的形状必须一致。")
            if self._real.shape[0] != self.frequencies.size:
                raise ValueError("压力谱行数必须与 frequencies_hz 匹配。")
            self.source_count = self._real.shape[1]
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """释放打开的 npz 成员流和压缩包句柄。"""
        if self._real is not None:
            self._real.close()
            self._real = None
        if self._imag is not None:
            self._imag.close()
            self._imag = None
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def close(self) -> None:
        """显式关闭底层流，供非 with 调用路径使用。"""
        self.__exit__(None, None, None)

    def __del__(self) -> None:
        """兜底释放文件句柄。"""
        self.close()

    def read_complex_rows(self, row_indices: np.ndarray) -> np.ndarray:
        """读取指定频率行并合成为复数压力矩阵。"""
        if self._real is None or self._imag is None:
            raise RuntimeError("复数压力谱读取器尚未打开。")
        real = self._real.read_rows(row_indices)
        imag = self._imag.read_rows(row_indices)
        return real + 1j * imag


def _is_non_decreasing(values: list[int]) -> bool:
    """判断索引列表是否满足流式读取的非递减约束。"""
    return all(left <= right for left, right in zip(values, values[1:]))


def _target_node_label_prefix(
    model: AbaqusModel,
    target_set: str,
    target_set_type: Literal["nset", "elset"],
) -> str:
    """返回装配体实例节点在 *Cload 中需要使用的标签前缀。"""
    instance_name = model.set_instances.get(
        (target_set_type.lower(), target_set.upper())
    )
    return f"{instance_name}." if instance_name else ""


def _format_cload_node(node_id: int, node_label_prefix: str = "") -> str:
    """格式化 *Cload 节点标签，兼容实例前缀写法。"""
    return f"{node_label_prefix}{node_id}" if node_label_prefix else str(node_id)


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
        raise FileNotFoundError(f"找不到必需的时域压力文件：{path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    times = np.asarray(payload["time_s"], dtype=float)
    pressure = np.asarray(payload["pulsating_pressure"], dtype=float)
    if times.ndim != 1:
        raise ValueError("time_s 必须是一维数组。")
    if pressure.ndim != 2:
        raise ValueError("pulsating_pressure 必须为形状 (time_count, source_count) 的数组。")
    if pressure.shape[0] != times.shape[0]:
        raise ValueError("time_s 的长度必须与 pulsating_pressure 行数匹配。")
    if pressure.shape[0] == 0 or pressure.shape[1] == 0:
        raise ValueError("pulsating_pressure 至少需要包含一个时间样本和一个源面样本。")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(pressure)):
        raise ValueError("时域压力数据中包含 NaN 或无穷值。")
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


def ensure_preprint_echo_off(lines: list[str]) -> list[str]:
    """关闭 Abaqus 输入回显，避免大载荷 include 膨胀 .dat 文件。"""
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("**"):
            continue
        keyword = _keyword_name(stripped)
        if keyword == "*preprint":
            updated = list(lines)
            updated[index] = PREPRINT_NO_ECHO_LINE
            return updated
        if keyword == "*step":
            return lines[:index] + [PREPRINT_NO_ECHO_LINE] + lines[index:]
    return [PREPRINT_NO_ECHO_LINE] + list(lines)


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
    """描述压力映射报告中记录的物理假设。"""
    return {
        "pressure_sign": "-pressure * area_vector",
        "node_force_distribution": "consistent_shape_function_integration",
        "interpolation": "inverse_distance_weighting_at_face_integration_points",
        "global_conservation": "minimum_norm_total_force_and_moment_correction",
        "supported_element_scope": "linear shell/surface elements and first-order C3D4/C3D8 exterior faces",
    }


def _auto_batch_size(
    n_frequencies: int,
    n_sources: int,
    n_scatter: int,
    *,
    memory_limit_mb: float = 512.0,
    min_batch: int = 1,
) -> int:
    """选择频率批大小，使峰值内存控制在 *memory_limit_mb* 内。

    批处理中占主导的数组包括：

    * ``pressures``: ``(batch, n_sources)`` complex128
    * ``pressure_at_points``: ``(batch, n_integration_points)`` float64
    * ``scattered_pressure``: ``(batch, n_scatter)`` float64
    * ``forces``: ``(batch, n_nodes, 3)`` float64
    """
    bytes_per_complex = 16
    bytes_per_float = 8
    # 根据主要中间数组粗略估计每个频率所需字节数。
    per_freq_bytes = (
        n_sources * bytes_per_complex
        + n_scatter * bytes_per_float * 3  # 压力积分点、散布压力和贡献项。
        + n_scatter * bytes_per_float      # 形函数值和面积向量广播。
    )
    if per_freq_bytes <= 0:
        per_freq_bytes = 1
    limit_bytes = memory_limit_mb * 1024 * 1024
    batch = max(min_batch, min(n_frequencies, int(limit_bytes / per_freq_bytes)))
    return batch


def _default_output_path(inp_path: Path) -> Path:
    """根据原始 INP 路径生成默认输出路径。"""
    return inp_path.with_name(f"{inp_path.stem}_mapped{inp_path.suffix}")


def _frequency_suffix(frequency_hz: float) -> str:
    """将频率值转换为可用于文件名的后缀。"""
    text = f"{float(frequency_hz):g}".replace("-", "m").replace(".", "p")
    return f"{text}Hz"


def _split_frequency_groups(
    frequencies_hz: list[float],
    mode: FrequencyGroupMode,
    value: float | int | None,
) -> list[tuple[int, int]]:
    """把连续频率列表拆成输出组的半开区间。"""
    if mode == "none":
        return [(0, len(frequencies_hz))]
    if not frequencies_hz:
        return []
    if value is None:
        raise ValueError("启用频率分组时必须提供 frequency_group_value。")
    if mode == "groups":
        group_count_float = float(value)
        group_count = int(group_count_float)
        if group_count <= 0 or not math.isclose(group_count_float, float(group_count)):
            raise ValueError("groups 模式下 frequency_group_value 必须是正整数。")
        if group_count > len(frequencies_hz):
            raise ValueError("frequency_group_value 不能超过请求的频率数量。")
        base_size, extra = divmod(len(frequencies_hz), group_count)
        groups: list[tuple[int, int]] = []
        start = 0
        for group_index in range(group_count):
            size = base_size + (1 if group_index < extra else 0)
            end = start + size
            groups.append((start, end))
            start = end
        return groups
    if mode == "bandwidth":
        bandwidth = float(value)
        if not math.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("bandwidth 模式下 frequency_group_value 必须为正数。")
        groups: list[tuple[int, int]] = []
        start = 0
        tolerance = bandwidth * 1.0e-12
        while start < len(frequencies_hz):
            end = start + 1
            group_start = float(frequencies_hz[start])
            while (
                end < len(frequencies_hz)
                and float(frequencies_hz[end]) - group_start < bandwidth - tolerance
            ):
                end += 1
            groups.append((start, end))
            start = end
        return groups
    raise ValueError(f"不支持的 frequency_group_mode：{mode}")


def _grouped_output_path(output: Path, group_index: int, frequencies_hz: list[float]) -> Path:
    """生成带组号和频率范围的 INP 输出路径。"""
    start = _frequency_suffix(frequencies_hz[0])
    end = _frequency_suffix(frequencies_hz[-1])
    return output.with_name(f"{output.stem}_g{group_index:03d}_{start}-{end}{output.suffix}")


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
        raise ValueError("INP 文件中没有 Abaqus *Step。")
    if requested_step is None:
        return model.steps[0]
    for step in model.steps:
        if step.name.upper() == requested_step.upper():
            return step
    raise ValueError(f"找不到分析步 '{requested_step}'。")


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
    conserve_global_loads: bool = True,
    frequency_batch_size: int | None = None,
    frequency_group_mode: FrequencyGroupMode = "none",
    frequency_group_value: float | int | None = None,
    relative_zero_tolerance: float = 1.0e-12,
    show_progress: bool = True,
    progress_callback: ProgressCallback | None = None,
    num_workers: int = 1,
) -> MappingResult:
    """执行完整压力映射流程并写出新 INP。

    稳态动力学按指定频率读取复数压力谱；多个频率共享同一个分析步和
    include 文件。瞬态/显式动力学读取时域压力，并在写 include 前检查
    展开规模。

    Args:
        inp_path: 原始 Abaqus INP 文件路径。
        extracted_dir: CGNS 提取结果目录。
        target_set: 受载 Nset 或 Elset 名称。
        target_set_type: 目标集合类型。
        frequencies: 稳态动力学目标频率列表。
        output_path: 输出 INP 基础路径；为 None 时自动加 ``_mapped``。
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
        conserve_global_loads: 是否修正节点力以守恒源 CGNS 总力和总力矩。
        frequency_batch_size: 频率分块大小；为 None 时根据内存自动计算。
        frequency_group_mode: 输出 INP 分组方式；默认 ``none`` 保持单个输出。
        frequency_group_value: 分组值；``groups`` 时表示总组数，``bandwidth`` 时表示 Hz 带宽。
        relative_zero_tolerance: 相对全局最大力幅值的近零过滤阈值。
        show_progress: 是否在控制台输出进度信息；默认开启。
        progress_callback: 可选 GUI/调用方进度回调。
        num_workers: 并行处理频率批次的线程数；默认 1（串行）。
            设为 0 则使用 ``min(32, os.cpu_count() + 4)``。
            多线程通过 ``ThreadPoolExecutor`` 实现，利用 NumPy
            在多数运算中释放 GIL 的特性。

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
        raise ValueError("输出 INP 不能覆盖原始 INP。")
    if not target_set:
        raise ValueError("必须提供 target_set；不支持直接映射全部节点。")

    model = parse_inp_file(inp)
    step = _load_model_step(model, step_name)
    target_faces = select_target_faces(model, target_set, target_set_type)
    node_label_prefix = _target_node_label_prefix(model, target_set, target_set_type)
    source_centers, source_area_vectors = _load_surface_geometry(extracted, surface_geometry_name)
    transformed_source_centers = apply_coordinate_transform(
        source_centers,
        scale=scale,
        translate=translate,
        axis_order=axis_order,
        axis_sign=axis_sign,
    )
    transformed_source_area_vectors = transform_area_vectors(
        source_area_vectors,
        scale=scale,
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
        "load_node_label_prefix": node_label_prefix,
        "mapping_assumptions": _mapping_assumptions(),
    }
    progress_total: int | None = None

    if step.kind == "steady_state":
        if not frequencies:
            raise ValueError("稳态映射至少需要提供一个 --frequency。")

        if show_progress:
            print("加载复数压力谱…")
        spectrum_reader = _ComplexSpectrumRowReader(extracted / complex_spectrum_name)
        spectrum_reader.__enter__()
        extracted_frequencies = spectrum_reader.frequencies

        if show_progress:
            print(f"  源面数量: {spectrum_reader.source_count}, "
                  f"提取频率数: {len(extracted_frequencies)}")
            print("构建一致力积分方案…")
        consistent_force_plan = build_consistent_force_plan(
            target_faces.face_points,
            target_faces.node_ids,
            transformed_source_centers,
            k=k,
        )
        if show_progress:
            print(f"  积分点数: {consistent_force_plan.integration_point_count}, "
                  f"受载节点数: {len(consistent_force_plan.node_ids)}")

        spectrum_indices = [
            find_frequency_index(
                extracted_frequencies,
                float(requested_frequency),
                tolerance_hz=frequency_tolerance_hz,
            )
            for requested_frequency in frequencies
        ]
        if not _is_non_decreasing(spectrum_indices):
            spectrum_reader.close()
            raise ValueError(
                "流式频率映射要求请求频率按升序排列。"
            )
        mapped_frequencies = [
            float(extracted_frequencies[spectrum_index])
            for spectrum_index in spectrum_indices
        ]
        group_ranges = _split_frequency_groups(
            mapped_frequencies,
            frequency_group_mode,
            frequency_group_value,
        )

        # ── 频率分块 ────────────────────────────────────
        if frequency_batch_size is not None and frequency_batch_size > 0:
            batch_size = int(frequency_batch_size)
        else:
            batch_size = _auto_batch_size(
                len(spectrum_indices),
                n_sources=spectrum_reader.source_count,
                n_scatter=consistent_force_plan.indices.shape[0] * 4,
                memory_limit_mb=512.0,
            )
        batch_count = max(
            1,
            sum(
                (end - start + batch_size - 1) // batch_size
                for start, end in group_ranges
            ),
        )
        worker_count = _resolve_worker_count(num_workers)
        progress_total = batch_count + 4
        _report_mapping_progress(
            progress_callback,
            1,
            progress_total,
            "数据已加载",
        )
        _report_mapping_progress(
            progress_callback,
            2,
            progress_total,
            "积分方案已构建",
        )

        if show_progress and batch_count > 1:
            print(f"  频率分块: 每块 {batch_size} 个频率, 共 {batch_count} 块 "
                  f"(共 {len(spectrum_indices)} 个频率)")

        # 按输出组生成批次切片，避免先攒完整频带的节点力。
        def _group_batch_specs(start: int, end: int) -> list[tuple[int, np.ndarray]]:
            specs: list[tuple[int, np.ndarray]] = []
            for batch_start in range(start, end, batch_size):
                batch_end = min(batch_start + batch_size, end)
                specs.append(
                    (
                        batch_start,
                        np.asarray(
                            spectrum_indices[batch_start:batch_end],
                            dtype=int,
                        ),
                    )
                )
            return specs

        # 单个批次的工作函数（闭包捕获只读共享数据，线程安全）
        def _map_batch_pressures(batch_pressures: np.ndarray) -> tuple[
            list[dict[int, np.ndarray]],
            list[dict[str, float | int]],
        ]:
            """读取一个频率批次并映射成节点力。"""
            return _map_complex_pressure_batch_to_nodes(
                target_faces.node_ids,
                target_faces.face_points,
                transformed_source_centers,
                transformed_source_area_vectors,
                batch_pressures,
                apply_conservation=conserve_global_loads,
                consistent_force_plan=consistent_force_plan,
            )

        per_frequency_stats: list[dict[str, Any]] = []
        frequency_groups: list[dict[str, Any]] = []
        processed_batch_count = 0
        try:
            for group_number, (start, end) in enumerate(group_ranges, start=1):
                group_frequencies = mapped_frequencies[start:end]
                group_force_maps: list[dict[int, np.ndarray]] = []
                group_batch_stats: list[dict[str, float | int]] = []
                batch_specs = _group_batch_specs(start, end)

                def record_batch_result(
                    batch_number: int,
                    batch_force_maps: list[dict[int, np.ndarray]],
                    batch_chunk_stats: list[dict[str, float | int]],
                ) -> None:
                    nonlocal processed_batch_count
                    if show_progress and batch_count > 1:
                        print("    完成")
                    group_force_maps.extend(batch_force_maps)
                    group_batch_stats.extend(batch_chunk_stats)
                    processed_batch_count = batch_number
                    _report_mapping_progress(
                        progress_callback,
                        processed_batch_count + 2,
                        progress_total,
                        f"频率块 {processed_batch_count}/{batch_count}",
                    )

                if worker_count <= 1 or len(batch_specs) <= 1:
                    for spec in batch_specs:
                        next_batch_number = processed_batch_count + 1
                        if show_progress and batch_count > 1:
                            print(
                                f"  处理频率块 {next_batch_number}/{batch_count} "
                                f"({len(spec[1])} 个频率)..."
                            )
                        batch_pressures = spectrum_reader.read_complex_rows(spec[1])
                        batch_force_maps, batch_chunk_stats = _map_batch_pressures(
                            batch_pressures
                        )
                        record_batch_result(
                            next_batch_number,
                            batch_force_maps,
                            batch_chunk_stats,
                        )
                else:
                    pending: list[tuple[int, Any]] = []
                    next_batch_number = processed_batch_count
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        for spec in batch_specs:
                            next_batch_number += 1
                            if show_progress and batch_count > 1:
                                print(
                                    f"  处理频率块 {next_batch_number}/{batch_count} "
                                    f"({len(spec[1])} 个频率)..."
                                )
                            batch_pressures = spectrum_reader.read_complex_rows(spec[1])
                            pending.append(
                                (
                                    next_batch_number,
                                    executor.submit(_map_batch_pressures, batch_pressures),
                                )
                            )
                            if len(pending) >= worker_count:
                                batch_number, future = pending.pop(0)
                                batch_force_maps, batch_chunk_stats = future.result()
                                record_batch_result(
                                    batch_number,
                                    batch_force_maps,
                                    batch_chunk_stats,
                                )
                        while pending:
                            batch_number, future = pending.pop(0)
                            batch_force_maps, batch_chunk_stats = future.result()
                            record_batch_result(
                                batch_number,
                                batch_force_maps,
                                batch_chunk_stats,
                            )

                for requested_frequency, mapped_frequency, stats in zip(
                    frequencies[start:end],
                    group_frequencies,
                    group_batch_stats,
                ):
                    per_frequency_stats.append(
                        {
                            "requested_frequency_hz": float(requested_frequency),
                            "mapped_frequency_hz": float(mapped_frequency),
                            **stats,
                        }
                    )

                if show_progress:
                    n_nodes = len(set(nid for nf in group_force_maps for nid in nf))
                    print(
                        f"写出载荷 include ({len(group_force_maps)} 个频率, "
                        f"{n_nodes} 个受载节点)..."
                    )

                group_output = (
                    output
                    if frequency_group_mode == "none"
                    else _grouped_output_path(output, group_number, group_frequencies)
                )
                include_path = group_output.with_name(f"{group_output.stem}_loads.inc")
                if len(group_force_maps) == 1:
                    write_frequency_load_include(
                        include_path,
                        group_force_maps[0],
                        frequency_hz=group_frequencies[0],
                        node_label_prefix=node_label_prefix,
                    )
                    frequency_table_stats: dict[str, int | float] = {
                        "frequency_count": 1,
                        "load_table_count": 0,
                        "active_cload_count": 0,
                        "skipped_near_zero_components": 0,
                        "global_max_abs_force": 0.0,
                        "relative_zero_tolerance": float(relative_zero_tolerance),
                    }
                else:
                    frequency_table_stats = write_frequency_table_load_include(
                        include_path,
                        group_force_maps,
                        group_frequencies,
                        relative_zero_tolerance=relative_zero_tolerance,
                        node_label_prefix=node_label_prefix,
                    )
                include_paths.append(include_path)
                output_paths.append(group_output)
                frequency_groups.append(
                    {
                        "group_index": group_number,
                        "frequency_count": len(group_frequencies),
                        "frequency_start_hz": group_frequencies[0],
                        "frequency_end_hz": group_frequencies[-1],
                        "frequencies_hz": group_frequencies,
                        "output_inp": str(group_output),
                        "include_file": str(include_path),
                        "frequency_table_output": frequency_table_stats,
                    }
                )
        finally:
            spectrum_reader.close()
        _report_mapping_progress(
            progress_callback,
            progress_total - 1,
            progress_total,
            "载荷文件已写出",
        )
        report.update(
            {
                "frequencies_hz": mapped_frequencies,
                "mapping_stats": per_frequency_stats[0]
                if len(per_frequency_stats) == 1
                else per_frequency_stats,
                "frequency_table_output": frequency_groups[0]["frequency_table_output"]
                if len(frequency_groups) == 1
                else [group["frequency_table_output"] for group in frequency_groups],
                "frequency_group_mode": frequency_group_mode,
                "frequency_group_value": frequency_group_value,
                "frequency_groups": frequency_groups,
                "include_file_count": len(include_paths),
                "output_inp_count": len(output_paths),
                "output_inps": [str(path) for path in output_paths],
                "include_files": [str(path) for path in include_paths],
            }
        )
    elif step.kind in {"dynamic", "dynamic_explicit"}:
        times, pressure_time = _load_time_pressure(extracted, pressure_time_name)
        consistent_force_plan = build_consistent_force_plan(
            target_faces.face_points,
            target_faces.node_ids,
            transformed_source_centers,
            k=k,
        )
        time_sample_count = int(pressure_time.shape[0])
        progress_total = time_sample_count + 3
        _report_mapping_progress(
            progress_callback,
            1,
            progress_total,
            "数据已加载",
        )
        node_force_array, time_node_ids, time_stats = _map_complex_pressure_batch_to_force_array(
            target_faces.node_ids,
            target_faces.face_points,
            transformed_source_centers,
            transformed_source_area_vectors,
            pressure_time,
            apply_conservation=conserve_global_loads,
            consistent_force_plan=consistent_force_plan,
        )
        del pressure_time
        for time_index in range(time_sample_count):
            _report_mapping_progress(
                progress_callback,
                time_index + 2,
                progress_total,
                f"时间步 {time_index + 1}/{time_sample_count}",
            )
        include_path = output.with_name(f"{output.stem}_loads.inc")
        write_time_load_include_from_force_array(
            include_path,
            time_node_ids,
            node_force_array,
            times,
            max_records=max_time_records,
            node_label_prefix=node_label_prefix,
        )
        include_paths.append(include_path)
        output_paths.append(output)
        _report_mapping_progress(
            progress_callback,
            progress_total - 1,
            progress_total,
            "载荷文件已写出",
        )
        report.update(
            {
                "time_sample_count": time_sample_count,
                "mapping_stats": time_stats[-1] if time_stats else {},
                "include_file_count": len(include_paths),
                "output_inp_count": len(output_paths),
                "output_inps": [str(output)],
                "include_files": [str(include_path)],
            }
        )
    else:
        raise ValueError(
            "当前映射只支持 *STEADY STATE DYNAMICS 和 *DYNAMIC 分析步。"
        )

    if show_progress:
        print("写出新 INP 文件…")
    for output_file, include_file in zip(output_paths, include_paths):
        new_lines = insert_include_before_step_end(model.lines, step, include_file)
        new_lines = ensure_preprint_echo_off(new_lines)
        output_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    report_path = output.with_name(report_name)
    _write_mapping_report(report_path, report)
    if progress_total is not None:
        _report_mapping_progress(
            progress_callback,
            progress_total,
            progress_total,
            "映射完成",
        )
    if show_progress:
        for output_file in output_paths:
            print(f"  {output_file}")
        for include_file in include_paths:
            print(f"  {include_file}")
        print(f"  {report_path}")
        print("映射完成。")
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
        raise argparse.ArgumentTypeError("需要输入三个逗号分隔的值。")
    return tuple(cast(part) for part in parts)  # type: ignore[return-value]


def _expand_frequency_range(text: str) -> list[float]:
    """展开 start:end:step 写法表示的闭区间频率范围。"""
    try:
        start, end, step = (float(part.strip()) for part in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "频率范围应写成 start:end:step，例如 1:800:1。"
        ) from exc
    if step <= 0.0:
        raise argparse.ArgumentTypeError("频率范围的 step 必须为正数。")
    if end < start:
        raise argparse.ArgumentTypeError("频率范围的 end 必须大于或等于 start。")

    values: list[float] = []
    current = start
    tolerance = abs(step) * 1.0e-9
    while current <= end + tolerance:
        values.append(float(round(current, 12)))
        current += step
    return values


def resolve_requested_frequencies(args: argparse.Namespace) -> list[float] | None:
    """合并重复 --frequency 和闭区间 --frequency-range 参数。"""
    frequencies = list(getattr(args, "frequency", None) or [])
    for frequency_range in getattr(args, "frequency_range", None) or []:
        frequencies.extend(_expand_frequency_range(frequency_range))
    return frequencies or None


def parse_frequency_text(text: str) -> list[float] | None:
    """解析 GUI 中逗号分隔的频率值或频率范围。"""
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


def _parse_gui_triplet(text: str, *, cast: Any) -> tuple[Any, Any, Any]:
    """把 GUI 三元组解析错误转成普通 ValueError。"""
    try:
        return _parse_triplet(text, cast=cast)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def parse_gui_translate(text: str) -> np.ndarray | None:
    """解析可选 GUI 平移量；空值表示不平移。"""
    if not text.strip():
        return None
    return np.array(_parse_gui_triplet(text, cast=float), dtype=float)


def parse_gui_axis_order(text: str) -> tuple[int, int, int]:
    """解析 GUI 坐标轴顺序文本。"""
    axis_order = _parse_gui_triplet(text, cast=int)
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError("axis_order 必须是 0、1、2 的排列。")
    return axis_order  # type: ignore[return-value]


def parse_gui_axis_sign(text: str) -> tuple[float, float, float]:
    """解析 GUI 坐标轴方向符号文本。"""
    return _parse_gui_triplet(text, cast=float)  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 命令行参数列表；为 None 时由 argparse 使用系统参数。

    Returns:
        argparse 解析后的命名空间。
    """
    parser = argparse.ArgumentParser(
        description="将已提取的 CGNS 压力数据映射到 Abaqus INP 模型。",
    )
    parser.add_argument("--inp", required=True, help="输入 Abaqus INP 文件路径。")
    parser.add_argument(
        "--extracted",
        required=True,
        help="包含 CGNS 提取结果的目录。",
    )
    parser.add_argument("--target-set", required=True, help="目标 Nset 或 Elset 名称。")
    parser.add_argument(
        "--target-set-type",
        choices=("nset", "elset"),
        required=True,
        help="说明 --target-set 指向 Nset 还是 Elset。",
    )
    parser.add_argument(
        "--frequency",
        action="append",
        type=float,
        help="目标频率，单位 Hz；可重复指定多个频率。",
    )
    parser.add_argument(
        "--frequency-range",
        action="append",
        help="闭区间频率范围，格式为 start:end:step，单位 Hz，例如 1:800:1。",
    )
    parser.add_argument("--output", help="输出 INP 文件路径。")
    parser.add_argument("--step-name", help="接收载荷 include 的分析步名称。")
    parser.add_argument(
        "--frequency-tolerance-hz",
        type=float,
        default=1.0e-6,
        help="请求频率与提取频率之间允许的最大差值。",
    )
    parser.add_argument("--k", type=int, default=4, help="IDW 插值使用的最近源面数量。")
    parser.add_argument("--scale", type=float, default=1.0, help="坐标比例系数。")
    parser.add_argument(
        "--translate",
        type=lambda text: np.array(_parse_triplet(text, cast=float), dtype=float),
        help="坐标平移量，格式为 dx,dy,dz。",
    )
    parser.add_argument(
        "--axis-order",
        type=lambda text: _parse_triplet(text, cast=int),
        default=(0, 1, 2),
        help="坐标轴顺序，例如 0,1,2 或 2,0,1。",
    )
    parser.add_argument(
        "--axis-sign",
        type=lambda text: _parse_triplet(text, cast=float),
        default=(1.0, 1.0, 1.0),
        help="坐标轴方向符号，例如 1,-1,1。",
    )
    parser.add_argument(
        "--max-time-records",
        type=int,
        default=500000,
        help="时域载荷展开记录数的安全上限。",
    )
    parser.add_argument(
        "--frequency-batch-size",
        type=int,
        default=None,
        help="每批处理的频率数量；默认一次处理全部频率。",
    )
    parser.add_argument(
        "--frequency-group-mode",
        choices=("none", "groups", "bandwidth"),
        default="none",
        help="稳态频率输出分组模式；默认写入单个 INP。",
    )
    parser.add_argument(
        "--frequency-group-value",
        type=float,
        default=None,
        help="分组值：总组数或 Hz 带宽。",
    )
    parser.add_argument(
        "--relative-zero-tolerance",
        type=float,
        default=1.0e-12,
        help="近零过滤相对阈值，小于该阈值乘全局最大力的分量会被跳过。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行处理频率批次的线程数；默认 1 表示串行，0 表示自动。",
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
            frequency_batch_size=args.frequency_batch_size,
            frequency_group_mode=args.frequency_group_mode,
            frequency_group_value=args.frequency_group_value,
            relative_zero_tolerance=args.relative_zero_tolerance,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    for output_path in result.output_inp_paths:
        print(f"已写入：{output_path}")
    for include_path in result.include_paths:
        print(f"已写入：{include_path}")
    print(f"已写入：{result.report_path}")
    return 0


def run_gui() -> int:
    """启动基于文件选择器的 Tkinter 映射界面。"""
    from starccm_pressure.mapping_gui import run_gui as _run_gui

    return _run_gui()

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

