"""从 STAR-CCM+ HDF5-CGNS 时间步文件中提取全场脉动压力数据。

功能概述
--------
1. 读取 CGNS/HDF5 格式的 CFD 瞬态计算结果；
2. 提取所有表面节点的压力时间序列；
3. 计算复数压力谱（FFT）、广义等效力谱；
4. 支持流式处理（分块 FFT）以应对大规模网格；
5. 输出为压缩 JSON（时间/频谱/平均）及 NPZ/CSV 增强格式。

关键数据结构：
- PressureTimeSeries       —— 压力时间序列
- SurfaceGeometry          —— 表面几何（节点坐标、面元、法向量）
- ComplexPressureSpectrum  —— 复数压力谱（实部+虚部+幅值+相位）

关键函数索引：
- read_pressure_time_series          —— 读取全节点压力时间序列
- compute_pressure_complex_spectrum  —— 计算复数 FFT 谱
- compute_pulsating_pressure         —— 去均值得到脉动压力
- compute_equivalent_force_spectrum  —— 压力谱 → 广义等效力谱
- compute_streaming_equivalent_force_summary —— 流式等效力汇总
- read_surface_geometry              —— 读取表面三角形几何
- write_outputs / write_enhanced_outputs —— JSON/NPZ/CSV 输出

打包命令（pyinstaller）：
    pyinstaller --noupx --onefile --windowed --name extract_cgns_pressure extract_cgns_pressure.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import re
import threading
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypedDict

import numpy as np


# === 全局常量 ===
DEFAULT_PRESSURE_NAMES = ("Pressure", "pressure")  # 默认压力变量名候选
SCHEMA_VERSION = 1                                  # JSON 输出格式版本
SURFACE_GEOMETRY_CACHE_SCHEMA_VERSION = 2
COORDINATE_NAMES = ("CoordinateX", "CoordinateY", "CoordinateZ")
TRI_3_MIXED_ELEMENT_CODE = 5  # CGNS MIXED 元素中 TRI_3 的类型码


# === 自定义异常 ===


class OperationCancelled(Exception):
    """用户取消长时间运行的提取操作时抛出。"""


# === 数据结构 ===


@dataclass(frozen=True)
class PressureTimeSeries:
    """从多个 CGNS 文件中读取的压力时间序列。

    Attributes:
        node_ids: 节点编号（1-based 整数数组）。
        pressures: 压力值数组，形状 (时间步数, 节点数)。
        dataset_path: CGNS 内部压力数据集的路径。
        file_paths: 源文件路径列表（按时间排序）。
        coordinates: 节点坐标 (N, 3)，未找到坐标时为 None。
    """
    node_ids: np.ndarray
    pressures: np.ndarray
    dataset_path: str
    file_paths: list[Path]
    coordinates: np.ndarray | None = None


@dataclass(frozen=True)
class SurfaceGeometry:
    """表面三角形网格的几何属性。

    所有向量/面积均基于三角形面元计算（叉积法求面积和法向）。

    Attributes:
        coordinates: 节点坐标 (N_nodes, 3)。
        faces: 三角形面元顶点索引 (N_faces, 3)，0-based。
        centers: 三角形中心坐标 (N_faces, 3)。
        area_vectors: 面元面积矢量 (N_faces, 3)，方向为法向，大小为面积。
        areas: 面元标量面积 (N_faces,)。
        normals: 面元单位法向量 (N_faces, 3)。
        connectivity_path: CGNS ElementConnectivity 路径。
        element_range_path: CGNS ElementRange 路径。
    """
    coordinates: np.ndarray
    faces: np.ndarray
    centers: np.ndarray
    area_vectors: np.ndarray
    areas: np.ndarray
    normals: np.ndarray
    connectivity_path: str = ""
    element_range_path: str = ""


@dataclass(frozen=True)
class ComplexPressureSpectrum:
    """复数压力谱——FFT 的完整输出。

    包含实部/虚部（用于复数运算如等效力投影）和幅值/相位（用于可视化）。

    Attributes:
        frequencies_hz: 频率轴（Hz）。
        pressure_real: 压力 FFT 实部 (N_freq, N_nodes)。
        pressure_imag: 压力 FFT 虚部 (N_freq, N_nodes)。
        pressure_amplitude: 单边幅值谱（已乘缩放系数）。
        pressure_phase_rad: 相位谱（弧度）。
        amplitude_scale: 各频率的幅值缩放系数。
    """
    frequencies_hz: np.ndarray
    pressure_real: np.ndarray
    pressure_imag: np.ndarray
    pressure_amplitude: np.ndarray
    pressure_phase_rad: np.ndarray
    amplitude_scale: np.ndarray


# === 类型别名 ===


class ProgressEvent(TypedDict, total=False):
    """进度事件——用于 GUI 进度条更新。

    Attributes:
        stage: 当前阶段（"read"/"write"/"stream_force"/"pressure_blocks"）。
        current: 当前进度。
        total: 总进度。
        message: 显示文本。
    """
    stage: str
    current: int
    total: int
    message: str


ProgressCallback = Callable[[ProgressEvent], None]


class MetadataPayload(TypedDict):
    """JSON 输出的元数据部分。"""
    dataset_path: str
    time_step_s: float
    include_dc: bool
    file_count: int
    node_count: int
    node_id_kind: str
    coordinates_available: bool
    source_files: list[str]


class TimePayload(TypedDict):
    """时间序列 JSON 有效载荷。"""
    schema_version: int
    metadata: MetadataPayload
    time_s: list[float]
    node_ids: list[int]
    coordinates: dict[str, list[float]] | None
    pulsating_pressure: list[list[float]]


class SpectrumPayload(TypedDict):
    """频谱 JSON 有效载荷。"""
    schema_version: int
    metadata: MetadataPayload
    frequencies_hz: list[float]
    node_ids: list[int]
    coordinates: dict[str, list[float]] | None
    amplitudes: list[list[float]]
    phases_rad: list[list[float]]
    phases_deg: list[list[float]]
    real_parts: list[list[float]]
    imaginary_parts: list[list[float]]


class AveragePayload(TypedDict):
    """空间平均 JSON 有效载荷。"""
    schema_version: int
    metadata: MetadataPayload
    time_s: list[float]
    average_pulsating_pressure_time: list[float]
    average_frequencies_hz: list[float]
    average_amplitudes: list[float]
    average_phases_rad: list[float]
    average_phases_deg: list[float]
    average_real_parts: list[float]
    average_imaginary_parts: list[float]


# === HDF5 加载 ===


def _load_h5py() -> Any:
    """按需加载 h5py——避免在没有安装 h5py 的环境中导入时直接报错。

    Raises:
        RuntimeError: h5py 未安装。
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to read HDF5-CGNS files. Install it with: pip install h5py"
        ) from exc
    return h5py


# === 文件路径展开 ===


def _natural_sort_key(text: str) -> tuple[object, ...]:
    """Build a natural-sort key so embedded numbers sort by numeric value."""
    parts = re.split(r"(\d+)", text)
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in parts
    )


def sort_time_step_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Sort CGNS time-step paths by filename using natural numeric ordering."""
    return sorted(
        [Path(path) for path in paths],
        key=lambda path: _natural_sort_key(path.name),
    )


def build_input_file_order_message(file_paths: Iterable[str | Path]) -> str:
    """Build a concise GUI message showing the resolved time-step file order."""
    paths = sort_time_step_paths(file_paths)
    if not paths:
        return "未识别到 CGNS 文件。"
    if len(paths) == 1:
        return f"已识别 1 个 CGNS 文件：{paths[0].name}"
    return (
        f"已识别 {len(paths)} 个 CGNS 文件；"
        f"首个：{paths[0].name}；末个：{paths[-1].name}"
    )


def expand_input_files(patterns: Iterable[str | Path]) -> list[Path]:
    """展开 glob 模式并筛选出存在的 CGNS 文件。

    支持通配符模式（如 "data/604@*.cgns"）和直接路径。
    未匹配的模式保留原样（若不存在则后续读取时自然会报错）。

    Args:
        patterns: 文件路径或 glob 模式的列表。

    Returns:
        按文件名排序的现有文件路径列表。

    Raises:
        FileNotFoundError: 所有模式均未匹配到任何现有文件。
    """
    paths: list[Path] = []
    for pattern in patterns:
        pattern_text = str(pattern)
        matches = [Path(match) for match in glob.glob(pattern_text)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern_text))

    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        raise FileNotFoundError("No CGNS files matched the input pattern.")
    return sort_time_step_paths(existing_paths)


# === HDF5 数据集导航 ===


def _iter_datasets(node: Any, path: str = ""):
    """递归遍历 HDF5 组/数据集树。

    对所有数据集生成 (路径, 数据集对象) 元组。
    组对象（有 items 方法的）被递归展开。

    Args:
        node: HDF5 组或数据集对象。
        path: 当前累积路径。

    Yields:
        (dataset_path, dataset_object)。
    """
    if hasattr(node, "items"):
        for name, child in node.items():
            child_path = f"{path}/{name}"
            yield from _iter_datasets(child, child_path)
        return
    yield path, node


def _dataset_basename(path: str) -> str:
    """获取数据集路径的最后一段名称。"""
    return path.rsplit("/", 1)[-1]


def _dataset_parent_path(path: str) -> str:
    """获取数据集路径的父路径。"""
    return path.rsplit("/", 1)[0]


def _dataset_match_names(path: str) -> tuple[str, ...]:
    """生成用于匹配的数据集名称候选元组。

    当数据集名为 "data" 时，同时用父级组名匹配——
    这是 CGNS 常见的层级结构：/Base/Zone/Pressure/data。
    """
    basename = _dataset_basename(path)
    if basename.strip().lower() == "data" and "/" in path.strip("/"):
        parent_path = _dataset_parent_path(path)
        return (basename, _dataset_basename(parent_path), parent_path.strip("/"))
    return (basename, path.strip("/"))


def _dataset_matches(path: str, pressure_name: str) -> bool:
    """判断数据集路径是否匹配目标压力变量名。"""
    normalized_name = pressure_name.strip("/")
    return any(
        match_name == normalized_name for match_name in _dataset_match_names(path)
    )


def _find_pressure_dataset(root: Any, pressure_name: str | None) -> tuple[str, Any]:
    """在 HDF5 树中查找压力数据集。

    查找策略：
    1. 若指定了名称，精确匹配；
    2. 否则按 DEFAULT_PRESSURE_NAMES 候选列表匹配；
    3. 全部失败时抛出带可用数据集列表的异常。

    Args:
        root: HDF5 File 对象。
        pressure_name: 目标压力数据集名称（可为 None）。

    Returns:
        (dataset_path, dataset_object)。

    Raises:
        ValueError: 未找到匹配的压力数据集。
    """
    datasets = list(_iter_datasets(root))
    if pressure_name is not None:
        for path, dataset in datasets:
            if _dataset_matches(path, pressure_name):
                return path, dataset
        raise ValueError(f"Pressure dataset '{pressure_name}' was not found.")

    by_name = {}
    for path, dataset in datasets:
        for match_name in _dataset_match_names(path):
            by_name.setdefault(match_name, (path, dataset))
    for candidate in DEFAULT_PRESSURE_NAMES:
        if candidate in by_name:
            return by_name[candidate]

    available = ", ".join(_dataset_match_names(path)[0] for path, _ in datasets) or "<none>"
    raise ValueError(
        "Pressure dataset was not found. Checked names: "
        f"{', '.join(DEFAULT_PRESSURE_NAMES)}. Available datasets: {available}"
    )


def _dataset_at_path(root: Any, dataset_path: str) -> Any:
    """通过路径字符串直接访问 HDF5 数据集。

    先尝试一步索引（root[path]），失败则逐级遍历。

    Args:
        root: HDF5 File 对象。
        dataset_path: 数据集在 HDF5 中的路径。

    Returns:
        数据集对象。

    Raises:
        KeyError: 路径不存在。
    """
    try:
        return root[dataset_path]
    except Exception:
        node = root
        try:
            for part in dataset_path.strip("/").split("/"):
                node = node[part]
        except Exception as exc:
            raise KeyError(dataset_path) from exc
        return node


def _find_named_dataset(root: Any, dataset_name: str) -> tuple[str, Any] | None:
    """在 HDF5 树中按名称搜索数据集，返回第一个匹配项。

    Args:
        root: HDF5 File 对象。
        dataset_name: 目标名称。

    Returns:
        (path, dataset) 或 None。
    """
    for path, dataset in _iter_datasets(root):
        if _dataset_matches(path, dataset_name):
            return path, dataset
    return None


# === 坐标数据读取 ===


def _coordinate_dataset_info(path: str) -> tuple[str, str] | None:
    """判断数据集是否为坐标数据（CoordinateX/Y/Z）。

    返回 (coordinate_name, group_path) 或 None。
    同样处理数据名为 "data" 的嵌套情形——坐标组的名称
    就是 CoordinateX/Y/Z。
    """
    basename = _dataset_basename(path)
    if basename.strip().lower() == "data" and "/" in path.strip("/"):
        coordinate_path = _dataset_parent_path(path)
        coordinate_name = _dataset_basename(coordinate_path)
        group_path = _dataset_parent_path(coordinate_path)
    else:
        coordinate_name = basename
        group_path = _dataset_parent_path(path)

    if coordinate_name not in COORDINATE_NAMES:
        return None
    return coordinate_name, group_path


def _pressure_vector(dataset: Any, dataset_path: str) -> np.ndarray:
    """读取压力数据集为一维向量，含完整性检查。

    Args:
        dataset: HDF5 数据集。
        dataset_path: 数据集路径（用于错误消息）。

    Returns:
        一维压力值数组。

    Raises:
        ValueError: 形状非一维、为空、含非有限值。
    """
    values = np.asarray(dataset, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"Pressure dataset {dataset_path} must be one-dimensional after squeezing; "
            f"got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError(f"Pressure dataset {dataset_path} is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Pressure dataset {dataset_path} contains NaN or infinite values.")
    return values


def _pressure_vector_slice(
    dataset: Any,
    dataset_path: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, int]:
    """读取压力数据集的一个切片（用于流式分块处理）。

    智能处理多维度数据集：自动检测非单例维度作为压力通道轴，
    使用原生 HDF5 切片避免加载整个数据集。

    Args:
        dataset: HDF5 数据集。
        dataset_path: 数据集路径。
        start: 通道起始索引。
        end: 通道结束索引。

    Returns:
        (sliced_vector, total_channel_count)。

    Raises:
        ValueError: 切片越界或结果不符合预期。
    """
    shape = getattr(dataset, "shape", None)
    if shape is None:
        vector = _pressure_vector(dataset, dataset_path)
        return vector[start:end], vector.size

    shape_values = tuple(int(dim) for dim in shape)
    # 找到非单例维度（即实际数据存储维度）
    non_singleton_axes = [
        axis for axis, dimension in enumerate(shape_values) if dimension != 1
    ]
    if len(non_singleton_axes) != 1:
        vector = _pressure_vector(dataset, dataset_path)
        return vector[start:end], vector.size

    axis = non_singleton_axes[0]
    channel_count = shape_values[axis]
    if start < 0 or end < start or end > channel_count:
        raise ValueError(
            f"Pressure slice [{start}:{end}] is outside dataset {dataset_path} "
            f"with {channel_count} channels."
        )

    if len(shape_values) == 1:
        selection: Any = slice(start, end)
    else:
        selection_parts: list[Any] = [0] * len(shape_values)
        selection_parts[axis] = slice(start, end)
        selection = tuple(selection_parts)

    try:
        values = np.asarray(dataset[selection], dtype=float).squeeze()
    except TypeError:
        vector = _pressure_vector(dataset, dataset_path)
        return vector[start:end], vector.size
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError(
            f"Pressure dataset {dataset_path} slice must be one-dimensional after "
            f"squeezing; got shape {values.shape}."
        )
    if values.size != end - start:
        raise ValueError(
            f"Pressure dataset {dataset_path} slice returned {values.size} values; "
            f"expected {end - start}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Pressure dataset {dataset_path} slice contains NaN or infinite values."
        )
    return values, channel_count


def _coordinate_vector(dataset: Any, dataset_path: str) -> np.ndarray:
    """读取坐标数据集为一维向量。

    Args:
        dataset: HDF5 数据集。
        dataset_path: 数据集路径。

    Returns:
        一维坐标数组。
    """
    values = np.asarray(dataset, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"Coordinate dataset {dataset_path} must be one-dimensional after squeezing; "
            f"got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError(f"Coordinate dataset {dataset_path} is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Coordinate dataset {dataset_path} contains NaN or infinite values.")
    return values


def _read_coordinates(root: Any, expected_node_count: int) -> np.ndarray | None:
    """从 HDF5 树中读取完整的 XYZ 节点坐标。

    按组路径聚合 CoordinateX/Y/Z 三个数据集，
    要求三者在同一组下且长度一致。

    Args:
        root: HDF5 File 对象。
        expected_node_count: 预期的节点数（用于验证）。

    Returns:
        (N, 3) 坐标数组或 None。
    """
    coordinate_groups: dict[str, dict[str, tuple[str, Any]]] = {}
    for path, dataset in _iter_datasets(root):
        info = _coordinate_dataset_info(path)
        if info is None:
            continue
        coordinate_name, group_path = info
        coordinate_groups.setdefault(group_path, {})[coordinate_name] = (path, dataset)

    for group in coordinate_groups.values():
        if not all(name in group for name in COORDINATE_NAMES):
            continue
        coordinate_vectors = []
        for name in COORDINATE_NAMES:
            dataset_path, dataset = group[name]
            vector = _coordinate_vector(dataset, dataset_path)
            if vector.size != expected_node_count:
                break
            coordinate_vectors.append(vector)
        if len(coordinate_vectors) == len(COORDINATE_NAMES):
            return np.column_stack(coordinate_vectors)

    return None


# === 表面几何解析（TRI_3 混合元素） ===


def parse_mixed_tri3_faces(connectivity: np.ndarray, element_range: np.ndarray) -> np.ndarray:
    """从 CGNS MIXED 元素格式中解析 TRI_3 三角形面元。

    CGNS MIXED 连接性格式：每个元素以类型码开头，后跟顶点索引。
    TRI_3 类型码为 5，每个 TRI_3 占用 4 个整数 (5, v1, v2, v3)。

    将 1-based 的 CGNS 顶点索引转为 0-based。

    Args:
        connectivity: ElementConnectivity 一维数组。
        element_range: ElementRange [start, end]。

    Returns:
        (N_faces, 3) 的 0-based 顶点索引数组。

    Raises:
        ValueError: 非 TRI_3 元素、数据截断、面数不匹配等。
    """
    connectivity_values = np.asarray(connectivity, dtype=int).ravel()
    element_range_values = np.asarray(element_range, dtype=int).ravel()
    if element_range_values.size != 2:
        raise ValueError("ElementRange must contain exactly two values.")
    expected_face_count = int(element_range_values[1] - element_range_values[0] + 1)
    if expected_face_count <= 0:
        raise ValueError("ElementRange must describe at least one element.")

    faces: list[np.ndarray] = []
    index = 0
    while index < connectivity_values.size:
        element_code = int(connectivity_values[index])
        if element_code != TRI_3_MIXED_ELEMENT_CODE:
            raise ValueError(
                "Only CGNS Mixed TRI_3 elements are supported; "
                f"got element code {element_code} at connectivity index {index}."
            )
        if index + 3 >= connectivity_values.size:
            raise ValueError("ElementConnectivity ended inside a TRI_3 record.")
        face = connectivity_values[index + 1 : index + 4]
        if np.any(face <= 0):
            raise ValueError("CGNS face vertex indices must be positive 1-based values.")
        faces.append(face - 1)  # 转 0-based
        index += 4

    if len(faces) != expected_face_count:
        raise ValueError(
            f"Parsed {len(faces)} faces, but ElementRange describes {expected_face_count}."
        )
    return np.vstack(faces).astype(int, copy=False)


def compute_triangle_surface_geometry(
    coordinates: np.ndarray,
    faces: np.ndarray,
) -> SurfaceGeometry:
    """计算三角形面元的几何属性。

    用叉积法计算：
    - area_vectors = 0.5 * (r2 - r1) × (r3 - r1)  —— 面积矢量
    - areas = ||area_vectors||                        —— 标量面积
    - normals = area_vectors / areas                  —— 单位法向量
    - centers = (r1 + r2 + r3) / 3                    —— 面元中心

    Args:
        coordinates: 节点坐标 (N_nodes, 3)。
        faces: 面元顶点索引 (N_faces, 3)，0-based。

    Returns:
        SurfaceGeometry 实例。
    """
    coordinate_array = np.asarray(coordinates, dtype=float)
    face_array = np.asarray(faces, dtype=int)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
        raise ValueError("coordinates must be a two-dimensional array with x/y/z columns.")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("faces must be a two-dimensional array with three vertex indices.")
    if face_array.size == 0:
        raise ValueError("faces must contain at least one triangle.")
    if np.any(face_array < 0) or np.any(face_array >= coordinate_array.shape[0]):
        raise ValueError("faces contain vertex indices outside the coordinate array.")
    if not np.all(np.isfinite(coordinate_array)):
        raise ValueError("coordinates contains NaN or infinite values.")

    r1 = coordinate_array[face_array[:, 0]]
    r2 = coordinate_array[face_array[:, 1]]
    r3 = coordinate_array[face_array[:, 2]]
    # 叉积方向遵循右手定则：从 r1→r2→r3 逆时针
    area_vectors = 0.5 * np.cross(r2 - r1, r3 - r1)
    areas = np.linalg.norm(area_vectors, axis=1)
    if not np.all(np.isfinite(area_vectors)) or not np.all(np.isfinite(areas)):
        raise ValueError("triangle area vectors contain NaN or infinite values.")
    if np.any(areas <= 0.0):
        raise ValueError("all triangle areas must be positive.")
    normals = area_vectors / areas[:, np.newaxis]
    if not np.all(np.isfinite(normals)):
        raise ValueError("triangle normals contain NaN or infinite values.")
    centers = (r1 + r2 + r3) / 3.0
    return SurfaceGeometry(
        coordinates=coordinate_array,
        faces=face_array,
        centers=centers,
        area_vectors=area_vectors,
        areas=areas,
        normals=normals,
    )


def _surface_path_from_connectivity_path(connectivity_path: str) -> str:
    """从 ElementConnectivity 路径提取表面组路径。

    路径结构约定：/Base/Zone/Face_to_Vertex_Map/ElementConnectivity
    返回 /Base/Zone/Face_to_Vertex_Map。
    """
    parts = connectivity_path.strip("/").split("/")
    try:
        map_index = parts.index("Face to Vertex Map")
    except ValueError as exc:
        raise ValueError(
            f"ElementConnectivity path is not under a Face to Vertex Map: {connectivity_path}"
        ) from exc
    return "/" + "/".join(parts[:map_index])


def _read_surface_coordinates(root: Any, surface_path: str) -> np.ndarray:
    """从指定表面路径下读取完整的 XYZ 节点坐标。

    仅搜索 surface_path 下的 CoordinateX/Y/Z 组。

    Args:
        root: HDF5 File 对象。
        surface_path: 表面组路径。

    Returns:
        (N, 3) 坐标数组。

    Raises:
        ValueError: 坐标不完整。
    """
    coordinate_groups: dict[str, dict[str, tuple[str, Any]]] = {}
    for path, dataset in _iter_datasets(root):
        info = _coordinate_dataset_info(path)
        if info is None:
            continue
        coordinate_name, group_path = info
        if not group_path.startswith(f"{surface_path}/"):
            continue
        coordinate_groups.setdefault(group_path, {})[coordinate_name] = (path, dataset)

    for group in coordinate_groups.values():
        if not all(name in group for name in COORDINATE_NAMES):
            continue
        coordinate_vectors = [
            _coordinate_vector(group[name][1], group[name][0])
            for name in COORDINATE_NAMES
        ]
        node_count = coordinate_vectors[0].size
        if all(vector.size == node_count for vector in coordinate_vectors):
            return np.column_stack(coordinate_vectors)
    raise ValueError(f"Complete x/y/z coordinates were not found under {surface_path}.")


def read_surface_geometry(
    file_path: str | Path,
    expected_face_count: int | None = None,
    h5_module: Any | None = None,
) -> SurfaceGeometry:
    """从 CGNS 文件中读取表面三角形几何。

    流程：
    1. 查找 ElementConnectivity 和 ElementRange；
    2. 解析 TRI_3 面元；
    3. 读取面元节点坐标；
    4. 计算几何属性（面积、法向量等）。

    Args:
        file_path: CGNS 文件路径。
        expected_face_count: 预期的面元数（用于验证与压力通道数一致）。
        h5_module: h5py 模块（避免重复导入）。

    Returns:
        SurfaceGeometry 实例。
    """
    h5 = h5_module if h5_module is not None else _load_h5py()
    with h5.File(file_path, "r") as root:
        connectivity_match = _find_named_dataset(root, "ElementConnectivity")
        element_range_match = _find_named_dataset(root, "ElementRange")
        if connectivity_match is None:
            raise ValueError("ElementConnectivity dataset was not found.")
        if element_range_match is None:
            raise ValueError("ElementRange dataset was not found.")
        connectivity_path, connectivity_dataset = connectivity_match
        element_range_path, element_range_dataset = element_range_match
        surface_path = _surface_path_from_connectivity_path(connectivity_path)
        faces = parse_mixed_tri3_faces(
            np.asarray(connectivity_dataset, dtype=int),
            np.asarray(element_range_dataset, dtype=int),
        )
        if expected_face_count is not None and faces.shape[0] != int(expected_face_count):
            raise ValueError(
                f"Parsed {faces.shape[0]} faces, but pressure has {expected_face_count} channels."
            )
        coordinates = _read_surface_coordinates(root, surface_path)

    geometry = compute_triangle_surface_geometry(coordinates, faces)
    return SurfaceGeometry(
        coordinates=geometry.coordinates,
        faces=geometry.faces,
        centers=geometry.centers,
        area_vectors=geometry.area_vectors,
        areas=geometry.areas,
        normals=geometry.normals,
        connectivity_path=connectivity_path,
        element_range_path=element_range_path,
    )


# === 进度与取消 ===


def _check_cancelled(cancel_event: Any | None) -> None:
    """检查取消事件，若已设置则抛出 OperationCancelled。

    Args:
        cancel_event: threading.Event 或 None。
    """
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled("Operation cancelled.")


def _report_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    """向进度回调报告当前进度。

    Args:
        progress_callback: 回调函数或 None。
        stage: 阶段名。
        current: 当前进度。
        total: 总进度。
        message: 进度消息。
    """
    if progress_callback is not None:
        progress_callback(
            {
                "stage": stage,
                "current": int(current),
                "total": int(total),
                "message": message,
            }
        )


# === 压力时间序列读取 ===


def read_pressure_time_series(
    file_paths: Iterable[str | Path],
    pressure_name: str | None = None,
    h5_module: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> PressureTimeSeries:
    """从多个 CGNS 文件中读取全节点压力时间序列。

    读取策略：
    - 从第一个文件确定压力数据集路径，后续文件优先按相同路径读取；
    - 若路径不存在则重新搜索（适应文件间结构变化）；
    - 仅在第一个文件读取坐标（假设坐标在时间序列中不变）；
    - 若不同文件的节点数不一致则报错。

    Args:
        file_paths: CGNS 文件路径列表（按时间排序）。
        pressure_name: 压力数据集名称。
        h5_module: h5py 模块。
        progress_callback: 进度回调。
        cancel_event: 取消事件。

    Returns:
        PressureTimeSeries 实例。
    """
    paths = sort_time_step_paths(file_paths)
    if not paths:
        raise ValueError("At least one CGNS file is required.")

    h5 = h5_module if h5_module is not None else _load_h5py()
    pressure_rows: list[np.ndarray] = []
    selected_dataset_path: str | None = None
    coordinates: np.ndarray | None = None
    coordinates_checked = False

    for index, path in enumerate(paths, start=1):
        _check_cancelled(cancel_event)
        with h5.File(path, "r") as root:
            if selected_dataset_path is None:
                dataset_path, dataset = _find_pressure_dataset(root, pressure_name)
            else:
                try:
                    dataset_path = selected_dataset_path
                    dataset = _dataset_at_path(root, selected_dataset_path)
                except KeyError:
                    # 文件结构变化 → 重新搜索
                    dataset_path, dataset = _find_pressure_dataset(root, pressure_name)
            vector = _pressure_vector(dataset, dataset_path)
            if not coordinates_checked:
                coordinates = _read_coordinates(root, vector.size)
                coordinates_checked = True
        if selected_dataset_path is None:
            selected_dataset_path = dataset_path
        elif dataset_path != selected_dataset_path:
            raise ValueError(
                "Pressure dataset path changed between files: "
                f"{selected_dataset_path} vs {dataset_path}"
            )
        if pressure_rows and vector.size != pressure_rows[0].size:
            raise ValueError(
                f"Pressure node count changed in {path}: "
                f"{vector.size} vs {pressure_rows[0].size}"
            )
        pressure_rows.append(vector)
        _report_progress(
            progress_callback,
            "read",
            index,
            len(paths),
            f"已读取 {path.name}",
        )
        _check_cancelled(cancel_event)

    pressures = np.vstack(pressure_rows)
    node_ids = np.arange(1, pressures.shape[1] + 1, dtype=int)
    return PressureTimeSeries(
        node_ids=node_ids,
        pressures=pressures,
        dataset_path=selected_dataset_path or "",
        file_paths=paths,
        coordinates=coordinates,
    )


# === 逐文件压力向量读取（流式处理用） ===


def _read_pressure_vector_from_file(
    h5: Any,
    file_path: str | Path,
    pressure_name: str | None,
    selected_dataset_path: str | None = None,
) -> tuple[str, np.ndarray]:
    """从单个文件读取完整压力向量。

    Returns:
        (dataset_path, vector)。
    """
    with h5.File(file_path, "r") as root:
        if selected_dataset_path is None:
            dataset_path, dataset = _find_pressure_dataset(root, pressure_name)
        else:
            try:
                dataset_path = selected_dataset_path
                dataset = _dataset_at_path(root, selected_dataset_path)
            except KeyError:
                dataset_path, dataset = _find_pressure_dataset(root, pressure_name)
        vector = _pressure_vector(dataset, dataset_path)
    return dataset_path, vector


def _read_pressure_vector_slice_from_file(
    h5: Any,
    file_path: str | Path,
    pressure_name: str | None,
    start: int,
    end: int,
    selected_dataset_path: str | None = None,
) -> tuple[str, np.ndarray, int]:
    """从单个文件读取压力向量切片。

    Returns:
        (dataset_path, sliced_vector, channel_count)。
    """
    with h5.File(file_path, "r") as root:
        if selected_dataset_path is None:
            dataset_path, dataset = _find_pressure_dataset(root, pressure_name)
        else:
            try:
                dataset_path = selected_dataset_path
                dataset = _dataset_at_path(root, selected_dataset_path)
            except KeyError:
                dataset_path, dataset = _find_pressure_dataset(root, pressure_name)
        vector, channel_count = _pressure_vector_slice(dataset, dataset_path, start, end)
    return dataset_path, vector, channel_count


# === FFT 频谱计算 ===


def _single_sided_amplitude_scale(sample_count: int) -> np.ndarray:
    """计算单边 FFT 幅值的缩放系数。

    对于实数 FFT（rfft）：
    - 非 DC/Nyquist 频率：2/N（补偿仅用正频率的一半能量）
    - DC 和 Nyquist（N 为偶数时）：1/N（非对称）

    Args:
        sample_count: 时域采样点数 N。

    Returns:
        形状为 (N//2+1,) 的缩放系数数组。
    """
    frequency_count = sample_count // 2 + 1
    scale = np.full(frequency_count, 2.0 / sample_count, dtype=float)
    scale[0] = 1.0 / sample_count
    if sample_count % 2 == 0:
        scale[-1] = 1.0 / sample_count
    return scale


def compute_pressure_complex_spectrum(
    pressures: np.ndarray,
    dt: float,
    include_dc: bool = False,
    remove_mean: bool = True,
) -> ComplexPressureSpectrum:
    """计算全节点复数压力谱。

    使用 numpy.fft.rfft（实数输入 → 仅正频率），
    保留实部/虚部以便后续复数运算（如等效力投影）。

    Args:
        pressures: 压力数组 (N_time, N_nodes)。
        dt: 采样时间间隔（秒）。
        include_dc: 是否保留 0 Hz 分量。
        remove_mean: 是否先减去时间均值（得到脉动压力）。

    Returns:
        ComplexPressureSpectrum 实例。
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    pressure_array = np.asarray(pressures, dtype=float)
    if pressure_array.ndim != 2:
        raise ValueError("pressures must be a two-dimensional array: time steps x nodes.")
    if pressure_array.shape[0] < 2:
        raise ValueError("At least two time steps are required for spectrum calculation.")
    if not np.all(np.isfinite(pressure_array)):
        raise ValueError("pressures contains NaN or infinite values.")

    if remove_mean:
        spectrum_input = pressure_array - np.mean(pressure_array, axis=0, keepdims=True)
    else:
        spectrum_input = pressure_array
    sample_count = spectrum_input.shape[0]
    fft_values = np.fft.rfft(spectrum_input, axis=0)
    frequencies = np.fft.rfftfreq(sample_count, d=dt)
    amplitude_scale = _single_sided_amplitude_scale(sample_count)
    amplitudes = np.abs(fft_values) * amplitude_scale[:, np.newaxis]
    phases_rad = np.angle(fft_values)

    if not include_dc:
        frequencies = frequencies[1:]
        fft_values = fft_values[1:, :]
        amplitudes = amplitudes[1:, :]
        phases_rad = phases_rad[1:, :]
        amplitude_scale = amplitude_scale[1:]

    return ComplexPressureSpectrum(
        frequencies_hz=frequencies,
        pressure_real=fft_values.real,
        pressure_imag=fft_values.imag,
        pressure_amplitude=amplitudes,
        pressure_phase_rad=phases_rad,
        amplitude_scale=amplitude_scale,
    )


def _compute_pressure_spectrum_components(
    pressures: np.ndarray,
    dt: float,
    include_dc: bool = False,
    remove_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算压力谱的六个分量，用于 JSON 有效载荷。

    Returns:
        (frequencies, amplitudes, phases_rad, phases_deg, real_parts, imaginary_parts)。
    """
    spectrum = compute_pressure_complex_spectrum(
        pressures,
        dt=dt,
        include_dc=include_dc,
        remove_mean=remove_mean,
    )
    phases_rad = spectrum.pressure_phase_rad
    phases_deg = np.degrees(phases_rad)
    return (
        spectrum.frequencies_hz,
        spectrum.pressure_amplitude,
        phases_rad,
        phases_deg,
        spectrum.pressure_real,
        spectrum.pressure_imag,
    )


def compute_pressure_spectrum(
    pressures: np.ndarray,
    dt: float,
    include_dc: bool = False,
    remove_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """计算压力幅值谱（简化接口——仅返回频率和幅值）。

    向后兼容的简化版本，不返回相位和复数值。
    """
    frequencies, amplitudes, _, _, _, _ = _compute_pressure_spectrum_components(
        pressures,
        dt=dt,
        include_dc=include_dc,
        remove_mean=remove_mean,
    )
    return frequencies, amplitudes


# === 等效力谱计算 ===


def compute_equivalent_force_spectrum(
    spectrum: ComplexPressureSpectrum,
    area_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    """从复数压力谱和面积矢量计算广义等效力谱。

    等效力 = 压力 × 面积矢量（在线性复数域中相乘后累加）：
    F_complex(f) = Σ p_i(f) · A_i

    对每个频率分量：
    - 复数压力投影到面积矢量方向 → 三分量力（x/y/z）；
    - 计算各分量幅值/相位及合力的模；
    - 额外统计：全场最大压力幅值、RMS 压力幅值、相干压力求和。

    Args:
        spectrum: 复数压力谱。
        area_vectors: 面元面积矢量 (N_faces, 3)。

    Returns:
        含频率轴和各种力谱分量的字典。
    """
    area_vector_array = np.asarray(area_vectors, dtype=float)
    if area_vector_array.ndim != 2 or area_vector_array.shape[1] != 3:
        raise ValueError("area_vectors must be a two-dimensional array with x/y/z columns.")

    pressure_complex = spectrum.pressure_real + 1j * spectrum.pressure_imag
    if pressure_complex.ndim != 2:
        raise ValueError("pressure spectrum arrays must be two-dimensional.")
    if pressure_complex.shape[1] != area_vector_array.shape[0]:
        raise ValueError(
            f"pressure spectrum has {pressure_complex.shape[1]} faces, "
            f"but area_vectors has {area_vector_array.shape[0]} rows."
        )
    if spectrum.amplitude_scale.shape[0] != pressure_complex.shape[0]:
        raise ValueError("amplitude_scale length must match the frequency count.")

    # 复数力 = 压力(复数) × 面积矢量（矩阵乘）
    force_complex = pressure_complex @ area_vector_array
    force_amplitude = np.abs(force_complex) * spectrum.amplitude_scale[:, np.newaxis]
    force_phase = np.angle(force_complex)
    force_magnitude = np.linalg.norm(force_amplitude, axis=1)
    # 相干压力求和：所有面元压力的直接复数求和（未加权面积）
    coherent_pressure = np.abs(np.sum(pressure_complex, axis=1)) * spectrum.amplitude_scale
    return {
        "frequencies_hz": spectrum.frequencies_hz,
        "force_real_x": force_complex[:, 0].real,
        "force_real_y": force_complex[:, 1].real,
        "force_real_z": force_complex[:, 2].real,
        "force_imag_x": force_complex[:, 0].imag,
        "force_imag_y": force_complex[:, 1].imag,
        "force_imag_z": force_complex[:, 2].imag,
        "force_amplitude_x": force_amplitude[:, 0],
        "force_amplitude_y": force_amplitude[:, 1],
        "force_amplitude_z": force_amplitude[:, 2],
        "force_phase_x": force_phase[:, 0],
        "force_phase_y": force_phase[:, 1],
        "force_phase_z": force_phase[:, 2],
        "force_magnitude": force_magnitude,
        "max_pressure_amplitude": np.max(spectrum.pressure_amplitude, axis=1),
        "rms_pressure_amplitude": np.sqrt(
            np.mean(spectrum.pressure_amplitude**2, axis=1)
        ),
        "coherent_pressure_sum": coherent_pressure,
    }


def _force_spectrum_from_time_series(
    force_time: np.ndarray,
    coherent_pressure_time: np.ndarray,
    dt: float,
    include_dc: bool,
) -> dict[str, np.ndarray]:
    """从力时间序列计算力谱（流式汇总模式用）。

    力时间序列 = 每个时间步的 Σ(p_i · A_i)，
    对力和相干压力分别做 FFT 得到频谱。
    """
    force_spectrum = compute_pressure_complex_spectrum(
        compute_pulsating_pressure(force_time),
        dt=dt,
        include_dc=include_dc,
        remove_mean=False,
    )
    coherent_spectrum = compute_pressure_complex_spectrum(
        compute_pulsating_pressure(coherent_pressure_time[:, np.newaxis]),
        dt=dt,
        include_dc=include_dc,
        remove_mean=False,
    )
    force_complex = force_spectrum.pressure_real + 1j * force_spectrum.pressure_imag
    return {
        "frequencies_hz": force_spectrum.frequencies_hz,
        "force_real_x": force_complex[:, 0].real,
        "force_real_y": force_complex[:, 1].real,
        "force_real_z": force_complex[:, 2].real,
        "force_imag_x": force_complex[:, 0].imag,
        "force_imag_y": force_complex[:, 1].imag,
        "force_imag_z": force_complex[:, 2].imag,
        "force_amplitude_x": force_spectrum.pressure_amplitude[:, 0],
        "force_amplitude_y": force_spectrum.pressure_amplitude[:, 1],
        "force_amplitude_z": force_spectrum.pressure_amplitude[:, 2],
        "force_phase_x": force_spectrum.pressure_phase_rad[:, 0],
        "force_phase_y": force_spectrum.pressure_phase_rad[:, 1],
        "force_phase_z": force_spectrum.pressure_phase_rad[:, 2],
        "force_magnitude": np.linalg.norm(force_spectrum.pressure_amplitude, axis=1),
        "coherent_pressure_sum": coherent_spectrum.pressure_amplitude[:, 0],
    }


# === 流式等效力汇总 ===


def compute_streaming_equivalent_force_summary(
    file_paths: Iterable[str | Path],
    geometry: SurfaceGeometry,
    dt: float,
    include_dc: bool = False,
    pressure_name: str | None = None,
    pressure_block_size: int = 16384,
    h5_module: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> dict[str, np.ndarray]:
    """流式处理模式：分块计算压力 FFT，避免一次性加载全部节点。

    适用于大规模网格（数百万面元）场景。分两个阶段：

    阶段 1 —— 力时间序列：
    逐时间步读取全部面元压力，与面积矢量相乘累加为
    (合力_x, 合力_y, 合力_z, 相干压力和)，得到 (N_time, 4) 矩阵。
    对合力做 FFT 得到力谱。

    阶段 2 —— 压力分块统计：
    将面元分为若干块，每块分别读取所有时间步的压力 →
    FFT → 记录最大幅值和平方和。最后合并为全场
    max_pressure_amplitude 和 rms_pressure_amplitude。

    Args:
        file_paths: CGNS 文件列表。
        geometry: 表面几何。
        dt: 时间步长。
        include_dc: 是否保留 DC。
        pressure_name: 压力变量名。
        pressure_block_size: 每块面元数（控制内存使用）。
        h5_module: h5py 模块。
        progress_callback: 进度回调。
        cancel_event: 取消事件。

    Returns:
        含力谱和各统计量的字典。
    """
    paths = sort_time_step_paths(file_paths)
    if not paths:
        raise ValueError("At least one CGNS file is required.")
    if pressure_block_size <= 0:
        raise ValueError("pressure_block_size must be positive.")

    area_vectors = np.asarray(geometry.area_vectors, dtype=float)
    if area_vectors.ndim != 2 or area_vectors.shape[1] != 3:
        raise ValueError("geometry.area_vectors must have x/y/z columns.")
    face_count = int(area_vectors.shape[0])
    if face_count == 0:
        raise ValueError("geometry must contain at least one face.")

    h5 = h5_module if h5_module is not None else _load_h5py()
    force_time = np.empty((len(paths), 3), dtype=float)
    coherent_pressure_time = np.empty(len(paths), dtype=float)
    selected_dataset_path: str | None = None

    # 阶段 1：逐时间步积分等效力
    for index, path in enumerate(paths):
        _check_cancelled(cancel_event)
        dataset_path, vector = _read_pressure_vector_from_file(
            h5, path, pressure_name, selected_dataset_path,
        )
        if selected_dataset_path is None:
            selected_dataset_path = dataset_path
        elif dataset_path != selected_dataset_path:
            raise ValueError(
                "Pressure dataset path changed between files: "
                f"{selected_dataset_path} vs {dataset_path}"
            )
        if vector.size != face_count:
            raise ValueError(
                f"Pressure channel count changed in {path}: {vector.size} vs {face_count}."
            )
        force_time[index, :] = vector @ area_vectors  # 矩阵向量乘
        coherent_pressure_time[index] = float(np.sum(vector))
        _report_progress(
            progress_callback,
            "stream_force",
            index + 1,
            len(paths),
            f"已积分 {path.name}",
        )

    force_result = _force_spectrum_from_time_series(
        force_time, coherent_pressure_time,
        dt=dt, include_dc=include_dc,
    )
    frequencies = force_result["frequencies_hz"]

    # 阶段 2：分块 FFT 统计全场压力谱
    max_pressure_amplitude = np.full(frequencies.shape, -np.inf, dtype=float)
    pressure_amplitude_square_sum = np.zeros(frequencies.shape, dtype=float)

    block_total = (face_count + pressure_block_size - 1) // pressure_block_size
    for block_index, start in enumerate(range(0, face_count, pressure_block_size), start=1):
        _check_cancelled(cancel_event)
        end = min(start + pressure_block_size, face_count)
        pressure_block = np.empty((len(paths), end - start), dtype=float)
        selected_dataset_path = None
        for time_index, path in enumerate(paths):
            dataset_path, vector, channel_count = _read_pressure_vector_slice_from_file(
                h5, path, pressure_name, start, end, selected_dataset_path,
            )
            if selected_dataset_path is None:
                selected_dataset_path = dataset_path
            elif dataset_path != selected_dataset_path:
                raise ValueError(
                    "Pressure dataset path changed between files: "
                    f"{selected_dataset_path} vs {dataset_path}"
                )
            if channel_count != face_count:
                raise ValueError(
                    f"Pressure channel count changed in {path}: "
                    f"{channel_count} vs {face_count}."
                )
            pressure_block[time_index, :] = vector

        block_spectrum = compute_pressure_complex_spectrum(
            compute_pulsating_pressure(pressure_block),
            dt=dt,
            include_dc=include_dc,
            remove_mean=False,
        )
        if not np.allclose(block_spectrum.frequencies_hz, frequencies):
            raise RuntimeError("Pressure block frequency grid changed unexpectedly.")
        # 滚动更新最大值和平方和
        max_pressure_amplitude = np.maximum(
            max_pressure_amplitude,
            np.max(block_spectrum.pressure_amplitude, axis=1),
        )
        pressure_amplitude_square_sum += np.sum(
            block_spectrum.pressure_amplitude**2,
            axis=1,
        )
        _report_progress(
            progress_callback,
            "pressure_blocks",
            block_index,
            block_total,
            f"已计算压力频谱分块 {block_index}/{block_total}",
        )

    force_result["max_pressure_amplitude"] = max_pressure_amplitude
    force_result["rms_pressure_amplitude"] = np.sqrt(
        pressure_amplitude_square_sum / face_count
    )
    return force_result


# === 脉动压力（去均值） ===


def compute_pulsating_pressure(pressures: np.ndarray) -> np.ndarray:
    """从绝对压力计算脉动压力：p' = p - mean(p)。

    沿时间轴 (axis=0) 减去每个通道的时间均值。

    Args:
        pressures: 压力数组 (N_time, N_channels)。

    Returns:
        脉动压力数组（相同形状）。
    """
    pressure_array = np.asarray(pressures, dtype=float)
    if pressure_array.ndim != 2:
        raise ValueError("pressures must be a two-dimensional array: time steps x nodes.")
    if pressure_array.shape[0] == 0 or pressure_array.shape[1] == 0:
        raise ValueError("pressures must contain at least one time step and one node.")
    if not np.all(np.isfinite(pressure_array)):
        raise ValueError("pressures contains NaN or infinite values.")
    return pressure_array - np.mean(pressure_array, axis=0, keepdims=True)


# === JSON 有效载荷构建 ===


def _base_metadata(
    series: PressureTimeSeries,
    dt: float,
    include_dc: bool,
) -> MetadataPayload:
    """构建元数据字典的基础部分。"""
    return {
        "dataset_path": series.dataset_path,
        "time_step_s": float(dt),
        "include_dc": bool(include_dc),
        "file_count": len(series.file_paths),
        "node_count": int(series.node_ids.size),
        "node_id_kind": "array_index_1_based",
        "coordinates_available": series.coordinates is not None,
        "source_files": [str(series.file_paths[0])] if series.file_paths else [],
    }


def _payload_common_parts(
    series: PressureTimeSeries,
    dt: float,
    include_dc: bool,
) -> tuple[np.ndarray, np.ndarray, MetadataPayload, list[int], dict[str, list[float]] | None]:
    """提取 JSON 有效载荷的公共组件：脉动压力、时间轴、元数据、节点ID、坐标。"""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    pulsating_pressure = compute_pulsating_pressure(series.pressures)
    times = np.arange(pulsating_pressure.shape[0], dtype=float) * float(dt)
    metadata = _base_metadata(series, dt, include_dc)
    node_ids = _as_int_list(series.node_ids)
    coordinates = _coordinates_payload(series)
    return pulsating_pressure, times, metadata, node_ids, coordinates


def _as_float_list(values: np.ndarray) -> list[float]:
    """numpy 数组 → float 列表（JSON 兼容）。"""
    return [float(value) for value in np.asarray(values, dtype=float).tolist()]


def _as_float_matrix(values: np.ndarray) -> list[list[float]]:
    """numpy 二维数组 → 嵌套 float 列表（JSON 兼容）。"""
    array = np.asarray(values, dtype=float)
    return [[float(value) for value in row] for row in array.tolist()]


def _as_int_list(values: np.ndarray) -> list[int]:
    """numpy 数组 → int 列表（JSON 兼容）。"""
    return [int(value) for value in np.asarray(values, dtype=int).tolist()]


def _coordinates_payload(series: PressureTimeSeries) -> dict[str, list[float]] | None:
    """格式化坐标为 JSON 兼容的字典。"""
    if series.coordinates is None:
        return None
    coordinates = np.asarray(series.coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must be a two-dimensional array with x/y/z columns.")
    if coordinates.shape[0] != series.node_ids.size:
        raise ValueError(
            f"coordinates has {coordinates.shape[0]} rows, "
            f"but node_ids has {series.node_ids.size} values."
        )
    return {
        "x": _as_float_list(coordinates[:, 0]),
        "y": _as_float_list(coordinates[:, 1]),
        "z": _as_float_list(coordinates[:, 2]),
    }


def build_time_payload(
    metadata: MetadataPayload,
    times: np.ndarray,
    node_ids: list[int],
    coordinates: dict[str, list[float]] | None,
    pulsating_pressure: np.ndarray,
) -> TimePayload:
    """构建时间序列 JSON 有效载荷。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "time_s": _as_float_list(times),
        "node_ids": node_ids,
        "coordinates": coordinates,
        "pulsating_pressure": _as_float_matrix(pulsating_pressure),
    }


def build_spectrum_payload(
    metadata: MetadataPayload,
    node_ids: list[int],
    coordinates: dict[str, list[float]] | None,
    pulsating_pressure: np.ndarray,
    dt: float,
    include_dc: bool,
) -> SpectrumPayload:
    """构建频谱 JSON 有效载荷（含幅值、相位、实虚部）。"""
    (
        frequencies,
        amplitudes,
        phases_rad,
        phases_deg,
        real_parts,
        imaginary_parts,
    ) = _compute_pressure_spectrum_components(
        pulsating_pressure,
        dt=dt,
        include_dc=include_dc,
        remove_mean=False,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "frequencies_hz": _as_float_list(frequencies),
        "node_ids": node_ids,
        "coordinates": coordinates,
        "amplitudes": _as_float_matrix(amplitudes),
        "phases_rad": _as_float_matrix(phases_rad),
        "phases_deg": _as_float_matrix(phases_deg),
        "real_parts": _as_float_matrix(real_parts),
        "imaginary_parts": _as_float_matrix(imaginary_parts),
    }


def build_average_payload(
    metadata: MetadataPayload,
    times: np.ndarray,
    pulsating_pressure: np.ndarray,
    dt: float,
    include_dc: bool,
) -> AveragePayload:
    """构建空间平均 JSON 有效载荷。

    对所有节点取空间平均脉动压力 → 对平均时间序列做 FFT。
    """
    average_time = np.mean(pulsating_pressure, axis=1)
    (
        average_frequencies,
        average_amplitudes,
        average_phases_rad,
        average_phases_deg,
        average_real_parts,
        average_imaginary_parts,
    ) = _compute_pressure_spectrum_components(
        average_time[:, np.newaxis],
        dt=dt,
        include_dc=include_dc,
        remove_mean=False,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "time_s": _as_float_list(times),
        "average_pulsating_pressure_time": _as_float_list(average_time),
        "average_frequencies_hz": _as_float_list(average_frequencies),
        "average_amplitudes": _as_float_list(average_amplitudes[:, 0]),
        "average_phases_rad": _as_float_list(average_phases_rad[:, 0]),
        "average_phases_deg": _as_float_list(average_phases_deg[:, 0]),
        "average_real_parts": _as_float_list(average_real_parts[:, 0]),
        "average_imaginary_parts": _as_float_list(average_imaginary_parts[:, 0]),
    }


def build_json_payloads(
    series: PressureTimeSeries,
    dt: float,
    include_dc: bool = False,
) -> tuple[TimePayload, SpectrumPayload, AveragePayload]:
    """一次构建全部三个 JSON 有效载荷。

    Args:
        series: 压力时间序列。
        dt: 时间步长。
        include_dc: 是否保留 DC。

    Returns:
        (time_payload, spectrum_payload, average_payload)。
    """
    pulsating_pressure, times, metadata, node_ids, coordinates = _payload_common_parts(
        series, dt, include_dc,
    )
    return (
        build_time_payload(metadata, times, node_ids, coordinates, pulsating_pressure),
        build_spectrum_payload(
            metadata, node_ids, coordinates, pulsating_pressure, dt, include_dc,
        ),
        build_average_payload(metadata, times, pulsating_pressure, dt, include_dc),
    )


# === JSON/NPZ/CSV 文件写入 ===


def write_json_file(path: str | Path, payload: Any) -> None:
    """将有效载荷写入 JSON 文件（支持 .gz 压缩）。

    使用紧凑分隔符 (",", ":") 减少文件体积，
    ensure_ascii=False 保留中文字符。
    """
    output_path = Path(path)
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as json_file:
            json.dump(payload, json_file, ensure_ascii=False, separators=(",", ":"))
            json_file.write("\n")
        return

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, separators=(",", ":"))
        json_file.write("\n")


def write_time_json_file(
    path: str | Path,
    payload: TimePayload,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> None:
    """写入时间序列 JSON（含进度报告和取消检查）。"""
    _check_cancelled(cancel_event)
    write_json_file(path, payload)
    _report_progress(progress_callback, "write", 1, 3, f"已写入 {Path(path).name}")
    _check_cancelled(cancel_event)


def write_spectrum_json_file(
    path: str | Path,
    payload: SpectrumPayload,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> None:
    """写入频谱 JSON。"""
    _check_cancelled(cancel_event)
    write_json_file(path, payload)
    _report_progress(progress_callback, "write", 2, 3, f"已写入 {Path(path).name}")
    _check_cancelled(cancel_event)


def write_average_json_file(
    path: str | Path,
    payload: AveragePayload,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> None:
    """写入空间平均 JSON。"""
    _check_cancelled(cancel_event)
    write_json_file(path, payload)
    _report_progress(progress_callback, "write", 3, 3, f"已写入 {Path(path).name}")
    _check_cancelled(cancel_event)


def write_surface_geometry_npz(path: str | Path, geometry: SurfaceGeometry) -> None:
    """保存表面几何为压缩 NPZ 文件。"""
    coordinates = np.asarray(geometry.coordinates, dtype=float)
    areas = np.asarray(geometry.areas, dtype=float)
    np.savez_compressed(
        path,
        faces=geometry.faces,
        centers=geometry.centers,
        area_vectors=geometry.area_vectors,
        areas=geometry.areas,
        normals=geometry.normals,
        coordinates=coordinates,
        connectivity_path=np.asarray(geometry.connectivity_path),
        element_range_path=np.asarray(geometry.element_range_path),
        geometry_cache_schema_version=np.asarray(SURFACE_GEOMETRY_CACHE_SCHEMA_VERSION),
        node_count=np.asarray(coordinates.shape[0]),
        face_count=np.asarray(geometry.faces.shape[0]),
        coordinate_min=np.min(coordinates, axis=0),
        coordinate_max=np.max(coordinates, axis=0),
        total_area=np.asarray(float(np.sum(areas))),
    )


def validate_cached_surface_geometry(geometry: SurfaceGeometry) -> None:
    """Validate cached geometry arrays before they are reused."""
    coordinates = np.asarray(geometry.coordinates, dtype=float)
    faces = np.asarray(geometry.faces, dtype=int)
    centers = np.asarray(geometry.centers, dtype=float)
    area_vectors = np.asarray(geometry.area_vectors, dtype=float)
    areas = np.asarray(geometry.areas, dtype=float)
    normals = np.asarray(geometry.normals, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Cached geometry coordinates must have x/y/z columns.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Cached geometry faces must have three vertex indices.")
    face_count = faces.shape[0]
    if face_count == 0:
        raise ValueError("Cached geometry must contain at least one face.")
    if centers.shape != (face_count, 3):
        raise ValueError("Cached geometry centers shape does not match faces.")
    if area_vectors.shape != (face_count, 3):
        raise ValueError("Cached geometry area_vectors shape does not match faces.")
    if areas.shape != (face_count,):
        raise ValueError("Cached geometry areas shape does not match faces.")
    if normals.shape != (face_count, 3):
        raise ValueError("Cached geometry normals shape does not match faces.")
    if not (
        np.all(np.isfinite(coordinates))
        and np.all(np.isfinite(centers))
        and np.all(np.isfinite(area_vectors))
        and np.all(np.isfinite(areas))
        and np.all(np.isfinite(normals))
    ):
        raise ValueError("Cached geometry contains NaN or infinite values.")
    if np.any(faces < 0) or np.any(faces >= coordinates.shape[0]):
        raise ValueError("Cached geometry faces contain indices outside the coordinate array.")
    if np.any(areas <= 0.0):
        raise ValueError("Cached geometry areas must be positive.")


def load_surface_geometry_npz(
    path: str | Path,
    expected_face_count: int | None = None,
) -> SurfaceGeometry:
    """从 NPZ 文件加载缓存的表面几何。

    Args:
        path: NPZ 文件路径。
        expected_face_count: 预期的面元数（用于验证）。

    Returns:
        SurfaceGeometry 实例。
    """
    with np.load(path) as saved:
        geometry = SurfaceGeometry(
            coordinates=np.asarray(saved["coordinates"], dtype=float),
            faces=np.asarray(saved["faces"], dtype=int),
            centers=np.asarray(saved["centers"], dtype=float),
            area_vectors=np.asarray(saved["area_vectors"], dtype=float),
            areas=np.asarray(saved["areas"], dtype=float),
            normals=np.asarray(saved["normals"], dtype=float),
            connectivity_path=str(saved["connectivity_path"])
            if "connectivity_path" in saved
            else "",
            element_range_path=str(saved["element_range_path"])
            if "element_range_path" in saved
            else "",
        )
    validate_cached_surface_geometry(geometry)
    if expected_face_count is not None and geometry.faces.shape[0] != int(
        expected_face_count
    ):
        raise ValueError(
            f"Cached geometry has {geometry.faces.shape[0]} faces, "
            f"but pressure has {expected_face_count} channels."
        )
    return geometry


def write_pressure_complex_spectrum_npz(
    path: str | Path,
    spectrum: ComplexPressureSpectrum,
) -> None:
    """保存复数压力谱为压缩 NPZ。"""
    np.savez_compressed(
        path,
        frequencies_hz=spectrum.frequencies_hz,
        pressure_real=spectrum.pressure_real,
        pressure_imag=spectrum.pressure_imag,
        pressure_amplitude=spectrum.pressure_amplitude,
        pressure_phase_rad=spectrum.pressure_phase_rad,
    )


def write_equivalent_force_spectrum_csv(
    path: str | Path,
    force_spectrum: dict[str, np.ndarray],
) -> None:
    """将等效力谱写入 CSV 文件。

    包含 16 列：频率 + 3× 力分量(实部/虚部/幅值/相位) + 合力模 + 统计量。
    """
    fieldnames = [
        "frequency_hz",
        "force_real_x", "force_real_y", "force_real_z",
        "force_imag_x", "force_imag_y", "force_imag_z",
        "force_amplitude_x", "force_amplitude_y", "force_amplitude_z",
        "force_phase_x", "force_phase_y", "force_phase_z",
        "force_magnitude",
        "max_pressure_amplitude",
        "rms_pressure_amplitude",
        "coherent_pressure_sum",
    ]
    frequencies = np.asarray(force_spectrum["frequencies_hz"], dtype=float)
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, frequency in enumerate(frequencies):
            row = {"frequency_hz": float(frequency)}
            for fieldname in fieldnames[1:]:
                row[fieldname] = float(np.asarray(force_spectrum[fieldname])[index])
            writer.writerow(row)


# === 元数据构建 ===


def build_sampling_quality_metadata(sample_count: int, dt: float) -> dict[str, float | int]:
    """Build sampling metadata used to judge FFT resolution and valid frequency range."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    record_duration = float(sample_count) * float(dt)
    return {
        "sample_count": int(sample_count),
        "record_duration_s": record_duration,
        "frequency_resolution_hz": 1.0 / record_duration,
        "nyquist_hz": 0.5 / float(dt),
    }


def build_extraction_metadata(
    series: PressureTimeSeries,
    dt: float,
    include_dc: bool,
    remove_mean: bool,
    geometry: SurfaceGeometry | None = None,
) -> dict[str, Any]:
    """构建提取操作的元数据字典。"""
    metadata: dict[str, Any] = {
        "pressure_dataset_path": series.dataset_path,
        "time_step_s": float(dt),
        "include_dc": bool(include_dc),
        "remove_mean": bool(remove_mean),
        "file_count": len(series.file_paths),
        "pressure_channel_count": int(series.pressures.shape[1]),
        "source_files": [str(path) for path in series.file_paths],
    }
    metadata.update(build_sampling_quality_metadata(series.pressures.shape[0], dt))
    if geometry is not None:
        metadata.update({
            "node_count": int(geometry.coordinates.shape[0]),
            "face_count": int(geometry.faces.shape[0]),
            "connectivity_path": geometry.connectivity_path,
            "element_range_path": geometry.element_range_path,
        })
    else:
        metadata["node_count"] = int(series.node_ids.size)
        metadata["face_count"] = None
    return metadata


def build_streaming_extraction_metadata(
    file_paths: Iterable[str | Path],
    pressure_dataset_path: str,
    pressure_channel_count: int,
    dt: float,
    include_dc: bool,
    remove_mean: bool,
    geometry: SurfaceGeometry,
    surface_geometry_cache: str | Path | None = None,
    pressure_block_size: int | None = None,
) -> dict[str, Any]:
    """构建流式提取操作的元数据字典。"""
    paths = [Path(path) for path in file_paths]
    metadata: dict[str, Any] = {
        "pressure_dataset_path": pressure_dataset_path,
        "time_step_s": float(dt),
        "include_dc": bool(include_dc),
        "remove_mean": bool(remove_mean),
        "file_count": len(paths),
        "pressure_channel_count": int(pressure_channel_count),
        "source_files": [str(path) for path in paths],
        "node_count": int(geometry.coordinates.shape[0]),
        "face_count": int(geometry.faces.shape[0]),
        "connectivity_path": geometry.connectivity_path,
        "element_range_path": geometry.element_range_path,
        "streaming_equivalent_force": True,
    }
    metadata.update(build_sampling_quality_metadata(len(paths), dt))
    if surface_geometry_cache is not None:
        metadata["surface_geometry_cache"] = str(surface_geometry_cache)
    if pressure_block_size is not None:
        metadata["pressure_block_size"] = int(pressure_block_size)
    return metadata


# === 增强输出 ===


def write_enhanced_outputs(
    output_dir: str | Path,
    series: PressureTimeSeries,
    dt: float,
    include_dc: bool = False,
    export_complex_spectrum: bool = False,
    export_surface_geometry: bool = False,
    export_equivalent_force: bool = False,
    surface_geometry_cache: str | Path | None = None,
    h5_module: Any | None = None,
) -> dict[str, Path]:
    """写入增强输出：表面几何 NPZ、复数压力谱 NPZ、等效力 CSV。

    优先使用缓存的表面几何（surface_geometry_cache）以加速重复运行。

    Args:
        output_dir: 输出目录。
        series: 压力时间序列。
        dt: 时间步长。
        include_dc: 是否保留 DC。
        export_complex_spectrum: 是否导出复数压力谱。
        export_surface_geometry: 是否导出表面几何。
        export_equivalent_force: 是否导出等效力谱。
        surface_geometry_cache: 表面几何 NPZ 缓存路径。
        h5_module: h5py 模块。

    Returns:
        已写入文件的路径字典。
    """
    if not (export_complex_spectrum or export_surface_geometry or export_equivalent_force):
        return {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    geometry: SurfaceGeometry | None = None
    spectrum: ComplexPressureSpectrum | None = None

    if export_surface_geometry or export_equivalent_force:
        if surface_geometry_cache is not None:
            geometry = load_surface_geometry_npz(
                surface_geometry_cache,
                expected_face_count=series.pressures.shape[1],
            )
        else:
            geometry = read_surface_geometry(
                series.file_paths[0],
                expected_face_count=series.pressures.shape[1],
                h5_module=h5_module,
            )
        if export_surface_geometry:
            geometry_path = output_path / "surface_geometry.npz"
            write_surface_geometry_npz(geometry_path, geometry)
            paths["surface_geometry"] = geometry_path

    if export_complex_spectrum or export_equivalent_force:
        pulsating_pressure = compute_pulsating_pressure(series.pressures)
        spectrum = compute_pressure_complex_spectrum(
            pulsating_pressure, dt=dt, include_dc=include_dc, remove_mean=False,
        )
        if export_complex_spectrum:
            spectrum_path = output_path / "pressure_complex_spectrum.npz"
            write_pressure_complex_spectrum_npz(spectrum_path, spectrum)
            paths["pressure_complex_spectrum"] = spectrum_path

    if export_equivalent_force:
        if geometry is None or spectrum is None:
            raise RuntimeError("surface geometry and complex pressure spectrum are required.")
        force_spectrum = compute_equivalent_force_spectrum(spectrum, geometry.area_vectors)
        force_path = output_path / "equivalent_force_spectrum.csv"
        write_equivalent_force_spectrum_csv(force_path, force_spectrum)
        paths["equivalent_force_spectrum"] = force_path

    metadata_path = output_path / "extraction_metadata.json"
    write_json_file(
        metadata_path,
        build_extraction_metadata(
            series, dt=dt, include_dc=include_dc, remove_mean=True, geometry=geometry,
        ),
    )
    paths["extraction_metadata"] = metadata_path
    return paths


def write_streaming_summary_outputs(
    output_dir: str | Path,
    file_paths: Iterable[str | Path],
    dt: float,
    pressure_name: str | None = None,
    include_dc: bool = False,
    surface_geometry_cache: str | Path | None = None,
    export_surface_geometry: bool = True,
    pressure_block_size: int = 16384,
    h5_module: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> dict[str, Path]:
    """流式汇总模式输出：仅写入表面几何和等效力汇总。

    跳过全节点时间序列和频谱 JSON，适合大规模网格场景。

    Args:
        output_dir: 输出目录。
        file_paths: CGNS 文件列表。
        dt: 时间步长。
        pressure_name: 压力变量名。
        include_dc: 是否保留 DC。
        surface_geometry_cache: 几何缓存路径。
        export_surface_geometry: 是否导出几何。
        pressure_block_size: FFT 分块大小。
        h5_module: h5py 模块。
        progress_callback: 进度回调。
        cancel_event: 取消事件。

    Returns:
        已写入文件的路径字典。
    """
    paths = sort_time_step_paths(file_paths)
    if not paths:
        raise ValueError("At least one CGNS file is required.")
    if pressure_block_size <= 0:
        raise ValueError("pressure_block_size must be positive.")

    h5 = h5_module if h5_module is not None else _load_h5py()
    first_dataset_path, first_vector = _read_pressure_vector_from_file(
        h5, paths[0], pressure_name,
    )
    if surface_geometry_cache is not None:
        geometry = load_surface_geometry_npz(
            surface_geometry_cache,
            expected_face_count=first_vector.size,
        )
    else:
        geometry = read_surface_geometry(
            paths[0],
            expected_face_count=first_vector.size,
            h5_module=h5,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if export_surface_geometry:
        geometry_path = output_path / "surface_geometry.npz"
        write_surface_geometry_npz(geometry_path, geometry)
        written["surface_geometry"] = geometry_path

    force_spectrum = compute_streaming_equivalent_force_summary(
        paths, geometry, dt=dt, include_dc=include_dc,
        pressure_name=pressure_name, pressure_block_size=pressure_block_size,
        h5_module=h5, progress_callback=progress_callback, cancel_event=cancel_event,
    )
    force_path = output_path / "equivalent_force_spectrum.csv"
    write_equivalent_force_spectrum_csv(force_path, force_spectrum)
    written["equivalent_force_spectrum"] = force_path

    metadata_path = output_path / "extraction_metadata.json"
    write_json_file(
        metadata_path,
        build_streaming_extraction_metadata(
            paths,
            pressure_dataset_path=first_dataset_path,
            pressure_channel_count=first_vector.size,
            dt=dt, include_dc=include_dc, remove_mean=True,
            geometry=geometry,
            surface_geometry_cache=surface_geometry_cache,
            pressure_block_size=pressure_block_size,
        ),
    )
    written["extraction_metadata"] = metadata_path
    return written


def write_outputs(
    output_dir: str | Path,
    series: PressureTimeSeries,
    dt: float,
    include_dc: bool = False,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> tuple[Path, Path, Path]:
    """写入传统 JSON 输出三件套：时间序列 + 频谱 + 平均。

    这是 old-style 输出（每个 .gz 文件约数 GB）。

    Args:
        output_dir: 输出目录。
        series: 压力时间序列。
        dt: 时间步长。
        include_dc: 是否保留 DC。
        progress_callback: 进度回调。
        cancel_event: 取消事件。

    Returns:
        (time_json, spectrum_json, average_json) 路径元组。
    """
    _check_cancelled(cancel_event)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pulsating_pressure, times, metadata, node_ids, coordinates = _payload_common_parts(
        series, dt, include_dc,
    )
    _check_cancelled(cancel_event)

    time_json = output_path / "pressure_time.json.gz"
    spectrum_json = output_path / "pressure_spectrum.json.gz"
    average_json = output_path / "pressure_average.json.gz"

    time_payload = build_time_payload(metadata, times, node_ids, coordinates, pulsating_pressure)
    write_time_json_file(time_json, time_payload, progress_callback, cancel_event)
    del time_payload  # 及时释放内存

    spectrum_payload = build_spectrum_payload(
        metadata, node_ids, coordinates, pulsating_pressure, dt, include_dc,
    )
    write_spectrum_json_file(spectrum_json, spectrum_payload, progress_callback, cancel_event)
    del spectrum_payload

    average_payload = build_average_payload(metadata, times, pulsating_pressure, dt, include_dc)
    del pulsating_pressure
    write_average_json_file(average_json, average_payload, progress_callback, cancel_event)
    return time_json, spectrum_json, average_json


# === CLI / GUI ===


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "Extract all-node pulsating pressure data from STAR-CCM+ "
            "HDF5-CGNS time-step files."
        )
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="CGNS file path or glob pattern, for example other_files/data/604@*.cgns",
    )
    parser.add_argument(
        "--dt", type=float, required=True,
        help="Physical time interval between adjacent CGNS files, in seconds.",
    )
    parser.add_argument(
        "--output-dir", default="cgns_pressure_output",
        help="Directory for pressure_time.json.gz, pressure_spectrum.json.gz, and pressure_average.json.gz.",
    )
    parser.add_argument("--pressure-name", help="Pressure dataset basename or full CGNS/HDF5 path.")
    parser.add_argument("--include-dc", action="store_true", help="Keep the 0 Hz component.")
    parser.add_argument(
        "--export-complex-spectrum", action="store_true",
        help="Write pressure_complex_spectrum.npz with raw complex FFT pressure data.",
    )
    parser.add_argument(
        "--export-surface-geometry", action="store_true",
        help="Write surface_geometry.npz with triangle faces, centers, areas, and normals.",
    )
    parser.add_argument(
        "--export-equivalent-force", action="store_true",
        help="Write equivalent_force_spectrum.csv from coherent face pressure integration.",
    )
    parser.add_argument(
        "--export-analysis-summary", action="store_true",
        help="Write summary analysis outputs only: surface_geometry.npz, equivalent_force_spectrum.csv, and extraction_metadata.json.",
    )
    parser.add_argument(
        "--export-all-data", action="store_true",
        help="Write summary analysis outputs plus the full pressure_complex_spectrum.npz.",
    )
    parser.add_argument(
        "--skip-legacy-json", action="store_true",
        help="Skip legacy JSON outputs, implies summary outputs.",
    )
    parser.add_argument(
        "--surface-geometry-cache",
        help="Load an existing surface_geometry.npz instead of re-reading CGNS geometry.",
    )
    parser.add_argument(
        "--pressure-block-size", type=int, default=16384,
        help="Number of pressure faces per FFT block for streaming summary outputs.",
    )
    return parser.parse_args(argv)


def resolve_enhanced_output_options(args: argparse.Namespace) -> dict[str, bool]:
    """解析增强输出的选项组合。

    --export-all-data 自动启用所有子选项；
    --export-analysis-summary 和 --skip-legacy-json 自动启用
    表面几何和等效力导出。
    """
    export_all_data = bool(getattr(args, "export_all_data", False))
    export_analysis_summary = bool(getattr(args, "export_analysis_summary", False))
    skip_legacy_json = bool(getattr(args, "skip_legacy_json", False))
    export_equivalent_force = (
        bool(getattr(args, "export_equivalent_force", False))
        or export_analysis_summary or export_all_data or skip_legacy_json
    )
    export_surface_geometry = (
        bool(getattr(args, "export_surface_geometry", False))
        or export_equivalent_force or export_analysis_summary
        or export_all_data or skip_legacy_json
    )
    export_complex_spectrum = (
        bool(getattr(args, "export_complex_spectrum", False)) or export_all_data
    )
    return {
        "export_complex_spectrum": export_complex_spectrum,
        "export_surface_geometry": export_surface_geometry,
        "export_equivalent_force": export_equivalent_force,
    }


def run_cli(argv: list[str] | None = None) -> int:
    """CLI 模式入口。

    Args:
        argv: 命令行参数。

    Returns:
        退出码。
    """
    args = parse_args(argv)
    series: PressureTimeSeries | None = None
    time_json: Path | None = None
    spectrum_json: Path | None = None
    average_json: Path | None = None
    try:
        paths = expand_input_files(args.inputs)
        enhanced_options = resolve_enhanced_output_options(args)
        # 判断是否可用流式汇总模式（跳过旧版 JSON + 不需要全场复数谱）
        use_streaming_summary = (
            args.skip_legacy_json
            and enhanced_options["export_equivalent_force"]
            and not enhanced_options["export_complex_spectrum"]
        )
        if use_streaming_summary:
            enhanced_outputs = write_streaming_summary_outputs(
                args.output_dir, paths,
                dt=args.dt, pressure_name=args.pressure_name,
                include_dc=args.include_dc,
                surface_geometry_cache=args.surface_geometry_cache,
                export_surface_geometry=enhanced_options["export_surface_geometry"],
                pressure_block_size=args.pressure_block_size,
            )
        else:
            series = read_pressure_time_series(paths, pressure_name=args.pressure_name)
            if not args.skip_legacy_json:
                time_json, spectrum_json, average_json = write_outputs(
                    args.output_dir, series, dt=args.dt, include_dc=args.include_dc,
                )
            enhanced_outputs = write_enhanced_outputs(
                args.output_dir, series,
                dt=args.dt, include_dc=args.include_dc,
                surface_geometry_cache=args.surface_geometry_cache,
                **enhanced_options,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if series is not None:
        print(f"Read {len(series.file_paths)} CGNS files.")
        print(f"Pressure dataset: {series.dataset_path}")
        print(f"Pressure channels: {series.node_ids.size}")
    else:
        print(f"Processed {len(paths)} CGNS files with streaming summary mode.")
    for path in (time_json, spectrum_json, average_json):
        if path is not None:
            print(f"Wrote: {path}")
    for path in enhanced_outputs.values():
        print(f"Wrote: {path}")
    return 0


def build_success_message(
    series: PressureTimeSeries,
    time_json: Path | None,
    spectrum_json: Path | None,
    average_json: Path | None,
    enhanced_outputs: dict[str, Path] | None = None,
) -> str:
    """构建 GUI 成功消息字符串。"""
    written_paths = [
        path for path in (time_json, spectrum_json, average_json) if path is not None
    ]
    if enhanced_outputs:
        written_paths.extend(enhanced_outputs.values())
    coordinate_status = "available" if series.coordinates is not None else "not available"
    return (
        f"读取 {len(series.file_paths)} 个文件，{series.node_ids.size} 个压力通道。\n"
        f"旧版 JSON 坐标：{'可用' if coordinate_status == 'available' else '不可用'}。\n"
        "已写入：\n"
        + "\n".join(str(path) for path in written_paths)
    )


def build_streaming_success_message(
    file_paths: Iterable[str | Path],
    enhanced_outputs: dict[str, Path],
) -> str:
    """构建流式汇总模式的成功消息。"""
    paths = [Path(path) for path in file_paths]
    return (
        f"流式汇总处理 {len(paths)} 个文件。\n"
        "已写入：\n"
        + "\n".join(str(path) for path in enhanced_outputs.values())
    )


def gui_default_export_options() -> dict[str, bool]:
    """Return GUI defaults optimized for lightweight hand-off usage."""
    return {
        "skip_legacy_json": True,
        "export_complex_spectrum": False,
        "export_surface_geometry": True,
        "export_equivalent_force": True,
    }


def run_gui_extraction_job(
    raw_inputs: list[str],
    *,
    dt: float,
    output_dir: str,
    pressure_name: str | None,
    include_dc: bool,
    skip_legacy_json: bool,
    export_complex_spectrum: bool,
    export_surface_geometry: bool,
    export_equivalent_force: bool,
    surface_geometry_cache: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> str:
    """运行 GUI 触发的提取作业（在后台线程中运行）。

    Args:
        raw_inputs: 原始输入字符串列表（分号分隔的文件路径）。
        dt: 时间步长。
        output_dir: 输出目录。
        pressure_name: 压力变量名。
        include_dc: 是否保留 DC。
        skip_legacy_json: 是否跳过旧版 JSON。
        export_complex_spectrum: 是否导出复数谱。
        export_surface_geometry: 是否导出几何。
        export_equivalent_force: 是否导出等效力。
        surface_geometry_cache: 可复用的表面几何缓存路径。
        progress_callback: 进度回调。
        cancel_event: 取消事件。

    Returns:
        成功消息字符串。
    """
    paths = expand_input_files(raw_inputs)
    use_default_summary_outputs = skip_legacy_json and not (
        export_complex_spectrum or export_surface_geometry or export_equivalent_force
    )
    effective_export_surface_geometry = (
        export_surface_geometry or use_default_summary_outputs
    )
    effective_export_equivalent_force = (
        export_equivalent_force or use_default_summary_outputs
    )
    use_streaming_summary = (
        skip_legacy_json
        and effective_export_equivalent_force
        and not export_complex_spectrum
    )

    if use_streaming_summary:
        enhanced_outputs = write_streaming_summary_outputs(
            output_dir, paths, dt=dt, pressure_name=pressure_name,
            include_dc=include_dc,
            surface_geometry_cache=surface_geometry_cache,
            export_surface_geometry=effective_export_surface_geometry,
            progress_callback=progress_callback, cancel_event=cancel_event,
        )
        return build_streaming_success_message(paths, enhanced_outputs)

    series = read_pressure_time_series(
        paths, pressure_name=pressure_name,
        progress_callback=progress_callback, cancel_event=cancel_event,
    )
    if skip_legacy_json:
        time_json = spectrum_json = average_json = None
    else:
        time_json, spectrum_json, average_json = write_outputs(
            output_dir, series, dt=dt, include_dc=include_dc,
            progress_callback=progress_callback, cancel_event=cancel_event,
        )
    enhanced_outputs = write_enhanced_outputs(
        output_dir, series, dt=dt, include_dc=include_dc,
        export_complex_spectrum=export_complex_spectrum,
        export_surface_geometry=effective_export_surface_geometry,
        export_equivalent_force=effective_export_equivalent_force,
        surface_geometry_cache=surface_geometry_cache,
    )
    return build_success_message(series, time_json, spectrum_json, average_json, enhanced_outputs)


def run_gui() -> int:
    """GUI 模式入口——构建 tkinter 窗口并启动事件循环。

    Returns:
        退出码。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        print(f"Error: Tkinter is required for GUI mode: {exc}", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("CGNS 压力数据提取")
    root.geometry("820x620")
    root.resizable(False, False)

    # --- GUI 状态变量 ---
    default_exports = gui_default_export_options()
    input_var = tk.StringVar(value="other_files\\数据\\604@*.cgns")
    dt_var = tk.StringVar(value="0.001")
    output_var = tk.StringVar(value="cgns_pressure_output")
    pressure_name_var = tk.StringVar(value="Pressure")
    surface_geometry_cache_var = tk.StringVar(value="")
    include_dc_var = tk.BooleanVar(value=False)
    skip_legacy_json_var = tk.BooleanVar(value=default_exports["skip_legacy_json"])
    export_complex_spectrum_var = tk.BooleanVar(
        value=default_exports["export_complex_spectrum"],
    )
    export_surface_geometry_var = tk.BooleanVar(
        value=default_exports["export_surface_geometry"],
    )
    export_equivalent_force_var = tk.BooleanVar(
        value=default_exports["export_equivalent_force"],
    )
    status_var = tk.StringVar(value="就绪。")
    progress_var = tk.DoubleVar(value=0.0)
    progress_text_var = tk.StringVar(value="")
    cancel_event = threading.Event()

    main_frame = ttk.Frame(root, padding=16)
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.columnconfigure(1, weight=1)

    def browse_input() -> None:
        files = filedialog.askopenfilenames(
            title="选择 CGNS 文件",
            filetypes=[("CGNS 文件", "*.cgns"), ("所有文件", "*.*")],
        )
        if files:
            input_var.set(";".join(files))

    def browse_output() -> None:
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            output_var.set(directory)

    def browse_surface_geometry_cache() -> None:
        file_path = filedialog.askopenfilename(
            title="选择表面几何缓存",
            filetypes=[("NPZ 文件", "*.npz"), ("所有文件", "*.*")],
        )
        if file_path:
            surface_geometry_cache_var.set(file_path)

    def clear_surface_geometry_cache() -> None:
        surface_geometry_cache_var.set("")

    def set_running(is_running: bool) -> None:
        """切换 UI 的启用/禁用状态。"""
        state = "disabled" if is_running else "normal"
        run_button.configure(state=state)
        input_button.configure(state=state)
        output_button.configure(state=state)
        cache_button.configure(state=state)
        clear_cache_button.configure(state=state)
        include_dc_check.configure(state=state)
        skip_legacy_json_check.configure(state=state)
        complex_spectrum_check.configure(state=state)
        surface_geometry_check.configure(state=state)
        equivalent_force_check.configure(state=state)
        cancel_button.configure(state="normal" if is_running else "disabled")

    def collect_inputs():
        """收集并校验所有 GUI 输入。"""
        raw_inputs = [
            item.strip()
            for item in input_var.get().split(";")
            if item.strip()
        ]
        if not raw_inputs:
            raise ValueError("请选择至少一个 CGNS 文件或输入匹配模式。")

        try:
            dt = float(dt_var.get())
        except ValueError as exc:
            raise ValueError("时间步长 dt 必须是数字。") from exc
        if dt <= 0.0:
            raise ValueError("时间步长 dt 必须大于 0。")

        output_dir = output_var.get().strip()
        if not output_dir:
            raise ValueError("请选择输出目录。")

        pressure_name = pressure_name_var.get().strip() or None
        surface_geometry_cache = surface_geometry_cache_var.get().strip() or None
        return (
            raw_inputs, dt, output_dir, pressure_name,
            surface_geometry_cache,
            bool(include_dc_var.get()), bool(skip_legacy_json_var.get()),
            bool(export_complex_spectrum_var.get()),
            bool(export_surface_geometry_var.get()),
            bool(export_equivalent_force_var.get()),
        )

    def finish_success(message: str) -> None:
        progress_var.set(100.0)
        progress_text_var.set("已完成。")
        status_var.set(message)
        set_running(False)
        messagebox.showinfo("完成", message)

    def finish_error(message: str) -> None:
        progress_text_var.set("")
        status_var.set(f"错误：{message}")
        set_running(False)
        messagebox.showerror("提取失败", message)

    def finish_cancelled() -> None:
        progress_text_var.set("已取消。")
        status_var.set("已取消。")
        set_running(False)
        messagebox.showinfo("已取消", "操作已取消。")

    def cancel_extraction() -> None:
        cancel_event.set()
        status_var.set("正在取消...")
        progress_text_var.set("正在取消...")

    def update_progress(event: ProgressEvent) -> None:
        """在主线程中更新进度条（通过 root.after 线程安全调用）。"""
        total = max(int(event.get("total", 1)), 1)
        current = min(max(int(event.get("current", 0)), 0), total)
        stage = str(event.get("stage", ""))
        if stage == "read":
            percent = 70.0 * current / total
        elif stage == "write":
            percent = 70.0 + 30.0 * current / total
        else:
            percent = 100.0 * current / total
        progress_var.set(percent)
        progress_text_var.set(str(event.get("message", "")))

    def report_progress(event: ProgressEvent) -> None:
        root.after(0, update_progress, event)

    def run_extraction() -> None:
        """校验输入后在后台线程中启动提取作业。"""
        try:
            (
                raw_inputs, dt, output_dir, pressure_name,
                surface_geometry_cache,
                include_dc, skip_legacy_json,
                export_complex_spectrum, export_surface_geometry,
                export_equivalent_force,
            ) = collect_inputs()
        except Exception as exc:
            messagebox.showerror("输入无效", str(exc))
            return

        try:
            ordered_preview_paths = expand_input_files(raw_inputs)
        except Exception as exc:
            messagebox.showerror("输入无效", str(exc))
            return

        cancel_event.clear()
        progress_var.set(0.0)
        progress_text_var.set("")
        set_running(True)
        status_var.set(
            build_input_file_order_message(ordered_preview_paths)
            + "\n正在读取 CGNS 文件..."
        )

        def worker() -> None:
            try:
                message = run_gui_extraction_job(
                    raw_inputs, dt=dt, output_dir=output_dir,
                    pressure_name=pressure_name, include_dc=include_dc,
                    skip_legacy_json=skip_legacy_json,
                    export_complex_spectrum=export_complex_spectrum,
                    export_surface_geometry=export_surface_geometry,
                    export_equivalent_force=export_equivalent_force,
                    surface_geometry_cache=surface_geometry_cache,
                    progress_callback=report_progress,
                    cancel_event=cancel_event,
                )
            except OperationCancelled:
                root.after(0, finish_cancelled)
                return
            except Exception as exc:
                root.after(0, finish_error, str(exc))
                return

            root.after(0, finish_success, message)

        threading.Thread(target=worker, daemon=True).start()

    # --- GUI 布局 ---
    # Rows of label, entry, button widgets
    ttk.Label(main_frame, text="CGNS 文件或匹配模式").grid(row=0, column=0, sticky="w", pady=6)
    ttk.Entry(main_frame, textvariable=input_var, width=70).grid(row=0, column=1, sticky="ew", pady=6)
    input_button = ttk.Button(main_frame, text="选择文件", command=browse_input)
    input_button.grid(row=0, column=2, padx=(8, 0), pady=6)

    ttk.Label(main_frame, text="时间步长 dt (s)").grid(row=1, column=0, sticky="w", pady=6)
    ttk.Entry(main_frame, textvariable=dt_var, width=18).grid(row=1, column=1, sticky="w", pady=6)

    ttk.Label(main_frame, text="输出目录").grid(row=2, column=0, sticky="w", pady=6)
    ttk.Entry(main_frame, textvariable=output_var, width=70).grid(row=2, column=1, sticky="ew", pady=6)
    output_button = ttk.Button(main_frame, text="浏览", command=browse_output)
    output_button.grid(row=2, column=2, padx=(8, 0), pady=6)

    ttk.Label(main_frame, text="压力变量").grid(row=3, column=0, sticky="w", pady=6)
    ttk.Entry(main_frame, textvariable=pressure_name_var, width=24).grid(row=3, column=1, sticky="w", pady=6)

    ttk.Label(main_frame, text="表面几何缓存").grid(row=4, column=0, sticky="w", pady=6)
    ttk.Entry(main_frame, textvariable=surface_geometry_cache_var, width=70).grid(row=4, column=1, sticky="ew", pady=6)
    cache_button = ttk.Button(main_frame, text="选择缓存", command=browse_surface_geometry_cache)
    cache_button.grid(row=4, column=2, padx=(8, 0), pady=6)
    clear_cache_button = ttk.Button(main_frame, text="清空", command=clear_surface_geometry_cache)
    clear_cache_button.grid(row=4, column=3, padx=(8, 0), pady=6)

    include_dc_check = ttk.Checkbutton(main_frame, text="包含 0 Hz 分量", variable=include_dc_var)
    include_dc_check.grid(row=5, column=1, sticky="w", pady=6)

    skip_legacy_json_check = ttk.Checkbutton(
        main_frame, text="跳过旧版 JSON 输出（适合大量文件）", variable=skip_legacy_json_var,
    )
    skip_legacy_json_check.grid(row=6, column=1, sticky="w", pady=3)

    complex_spectrum_check = ttk.Checkbutton(
        main_frame, text="导出全场复数压力谱（.npz，文件很大）", variable=export_complex_spectrum_var,
    )
    complex_spectrum_check.grid(row=7, column=1, sticky="w", pady=3)

    surface_geometry_check = ttk.Checkbutton(
        main_frame, text="导出表面几何汇总（.npz）", variable=export_surface_geometry_var,
    )
    surface_geometry_check.grid(row=8, column=1, sticky="w", pady=3)

    equivalent_force_check = ttk.Checkbutton(
        main_frame, text="导出等效复数激励力汇总（.csv）", variable=export_equivalent_force_var,
    )
    equivalent_force_check.grid(row=9, column=1, sticky="w", pady=3)

    run_button = ttk.Button(main_frame, text="开始提取", command=run_extraction)
    run_button.grid(row=10, column=1, sticky="w", pady=(14, 8))
    cancel_button = ttk.Button(main_frame, text="取消", command=cancel_extraction, state="disabled")
    cancel_button.grid(row=10, column=1, sticky="w", padx=(140, 0), pady=(14, 8))

    ttk.Progressbar(main_frame, variable=progress_var, maximum=100.0, length=420).grid(
        row=11, column=1, sticky="ew", pady=(8, 0),
    )
    ttk.Label(main_frame, textvariable=progress_text_var, wraplength=650).grid(
        row=12, column=0, columnspan=4, sticky="w", pady=(6, 0),
    )
    ttk.Label(main_frame, textvariable=status_var, wraplength=650).grid(
        row=13, column=0, columnspan=4, sticky="w", pady=(12, 0),
    )

    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    """主入口：无参数启动 GUI，有参数走 CLI。

    Args:
        argv: 命令行参数。

    Returns:
        退出码。
    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return run_gui()
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())


# === 注释说明 ===
# 1. 模块级 docstring：说明提取流程、关键数据结构和函数索引。
# 2. 数据类 docstring：PressureTimeSeries/SurfaceGeometry/ComplexPressureSpectrum
#    均有完整的 Attributes 说明。
# 3. TypedDict docstring：ProgressEvent/MetadataPayload/TimePayload 等 JSON
#    载荷类型均有字段说明。
# 4. 函数 docstring（Google 风格）：所有公开函数含 Args/Returns/Raises 说明，
#    重点解释了：
#    - HDF5 数据集查找的三级匹配策略（精确名/候选列表/报错提示）
#    - 流式分块 FFT 的两阶段算法（力时间序列 + 分块统计）
#    - 单边幅值谱的缩放系数公式（DC 1/N, 其他 2/N）
#    - 等效力为压力(复数) × 面积矢量的矩阵乘
# 5. 行内注释：对非直观算法步骤（HDF5 多维数组切片、CGNS MIXED 元素码解析、
#    滚动最大/平方和统计、GUI 进度百分比分配等）做了说明。
# 6. 块注释：按功能分为常量、异常、数据结构、HDF5 导航、坐标、几何、
#    进度、读取、FFT、等效力、脉动压力、JSON 载荷、文件写入、元数据、
#    增强输出、CLI/GUI 等逻辑段落。
# 7. 特殊标记：pyinstaller 打包命令注释在模块 docstring 中。
