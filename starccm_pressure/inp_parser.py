"""Small Abaqus INP parser used by the pressure mapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


StepKind = Literal["steady_state", "dynamic_explicit", "dynamic", "unknown"]


@dataclass(frozen=True)
class AbaqusElement:
    element_id: int
    element_type: str
    node_ids: list[int]


@dataclass(frozen=True)
class AbaqusStep:
    name: str
    start_line: int
    end_line: int
    kind: StepKind


@dataclass(frozen=True)
class AbaqusModel:
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


def _keyword_name(line: str) -> str:
    return line.split(",", 1)[0].strip().lower()


def _parse_keyword_params(line: str) -> dict[str, str | bool]:
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
    return [int(value.strip()) for value in line.split(",") if value.strip()]


def _parse_set_values(lines: list[str], start: int, generated: bool) -> tuple[set[int], int]:
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
    key = (set_type.lower(), set_name.upper())
    previous = set_instances.get(key)
    if previous is None and key in set_instances:
        return
    set_instances[key] = instance_name if previous in {None, instance_name} else None


def parse_inp_text(text: str) -> AbaqusModel:
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
    return parse_inp_text(Path(path).read_text(encoding="utf-8"))
