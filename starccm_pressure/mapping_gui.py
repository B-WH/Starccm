"""CGNS 压力到 Abaqus INP 映射的 Tkinter 图形界面。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from starccm_pressure.map_cgns_pressure_to_inp import (
    AlignmentPreview,
    _default_output_path,
    build_alignment_preview,
    parse_frequency_text,
    parse_gui_axis_order,
    parse_gui_axis_sign,
    parse_gui_translate,
    run_mapping,
)


def validate_mapping_gui_inputs(inp_path: str, extracted_dir: str, target_set: str) -> None:
    """运行预览或映射前校验必填输入。"""
    if not inp_path.strip():
        raise ValueError("请选择 INP 文件。")
    if not extracted_dir.strip():
        raise ValueError("请选择 CGNS 提取结果目录。")
    if not target_set.strip():
        raise ValueError("请输入 Abaqus 目标集合名称。")


def format_gui_error(exc: Exception) -> str:
    """把常见映射异常转换为面向用户的中文消息。"""
    message = str(exc)
    if "Required surface geometry file was not found:" in message:
        return f"找不到 surface_geometry.npz。请检查 CGNS 提取结果目录。\n\n原始信息：{message}"
    if "Target elset" in message and "was not found" in message:
        return f"找不到目标 elset。请检查目标集合名称和集合类型。\n\n原始信息：{message}"
    if "Target nset" in message and "was not found" in message:
        return f"找不到目标 nset。请检查目标集合名称和集合类型。\n\n原始信息：{message}"
    if "axis_order must be a permutation of 0, 1, 2" in message:
        return "轴顺序必须是 0,1,2 的排列，例如 0,1,2 或 2,0,1。"
    if "Expected three comma-separated values" in message:
        return "请输入三个逗号分隔的数值，例如 0,1,2 或 1,-1,1。"
    if "比例系数必须是数字" in message:
        return "比例系数必须是数字，例如 1.0。"
    if "Output INP must not overwrite the original INP" in message:
        return "输出 INP 不能覆盖原始 INP。请重新选择输出文件名。"
    return message


def run_gui() -> int:
    """启动基于文件选择器的 Tkinter 映射界面。"""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        print(f"错误：图形界面模式需要 Tkinter：{exc}", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("CGNS 压力到 INP 映射")
    root.geometry("840x680")
    root.resizable(False, False)

    inp_var = tk.StringVar(value="")
    extracted_var = tk.StringVar(value="cgns_pressure_output")
    output_var = tk.StringVar(value="")
    target_set_var = tk.StringVar(value="")
    target_type_var = tk.StringVar(value="elset")
    frequency_var = tk.StringVar(value="1:800:1")
    scale_var = tk.StringVar(value="1.0")
    translate_var = tk.StringVar(value="")
    axis_order_var = tk.StringVar(value="0,1,2")
    axis_sign_var = tk.StringVar(value="1,1,1")
    num_workers_var = tk.StringVar(value="1")
    status_var = tk.StringVar(value="就绪")

    frame = ttk.Frame(root, padding=16)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    def read_transform_options() -> tuple[
        float,
        np.ndarray | None,
        tuple[int, int, int],
        tuple[float, float, float],
    ]:
        try:
            scale = float(scale_var.get().strip() or "1.0")
        except ValueError as exc:
            raise ValueError("比例系数必须是数字。") from exc
        return (
            scale,
            parse_gui_translate(translate_var.get()),
            parse_gui_axis_order(axis_order_var.get()),
            parse_gui_axis_sign(axis_sign_var.get()),
        )

    def sample_points(points: np.ndarray, limit: int = 5000) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.shape[0] <= limit:
            return values
        indices = np.linspace(0, values.shape[0] - 1, limit).astype(int)
        return values[indices]

    def draw_projection(
        canvas: tk.Canvas,
        preview: AlignmentPreview,
        title: str,
        axes: tuple[int, int],
        origin_x: int,
        origin_y: int,
        width: int,
        height: int,
    ) -> None:
        padding = 28
        plot_left = origin_x + padding
        plot_top = origin_y + padding
        plot_right = origin_x + width - padding
        plot_bottom = origin_y + height - padding
        canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline="#8a8a8a")
        canvas.create_text(origin_x + width / 2, origin_y + 14, text=title, fill="#202020")

        x_axis, y_axis = axes
        x_min = float(preview.bounds_min[x_axis])
        x_max = float(preview.bounds_max[x_axis])
        y_min = float(preview.bounds_min[y_axis])
        y_max = float(preview.bounds_max[y_axis])
        if math.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        if math.isclose(y_min, y_max):
            y_min -= 0.5
            y_max += 0.5

        def map_point(point: np.ndarray) -> tuple[float, float]:
            x_value = float(point[x_axis])
            y_value = float(point[y_axis])
            x = plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left)
            y = plot_bottom - (y_value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            return x, y

        for point in sample_points(preview.source_centers):
            x, y = map_point(point)
            canvas.create_oval(x - 1, y - 1, x + 1, y + 1, outline="", fill="#1f77b4")

        for point in sample_points(preview.target_nodes):
            x, y = map_point(point)
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline="#b85c00", fill="#ff9f1c")

        for point in sample_points(preview.target_centers):
            x, y = map_point(point)
            canvas.create_line(x - 3, y, x + 3, y, fill="#202020")
            canvas.create_line(x, y - 3, x, y + 3, fill="#202020")

    def show_alignment_window(preview: AlignmentPreview) -> None:
        window = tk.Toplevel(root)
        window.title("CGNS / INP 坐标对齐预览")
        window.geometry("1000x760")
        window.resizable(False, False)
        summary = (
            f"CGNS 面数：{preview.source_count}    "
            f"INP 目标面数：{preview.target_face_count}    "
            f"INP 目标节点数：{preview.target_node_count}\n"
            "CGNS 到 INP 目标面的最近距离："
            f"min={preview.nearest_distance_min:.6g}, "
            f"mean={preview.nearest_distance_mean:.6g}, "
            f"max={preview.nearest_distance_max:.6g}\n"
            "边界最小值="
            f"({preview.bounds_min[0]:.6g}, {preview.bounds_min[1]:.6g}, {preview.bounds_min[2]:.6g})    "
            "最大值="
            f"({preview.bounds_max[0]:.6g}, {preview.bounds_max[1]:.6g}, {preview.bounds_max[2]:.6g})"
        )
        ttk.Label(window, text=summary, justify="left").pack(anchor="w", padx=12, pady=(12, 6))
        legend = ttk.Label(
            window,
            text="蓝色：变换后的 CGNS 面心    橙色：INP 目标节点    黑色十字：INP 目标面心",
        )
        legend.pack(anchor="w", padx=12, pady=(0, 8))
        canvas = tk.Canvas(window, width=976, height=640, background="white")
        canvas.pack(padx=12, pady=(0, 12))
        draw_projection(canvas, preview, "XY", (0, 1), 8, 8, 320, 620)
        draw_projection(canvas, preview, "XZ", (0, 2), 328, 8, 320, 620)
        draw_projection(canvas, preview, "YZ", (1, 2), 648, 8, 320, 620)

    def browse_inp() -> None:
        path = filedialog.askopenfilename(
            title="选择 Abaqus INP 文件",
            filetypes=[("Abaqus INP 文件", "*.inp"), ("所有文件", "*.*")],
        )
        if path:
            inp_var.set(path)
            if not output_var.get().strip():
                input_path = Path(path)
                output_var.set(str(_default_output_path(input_path)))

    def browse_extracted() -> None:
        path = filedialog.askdirectory(title="选择 CGNS 提取结果目录")
        if path:
            extracted_var.set(path)

    def browse_output() -> None:
        path = filedialog.asksaveasfilename(
            title="保存映射后的 INP",
            defaultextension=".inp",
            filetypes=[("Abaqus INP 文件", "*.inp"), ("所有文件", "*.*")],
        )
        if path:
            output_var.set(path)

    def run_job() -> None:
        try:
            validate_mapping_gui_inputs(
                inp_var.get(),
                extracted_var.get(),
                target_set_var.get(),
            )
            frequencies = parse_frequency_text(frequency_var.get())
            scale, translate, axis_order, axis_sign = read_transform_options()
            num_workers = int(num_workers_var.get().strip() or "1")
            result = run_mapping(
                inp_path=inp_var.get().strip(),
                extracted_dir=extracted_var.get().strip(),
                target_set=target_set_var.get().strip(),
                target_set_type=target_type_var.get().strip(),  # type: ignore[arg-type]
                frequencies=frequencies,
                output_path=output_var.get().strip() or None,
                scale=scale,
                translate=translate,
                axis_order=axis_order,
                axis_sign=axis_sign,
                num_workers=num_workers,
            )
        except Exception as exc:
            message = format_gui_error(exc)
            status_var.set(f"错误：{message}")
            messagebox.showerror("映射失败", message)
            return
        status_var.set(f"已写入 {result.output_inp_path}")
        messagebox.showinfo("完成", f"已写入：\n{result.output_inp_path}")

    def preview_alignment() -> None:
        try:
            validate_mapping_gui_inputs(
                inp_var.get(),
                extracted_var.get(),
                target_set_var.get(),
            )
            scale, translate, axis_order, axis_sign = read_transform_options()
            preview = build_alignment_preview(
                inp_path=inp_var.get().strip(),
                extracted_dir=extracted_var.get().strip(),
                target_set=target_set_var.get().strip(),
                target_set_type=target_type_var.get().strip(),  # type: ignore[arg-type]
                scale=scale,
                translate=translate,
                axis_order=axis_order,
                axis_sign=axis_sign,
            )
        except Exception as exc:
            message = format_gui_error(exc)
            status_var.set(f"错误：{message}")
            messagebox.showerror("预览失败", message)
            return
        status_var.set("已打开坐标对齐预览。")
        show_alignment_window(preview)

    ttk.Label(frame, text="INP 文件").grid(row=0, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=inp_var, width=72).grid(row=0, column=1, sticky="ew", pady=6)
    ttk.Button(frame, text="浏览", command=browse_inp).grid(row=0, column=2, padx=(8, 0))

    ttk.Label(frame, text="CGNS 提取结果目录").grid(row=1, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=extracted_var, width=72).grid(row=1, column=1, sticky="ew", pady=6)
    ttk.Button(frame, text="浏览", command=browse_extracted).grid(row=1, column=2, padx=(8, 0))

    ttk.Label(frame, text="输出 INP").grid(row=2, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=output_var, width=72).grid(row=2, column=1, sticky="ew", pady=6)
    ttk.Button(frame, text="浏览", command=browse_output).grid(row=2, column=2, padx=(8, 0))

    ttk.Label(frame, text="目标集合").grid(row=3, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=target_set_var, width=24).grid(row=3, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="集合类型").grid(row=4, column=0, sticky="w", pady=6)
    ttk.Combobox(
        frame,
        textvariable=target_type_var,
        values=("elset", "nset"),
        state="readonly",
        width=12,
    ).grid(row=4, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="频率 Hz 或范围").grid(row=5, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=frequency_var, width=32).grid(row=5, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="比例系数").grid(row=6, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=scale_var, width=16).grid(row=6, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="平移 dx,dy,dz").grid(row=7, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=translate_var, width=32).grid(row=7, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="轴顺序").grid(row=8, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=axis_order_var, width=16).grid(row=8, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="轴方向").grid(row=9, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=axis_sign_var, width=16).grid(row=9, column=1, sticky="w", pady=6)

    ttk.Label(frame, text="线程数 (1=串行, 0=自动)").grid(row=10, column=0, sticky="w", pady=6)
    ttk.Entry(frame, textvariable=num_workers_var, width=8).grid(row=10, column=1, sticky="w", pady=6)

    actions = ttk.Frame(frame)
    actions.grid(row=11, column=1, sticky="w", pady=(16, 8))
    ttk.Button(actions, text="预览坐标对齐", command=preview_alignment).grid(row=0, column=0)
    ttk.Button(actions, text="开始映射", command=run_job).grid(row=0, column=1, padx=(8, 0))

    ttk.Label(frame, textvariable=status_var, wraplength=700).grid(
        row=12,
        column=0,
        columnspan=3,
        sticky="w",
        pady=(12, 0),
    )

    root.mainloop()
    return 0
