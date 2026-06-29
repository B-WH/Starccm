# Windows EXE 打包说明

请在与目标机器相同架构的 Windows 环境中打包这两个工具。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install .[speed]
pip install pyinstaller
.\packaging\build_exe.ps1
```

可执行文件会写入 `dist/`：

- `dist/extract_cgns_pressure_cli.exe`
- `dist/extract_cgns_pressure_gui.exe`
- `dist/map_cgns_pressure_to_inp_cli.exe`
- `dist/map_cgns_pressure_to_inp_gui.exe`

需要批处理时使用 CLI 版本；希望双击操作时使用图形界面版本。

对于 `dt = 0.0005 s`、2000 个时间步的频率流程，频率分辨率为 1 Hz，
奈奎斯特频率上限为 1000 Hz，可按下面命令提取：

```powershell
.\dist\extract_cgns_pressure_cli.exe "data\*.cgns" `
  --dt 0.0005 `
  --output-dir cgns_pressure_output `
  --pressure-name Pressure `
  --skip-legacy-json `
  --export-all-data
```

映射完整的 1-800 Hz 频率范围（4 线程并行）：

```powershell
.\dist\map_cgns_pressure_to_inp_cli.exe `
  --inp model.inp `
  --extracted cgns_pressure_output `
  --target-set SURFACE `
  --target-set-type elset `
  --frequency-range 1:800:1 `
  --output model_mapped.inp `
  --num-workers 4
```

``--num-workers 0`` 可自动根据 CPU 核心数选择线程数；默认 ``1`` 为串行。

建议使用新的虚拟环境打包。`.[speed]` 额外依赖会安装 SciPy，使 PyInstaller
能够一并打包 cKDTree 加速路径。PyInstaller 也会从打包机器上收集 Python、
NumPy、h5py 和 Tkinter。