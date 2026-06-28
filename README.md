# STAR-CCM+ CGNS 压力数据处理工具

本仓库包含两个 Python 工具，用于把 STAR-CCM+ 导出的 HDF5-CGNS
表面压力数据提取出来，并进一步映射到 Abaqus INP 结构模型。

命令行入口仍保留在仓库根目录的两个脚本中，核心实现已拆分到
`starccm_pressure/` 包内，便于复用和测试。

## 工具说明

### `extract_cgns_pressure.py`

该脚本用于从 STAR-CCM+ HDF5-CGNS 时间步文件中提取压力数据，可导出
表面几何、复数压力谱和等效力汇总。

不带参数运行时会打开图形界面：

```powershell
python extract_cgns_pressure.py
```

用于后续 INP 映射的命令行示例：

```powershell
python extract_cgns_pressure.py "other_files\data\604@*.cgns" `
  --dt 0.001 `
  --output-dir cgns_pressure_output `
  --pressure-name Pressure `
  --export-all-data `
  --skip-legacy-json
```

主要输出文件：

- `surface_geometry.npz`：CGNS 表面面心、面积向量、法向和连通几何。
- `pressure_complex_spectrum.npz`：全场复数压力谱，用于频域压力映射。
- `pressure_time.json.gz`：时域脉动压力，仅在瞬态或显式动力学载荷映射时使用。
- `equivalent_force_spectrum.csv`：相干积分得到的全局等效力谱。
- `extraction_metadata.json`：提取过程和采样参数元数据。

### `map_cgns_pressure_to_inp.py`

该脚本将已提取的压力数据映射到 Abaqus INP 模型节点，并生成新的 INP
文件和载荷 include 文件。程序不会覆盖原始 INP 文件。

不带参数运行时会打开图形界面：

```powershell
python map_cgns_pressure_to_inp.py
```

频域映射命令行示例：

```powershell
python map_cgns_pressure_to_inp.py `
  --inp model.inp `
  --extracted cgns_pressure_output `
  --target-set SURFACE `
  --target-set-type elset `
  --frequency 100 `
  --output model_mapped.inp
```

若要映射连续频率范围，例如 `dt=0.0005 s`、2000 个时间步对应的 1 Hz 频率分辨率，
可以使用包含端点的范围写法：

```powershell
python map_cgns_pressure_to_inp.py `
  --inp model.inp `
  --extracted cgns_pressure_output `
  --target-set SURFACE `
  --target-set-type elset `
  --frequency-range 1:800:1 `
  --output model_mapped.inp
```

对于 `*STEADY STATE DYNAMICS` 稳态动力学分析步，映射程序使用：

- `surface_geometry.npz`
- `pressure_complex_spectrum.npz`

对于 `*DYNAMIC` 或 `*DYNAMIC, EXPLICIT` 动力学分析步，映射程序使用：

- `surface_geometry.npz`
- `pressure_time.json.gz`

受载区域必须指定为 Abaqus 中已有的 `Nset` 或 `Elset`。程序不会默认
把压力映射到全部节点，避免把内部节点或非受压区域错误加载。

映射程序在目标面的积分点上使用源面心反距离权重插值，并通过形函数积分
生成一致等效节点力；随后用最小范数修正保证目标节点载荷与源 CGNS
压力载荷的全局总力和总力矩一致。安装 `scipy` 后会自动使用
`scipy.spatial.cKDTree` 加速最近邻查询；如果当前环境没有 `scipy`，程序会回退到
NumPy 全量距离扫描。

`mapping_report.json` 会记录本次映射的主要物理假设：

- 压力等效面力方向为 `-pressure * area_vector`。
- 节点力通过线性三角面/四边形面的形函数积分得到。
- 全局总力和总力矩通过最小范数节点力修正保持守恒，并记录修正前后残差。
- 当前适用范围是线性壳/面单元，以及一阶 `C3D4`/`C3D8` 实体外表面。

## 安装依赖

建议使用 Python 3.11+，或使用项目本地虚拟环境。安装依赖：

```powershell
pip install -r requirements.txt
```

`tkinter`、`argparse`、`gzip` 和 `json` 属于 Python 标准库，不写入
`requirements.txt`。

## 打包 EXE

Windows 下可使用 PyInstaller 打包两个命令行工具：

```powershell
pip install -r requirements.txt
pip install pyinstaller
.\packaging\build_exe.ps1
```

输出文件位于 `dist/`：

- `extract_cgns_pressure_cli.exe`：命令行提取工具，适合批处理。
- `extract_cgns_pressure_gui.exe`：双击打开提取 GUI。
- `map_cgns_pressure_to_inp_cli.exe`：命令行映射工具，适合批处理。
- `map_cgns_pressure_to_inp_gui.exe`：双击打开映射 GUI。

映射 GUI 的频率输入框支持单个频率、逗号分隔频率和范围，例如 `1:800:1`。

## 测试

运行当前测试：

```powershell
python -B -m unittest tests.test_extract_cgns_pressure tests.test_map_cgns_pressure_to_inp
```

其中 `-B` 用于避免写入 `.pyc` 文件。在 Windows 环境中，`__pycache__`
文件有时会被锁定，使用 `-B` 可以减少这类缓存权限噪声。

## 仓库文件管理

不要提交原始 CFD/FEA 数据或生成的分析结果。`.gitignore` 已排除常见的大
文件和中间产物，包括：

- `.cgns`
- `.odb`
- `.npz`
- `.json.gz`
- 生成的映射 INP 文件
- 载荷 include 文件
- Python 缓存目录

建议上传到 GitHub 的核心文件包括：

- `extract_cgns_pressure.py`
- `map_cgns_pressure_to_inp.py`
- `starccm_pressure/`
- `tests/test_extract_cgns_pressure.py`
- `tests/test_map_cgns_pressure_to_inp.py`
- `.gitignore`
- `README.md`
- `requirements.txt`
- `pyproject.toml`
