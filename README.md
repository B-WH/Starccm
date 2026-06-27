# STAR-CCM+ CGNS 压力数据处理工具

本仓库包含两个 Python 工具，用于把 STAR-CCM+ 导出的 HDF5-CGNS
表面压力数据提取出来，并进一步映射到 Abaqus INP 结构模型。

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

对于 `*STEADY STATE DYNAMICS` 稳态动力学分析步，映射程序使用：

- `surface_geometry.npz`
- `pressure_complex_spectrum.npz`

对于 `*DYNAMIC` 或 `*DYNAMIC, EXPLICIT` 动力学分析步，映射程序使用：

- `surface_geometry.npz`
- `pressure_time.json.gz`

受载区域必须指定为 Abaqus 中已有的 `Nset` 或 `Elset`。程序不会默认
把压力映射到全部节点，避免把内部节点或非受压区域错误加载。

## 安装依赖

建议使用 Python 3.11+，或使用项目本地虚拟环境。安装依赖：

```powershell
pip install -r requirements.txt
```

`tkinter`、`argparse`、`gzip` 和 `json` 属于 Python 标准库，不写入
`requirements.txt`。

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
- `tests/test_extract_cgns_pressure.py`
- `tests/test_map_cgns_pressure_to_inp.py`
- `.gitignore`
- `README.md`
- `requirements.txt`
