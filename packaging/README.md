# Windows EXE Packaging

Build the two command-line tools on the same Windows architecture as the target
machine.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
.\packaging\build_exe.ps1
```

The executables are written to `dist/`:

- `dist/extract_cgns_pressure_cli.exe`
- `dist/extract_cgns_pressure_gui.exe`
- `dist/map_cgns_pressure_to_inp_cli.exe`
- `dist/map_cgns_pressure_to_inp_gui.exe`

Use the CLI executables for batch processing. Use the GUI executables when
double-click operation is preferred.

For the 2000-step frequency workflow with `dt = 0.0005 s`, extract 1 Hz bins up
to the 1000 Hz Nyquist limit:

```powershell
.\dist\extract_cgns_pressure_cli.exe "data\*.cgns" `
  --dt 0.0005 `
  --output-dir cgns_pressure_output `
  --pressure-name Pressure `
  --skip-legacy-json `
  --export-all-data
```

Map the full 1-800 Hz frequency range:

```powershell
.\dist\map_cgns_pressure_to_inp_cli.exe `
  --inp model.inp `
  --extracted cgns_pressure_output `
  --target-set SURFACE `
  --target-set-type elset `
  --frequency-range 1:800:1 `
  --output model_mapped.inp
```

Use a fresh virtual environment for packaging. PyInstaller bundles Python,
NumPy, SciPy, h5py, and Tkinter from the build machine.
