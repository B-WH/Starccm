param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Clean) {
    Write-Host "Clean requested. This script does not delete files automatically."
    Write-Host "Manual cleanup targets, if present:"
    Write-Host "  build\"
    Write-Host "  dist\"
    Write-Host "  extract_cgns_pressure_cli.spec"
    Write-Host "  extract_cgns_pressure_gui.spec"
    Write-Host "  map_cgns_pressure_to_inp_cli.spec"
    Write-Host "  map_cgns_pressure_to_inp_gui.spec"
}

python -B -m PyInstaller `
    --noconfirm `
    --onefile `
    --console `
    --name extract_cgns_pressure_cli `
    --collect-submodules scipy `
    --collect-submodules h5py `
    extract_cgns_pressure.py

python -B -m PyInstaller `
    --noconfirm `
    --onefile `
    --windowed `
    --name extract_cgns_pressure_gui `
    --collect-submodules scipy `
    --collect-submodules h5py `
    extract_cgns_pressure.py

python -B -m PyInstaller `
    --noconfirm `
    --onefile `
    --console `
    --name map_cgns_pressure_to_inp_cli `
    --collect-submodules scipy `
    --collect-submodules h5py `
    map_cgns_pressure_to_inp.py

python -B -m PyInstaller `
    --noconfirm `
    --onefile `
    --windowed `
    --name map_cgns_pressure_to_inp_gui `
    --collect-submodules scipy `
    --collect-submodules h5py `
    map_cgns_pressure_to_inp.py

Write-Host "Wrote dist\extract_cgns_pressure_cli.exe"
Write-Host "Wrote dist\extract_cgns_pressure_gui.exe"
Write-Host "Wrote dist\map_cgns_pressure_to_inp_cli.exe"
Write-Host "Wrote dist\map_cgns_pressure_to_inp_gui.exe"
