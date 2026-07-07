#.\packaging\build_exe.ps1
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Clean) {
    Write-Host "已请求清理。此脚本不会自动删除文件。"
    Write-Host "如需清理，请手动检查并处理以下位置："
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

Write-Host "已写入 dist\extract_cgns_pressure_cli.exe"
Write-Host "已写入 dist\extract_cgns_pressure_gui.exe"
Write-Host "已写入 dist\map_cgns_pressure_to_inp_cli.exe"
Write-Host "已写入 dist\map_cgns_pressure_to_inp_gui.exe"
