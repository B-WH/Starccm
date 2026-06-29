param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Clean) {
    if (Test-Path -LiteralPath "build") {
        Write-Host "Removing build directory..."
        Remove-Item -LiteralPath "build" -Recurse -Force
    }
    if (Test-Path -LiteralPath "dist") {
        Write-Host "Removing dist directory..."
        Remove-Item -LiteralPath "dist" -Recurse -Force
    }
    $specFiles = Get-ChildItem -Path . -Filter "*.spec" -ErrorAction SilentlyContinue
    foreach ($spec in $specFiles) {
        Write-Host "Removing $($spec.Name)..."
        Remove-Item -LiteralPath $spec.FullName -Force
    }
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
