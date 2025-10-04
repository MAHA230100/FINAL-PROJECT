param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "healthai"
)

$ErrorActionPreference = "Stop"

Write-Host "[Setup] Creating virtual environment..."
& $PythonExe -m venv .venv

Write-Host "[Setup] Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "[Setup] Upgrading pip..."
pip install --upgrade pip

Write-Host "[Setup] Installing requirements from $ProjectRoot/requirements.txt ..."
pip install -r "$ProjectRoot/requirements.txt"

Write-Host "[Setup] Done." 