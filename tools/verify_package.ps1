$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Repository Python environment missing. Follow README Installation to create .venv."
}
Set-Location -LiteralPath $projectRoot
& $pythonExe -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonExe scripts\verify\verify_package.py --root .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonExe scripts\verify\verify_clone.py --root .
exit $LASTEXITCODE
