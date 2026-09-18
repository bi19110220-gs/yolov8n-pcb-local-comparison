$ErrorActionPreference = "Stop"
$package = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $package ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Follow the package README to create .venv." }
& $pythonExe -m pytest (Join-Path $package 'reproducibility/tests') -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonExe (Join-Path $package 'reproducibility/scripts/verify/verify_package.py') --root (Split-Path -Parent $package)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonExe (Join-Path $package 'reproducibility/scripts/verify/verify_clone.py') --root (Split-Path -Parent $package)
exit $LASTEXITCODE
