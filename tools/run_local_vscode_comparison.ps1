param(
    [string]$OutputDir = "",
    [switch]$ResumeEnhancedOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Repository Python environment missing. Follow README Installation to create .venv."
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $projectRoot "results\local_vscode_comparison_$stamp"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $projectRoot $OutputDir
}

& $pythonExe -c "import sys, torch, ultralytics; assert sys.version_info[:2] == (3, 11), sys.version; assert torch.__version__ == '2.11.0+cu128', torch.__version__; assert torch.cuda.is_available(), f'CUDA unavailable in {sys.executable}'; assert ultralytics.__version__ == '8.4.84', ultralytics.__version__; print(sys.executable); print(torch.cuda.get_device_name(0)); print(ultralytics.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA/Ultralytics preflight failed."
}

Write-Host "OUTPUT_DIR=$OutputDir"
Set-Location -LiteralPath $projectRoot
if ($ResumeEnhancedOnly) {
    & $pythonExe -u -m "tools.run_local_vscode_comparison" --output-dir $OutputDir --resume-enhanced-only
} else {
    & $pythonExe -u -m "tools.run_local_vscode_comparison" --output-dir $OutputDir
}
exit $LASTEXITCODE
