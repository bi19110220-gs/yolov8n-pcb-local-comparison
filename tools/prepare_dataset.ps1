param(
    [string]$Archive = "release-assets\pcb_yolo_train_val_v1.0.0.zip"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Repository Python environment missing. Follow README Installation to create .venv."
}
Set-Location -LiteralPath $projectRoot
& $pythonExe -m scripts.package.extract_dataset --archive $Archive --checksum "manifests\dataset\pcb_yolo_train_val_v1.0.0.sha256" --destination "dataset"
exit $LASTEXITCODE
