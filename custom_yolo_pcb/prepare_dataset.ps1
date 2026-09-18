param(
    [string]$Archive = "",
    [string]$Destination = "",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Archive)) {
    $Archive = Join-Path (Split-Path -Parent $PSScriptRoot) "release-assets\pcb_yolo_train_val_v1.0.0.zip"
}
if ([string]::IsNullOrWhiteSpace($Destination)) { $Destination = Join-Path $PSScriptRoot "dataset" }
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing. Follow README setup or supply -Python with the interpreter path." }
$extractor = Join-Path $PSScriptRoot "reproducibility\scripts\package\extract_dataset.py"
$checksum = Join-Path $PSScriptRoot "reproducibility\manifests\dataset\pcb_yolo_train_val_v1.0.0.sha256"
& $Python $extractor --archive $Archive --checksum $checksum --destination $Destination
exit $LASTEXITCODE
