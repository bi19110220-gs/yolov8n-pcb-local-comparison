param([string]$Archive = "")
$ErrorActionPreference = "Stop"
$package = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
& (Join-Path $package 'prepare_dataset.ps1') -Archive $Archive
exit $LASTEXITCODE
