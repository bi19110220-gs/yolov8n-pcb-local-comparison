# Compatibility wrapper; training routes through the beginner entry point.
param([string]$OutputDir = "", [switch]$ResumeEnhancedOnly)
$ErrorActionPreference = "Stop"
$package = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $package ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Follow the package README to create .venv." }
$arguments = @((Join-Path $package 'train_local.py'))
if ($OutputDir) { $arguments += @('--output-dir', $OutputDir) }
if ($ResumeEnhancedOnly) { $arguments += '--resume-enhanced-only' }
& $pythonExe @arguments
exit $LASTEXITCODE
