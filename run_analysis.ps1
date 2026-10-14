Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python analyze_smoke_sample.py > logs\smoke_analysis.txt 2>&1
Write-Output "EXITCODE:$LASTEXITCODE"
