Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python smoke_test_joint_positive.py > logs\smoke_out.txt 2>&1
$ec = $LASTEXITCODE
Write-Output "EXITCODE:$ec"
