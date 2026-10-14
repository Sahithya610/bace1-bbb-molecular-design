$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=== Running 1A ==="
& $py "$root\scripts\audit_1a_docking_reconciliation.py" 2>&1 | Tee-Object -FilePath "$root\reports\audit_corrections\run_1a.log"

Write-Host "=== Running 1C ==="
& $py "$root\scripts\audit_1c_oracle_model_verification.py" 2>&1 | Tee-Object -FilePath "$root\reports\audit_corrections\run_1c.log"

Write-Host "=== DONE ==="
