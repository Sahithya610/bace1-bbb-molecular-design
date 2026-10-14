$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$proc = Start-Process -FilePath $py -ArgumentList "results\compute_50k_screening_stats.py" -Wait -PassThru -RedirectStandardOutput "results\_stats_out.txt" -RedirectStandardError "results\_stats_err.txt" -NoNewWindow
Write-Host "ExitCode=$($proc.ExitCode)"
