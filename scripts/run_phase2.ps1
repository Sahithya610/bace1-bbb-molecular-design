$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$script = Join-Path $PSScriptRoot "phase2_master_table.py"
$outfile = Join-Path $PSScriptRoot "phase2_master_out.txt"

& $python $script 2>$null | Tee-Object -FilePath $outfile
