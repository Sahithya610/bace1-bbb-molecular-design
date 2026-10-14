Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python "train_bace1_surrogate.py" `
    "--input" "druggen_data/bace1_continuous_bbb.csv" `
    "--output_dir" "results/bace1_surrogate_model" `
    "--epochs" "200"
