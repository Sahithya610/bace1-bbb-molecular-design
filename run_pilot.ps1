$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
Set-Location $root

# ---------------------------------------------------------------
# Step 0: import check
# ---------------------------------------------------------------
Write-Host "=== Import check ===" -ForegroundColor Cyan
& $python run_ca_import_check.py
if ($LASTEXITCODE -ne 0) { Write-Host "IMPORT FAILED"; exit 1 }

# ---------------------------------------------------------------
# Step 1: 500-step training
# ---------------------------------------------------------------
Write-Host ""
Write-Host "=== 500-step training ===" -ForegroundColor Cyan

$trainArgs = @(
    "train_joint_cross_attention.py",
    "--chembl_smi",       "data/chembl_train.smi",
    "--bace1_smi",        "druggen_data/bace1.smi",
    "--labels_csv",       "druggen_data/bace1_continuous_bbb.csv",
    "--et_labels",        "druggen_data/bace1_et_predictions.csv",
    "--init_generator",   "experiments/models/Issue3-ChemistryReadout/joint_generator.pt",
    "--init_discriminator","experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt",
    "--output_dir",       "experiments/models/Issue4-CrossAttention-Pilot500",
    "--log_file",         "results/cross_attention_pilot/training_log.txt",
    "--batch_size",       "32",
    "--max_steps",        "500",
    "--epochs",           "1",
    "--seed",             "1",
    "--lambda_gp",        "10.0",
    "--lambda_aux",       "1.0",
    "--lambda_condition", "5.0"
)

& $python @trainArgs
if ($LASTEXITCODE -ne 0) { Write-Host "TRAINING FAILED"; exit 1 }

# ---------------------------------------------------------------
# Step 2: Generate 1024 molecules @ pIC50=5.5
# ---------------------------------------------------------------
Write-Host ""
Write-Host "=== Generate cond55 ===" -ForegroundColor Cyan

$gen55Args = @(
    "generate_ca_molecules.py",
    "--checkpoint",    "experiments/models/Issue4-CrossAttention-Pilot500/joint_generator.pt",
    "--batch_size",    "128",
    "--num_batches",   "8",
    "--desired_pIC50", "5.5",
    "--desired_logBB", "0.0",
    "--output",        "results/cross_attention_pilot/gen_cond55.csv",
    "--seed",          "42"
)

& $python @gen55Args
if ($LASTEXITCODE -ne 0) { Write-Host "GEN cond55 FAILED"; exit 1 }

# ---------------------------------------------------------------
# Step 3: Generate 1024 molecules @ pIC50=8.0
# ---------------------------------------------------------------
Write-Host ""
Write-Host "=== Generate cond80 ===" -ForegroundColor Cyan

$gen80Args = @(
    "generate_ca_molecules.py",
    "--checkpoint",    "experiments/models/Issue4-CrossAttention-Pilot500/joint_generator.pt",
    "--batch_size",    "128",
    "--num_batches",   "8",
    "--desired_pIC50", "8.0",
    "--desired_logBB", "0.0",
    "--output",        "results/cross_attention_pilot/gen_cond80.csv",
    "--seed",          "42"
)

& $python @gen80Args
if ($LASTEXITCODE -ne 0) { Write-Host "GEN cond80 FAILED"; exit 1 }

# ---------------------------------------------------------------
# Step 4: Predict on cond55
# ---------------------------------------------------------------
Write-Host ""
Write-Host "=== Predict cond55 ===" -ForegroundColor Cyan

$pred55Args = @(
    "predict_joint.py",
    "--input",  "results/cross_attention_pilot/gen_cond55.csv",
    "--output", "results/cross_attention_pilot/pred_cond55.csv"
)
& $python @pred55Args
if ($LASTEXITCODE -ne 0) { Write-Host "PREDICT cond55 FAILED"; exit 1 }

# ---------------------------------------------------------------
# Step 5: Predict on cond80
# ---------------------------------------------------------------
Write-Host ""
Write-Host "=== Predict cond80 ===" -ForegroundColor Cyan

$pred80Args = @(
    "predict_joint.py",
    "--input",  "results/cross_attention_pilot/gen_cond80.csv",
    "--output", "results/cross_attention_pilot/pred_cond80.csv"
)
& $python @pred80Args
if ($LASTEXITCODE -ne 0) { Write-Host "PREDICT cond80 FAILED"; exit 1 }

# ---------------------------------------------------------------
# Step 6: Compute metrics and save results
# ---------------------------------------------------------------
Write-Host ""
Write-Host "=== Compute metrics ===" -ForegroundColor Cyan

& $python compute_ca_pilot_results.py
if ($LASTEXITCODE -ne 0) { Write-Host "METRICS FAILED"; exit 1 }

Write-Host ""
Write-Host "=== All steps complete ===" -ForegroundColor Green
