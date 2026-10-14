$stale = @(
    (Join-Path $PSScriptRoot "data\encoders\atom_bace1_joint_positive_train_chembl_train.pkl"),
    (Join-Path $PSScriptRoot "data\encoders\bond_bace1_joint_positive_train_chembl_train.pkl"),
    (Join-Path $PSScriptRoot "data\decoders\atom_bace1_joint_positive_train_chembl_train.pkl"),
    (Join-Path $PSScriptRoot "data\decoders\bond_bace1_joint_positive_train_chembl_train.pkl"),
    (Join-Path $PSScriptRoot "druggen_data\bace1_joint_positive_train45.pt")
)
foreach ($f in $stale) {
    if (Test-Path $f) {
        Remove-Item $f -Force
        Write-Host "Deleted: $f"
    } else {
        Write-Host "Not found (OK): $f"
    }
}
Write-Host "Cleanup done."
