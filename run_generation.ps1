Set-Location 'c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master'
$python = 'c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\.venv\Scripts\python.exe'
$script = 'c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\generate_joint_molecules.py'
& $python $script `
    --checkpoint 'experiments/models/Issue3-ChemistryReadout/joint_generator.pt' `
    --batch_size 128 `
    --num_batches 391 `
    --desired_pIC50 7.0 `
    --desired_logBB 0.0 `
    --seed 42 `
    --output 'results/baseline_screen_50000_generated.csv'
exit $LASTEXITCODE
