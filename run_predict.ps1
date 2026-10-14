$python = 'c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\.venv\Scripts\python.exe'
$script = 'c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\predict_joint.py'
$input_csv = 'c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\results\baseline_screen_50000_generated.csv'
$output_csv = 'c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\results\baseline_screen_50000_predictions.csv'

& $python $script --input $input_csv --output $output_csv
