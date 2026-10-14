@echo off
cd /d "c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master"
"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\.venv\Scripts\python.exe" "c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\analyze_generation.py" > "c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\analyze_generation_output.txt" 2>&1
echo ExitCode:%ERRORLEVEL% >> "c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\analyze_generation_output.txt"
