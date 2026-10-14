@echo off
cd /d c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master
c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\.venv\Scripts\python.exe results\compute_50k_screening_stats.py > results\_stats_out.txt 2> results\_stats_err.txt
echo ExitCode=%ERRORLEVEL%
