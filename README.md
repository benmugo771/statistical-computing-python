# BIT4131 Statistical Computing and Reporting — Python

This repository demonstrates all 23 topics in Python. All example datasets are synthetic teaching data.

## Run on Windows PowerShell
```powershell
cd "C:\path\to\BIT4131_Statistical_Computing_Python"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python analysis.py
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Create and publish the GitHub repository
1. Sign in to https://github.com and select **New repository**.
2. Name it `BIT4131-statistical-computing-python`; choose Public or Private according to course instructions.
3. Do not initialise a second README if this folder already contains one.
4. Open this folder in VS Code, then Terminal → New Terminal.
5. Run (replace the URL with the exact URL GitHub provides):
```powershell
git init
git add .
git commit -m "Add BIT4131 statistical computing demonstrations"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/BIT4131-statistical-computing-python.git
git push -u origin main
```
6. Refresh GitHub and verify the files. For later updates:
```powershell
git status
git add .
git commit -m "Update demonstrations"
git push
```
Use `git remote -v` to inspect a configured remote; use `git remote set-url origin <URL>` if it needs changing.

## Project files
- `analysis.py`: executable demonstrations for topics 1–23.
- `requirements.txt`: dependencies.
- `data/`: generated CSV examples.
- `figures/`: generated charts.
- `results.json`: numeric outputs.
- `BIT4131_Statistical_Computing_Group_Documentation.docx`: full report.

The topic brief lists both Python libraries and R packages. As requested, this project implements the work in Python only; R packages are not required. Do not upload confidential data, passwords or API keys. Cite any real external datasets used in a final submission.
