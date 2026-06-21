@echo off
setlocal
cd /d "%~dp0"
echo Creating conda environment "qx" ...
conda env create -f environment.yml
if errorlevel 1 (
    echo Environment may already exist. Updating instead ...
    conda env update -f environment.yml --prune
)
echo.
echo Done. Activate with:  conda activate qx
echo Then run:           python -m src.smoke_test --check-llm
