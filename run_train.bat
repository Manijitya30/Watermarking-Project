@echo off
cd /d "%~dp0"
call venv38\Scripts\activate.bat
python train.py
