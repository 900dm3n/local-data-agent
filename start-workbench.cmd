@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing local Python environment. Follow README.md to install dependencies.
  exit /b 1
)
".venv\Scripts\python.exe" -m langchain_data_agent serve %*
