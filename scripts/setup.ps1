$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m venv .venv
$pythonExe = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Install Ollama for Windows, reopen PowerShell, then rerun this setup script."
}
ollama pull qwen2.5:7b
& $pythonExe -m src.chunking
& $pythonExe -m src.indexing
Write-Host "Setup complete. Start the app with: .\.venv\Scripts\python.exe scripts\run_local.py"
