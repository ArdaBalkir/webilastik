$env:DISABLE_AUTH = "1"
Start-Process ".venv\Scripts\python.exe" "-m uvicorn backend.server:app --host 0.0.0.0 --port 8000"
Start-Process "cmd.exe" "/c npm run dev" -WorkingDirectory "frontend"