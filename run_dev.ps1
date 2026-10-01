$env:DISABLE_AUTH = "1"
Start-Process ".venv\Scripts\python.exe" "-m uvicorn backend.server:app --host 127.0.0.1 --port 8000" -WindowStyle Hidden
Start-Process "cmd.exe" "/c npm run dev" -WorkingDirectory "frontend" -WindowStyle Hidden
