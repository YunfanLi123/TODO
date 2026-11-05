@echo off
cd /d "%~dp0.."
set BETA_AUTH=0
set PUBLIC_ANONYMOUS=1
set REQUIRE_USER_MODEL=1
set MULTI_TENANT=1
"C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8765
