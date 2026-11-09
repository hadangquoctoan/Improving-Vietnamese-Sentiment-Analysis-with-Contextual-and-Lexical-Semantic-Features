@echo off
rem Nhap dup de chay demo tren may nay, roi mo http://127.0.0.1:8000
title Sentiment Lens - demo (may nay)
powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0run_api.ps1"
