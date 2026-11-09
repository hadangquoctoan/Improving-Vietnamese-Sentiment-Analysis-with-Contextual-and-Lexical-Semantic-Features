@echo off
rem Nhap dup de chay demo cho ca mang LAN. Cua so in dia chi http://<IP-may-nay>:8000 cho thiet bi khac.
title Sentiment Lens - demo (mang LAN)
powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0run_api.ps1" -Lan
