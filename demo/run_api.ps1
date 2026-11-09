# Start the demo (API + frontend) on port 8000. Docs at /docs.
#   run_api.ps1          -> only this machine:      http://127.0.0.1:8000
#   run_api.ps1 -Lan     -> other devices on the same LAN can open http://<this-PC-IP>:8000
# Paths are resolved in this order (first that exists wins), so it runs on any team machine:
#   Java:     $env:DEMO_JAVA_HOME -> $env:JAVA_HOME -> D:\tools\jdk21
#   Python:   $env:DEMO_PYTHON    -> D:\envs\nlpdevelop\Scripts\python.exe -> python on PATH
#   Frontend: $env:DEMO_WEB_FILE  -> demo\web\index.html in this repo
param(
    [switch]$Lan,
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$env:JAVA_HOME = if ($env:DEMO_JAVA_HOME) { $env:DEMO_JAVA_HOME } elseif ($env:JAVA_HOME) { $env:JAVA_HOME } else { "D:\tools\jdk21" }
$env:PATH = "$env:JAVA_HOME\bin;$env:JAVA_HOME\bin\server;$env:PATH"
$Python = if ($env:DEMO_PYTHON) { $env:DEMO_PYTHON }
          elseif (Test-Path "D:\envs\nlpdevelop\Scripts\python.exe") { "D:\envs\nlpdevelop\Scripts\python.exe" }
          else { "python" }
# Frontend page served at / (same origin as the API, so LAN clients need no CORS).
if (-not $env:DEMO_WEB_FILE) { $env:DEMO_WEB_FILE = Join-Path $Root "demo\web\index.html" }

if (-not (Test-Path "$env:JAVA_HOME\bin\server\jvm.dll")) { throw "Khong tim thay JDK o $env:JAVA_HOME. Cai JDK 17+ roi dat JAVA_HOME (hoac DEMO_JAVA_HOME)." }
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "Khong tim thay Python ($Python). Kich hoat moi truong hoac dat DEMO_PYTHON." }
foreach ($need in @("artifacts\checkpoints\best_fused.pt", "artifacts\hf-cache\phobert-base\pytorch_model.bin",
                    "artifacts\tools\vncorenlp\VnCoreNLP-1.2.jar", "data\lexicons\composite_12d\v0.2.0\composite_lexicon.json")) {
    if (-not (Test-Path (Join-Path $Root $need))) { throw "Thieu $need. Tai tu Drive cua nhom theo demo\README.md (muc Cai dat)." }
}
if (-not (Test-Path $env:DEMO_WEB_FILE)) { Write-Warning "Frontend not found at $env:DEMO_WEB_FILE; API only (set DEMO_WEB_FILE)" }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port dang bi chiem (co the demo da chay o cua so khac). Dong cua so do hoac chay voi -Port 8001."
}

$BindHost = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
Write-Host ""
Write-Host "Sentiment Lens dang khoi dong (nap model mat khoang 10-20 giay)..." -ForegroundColor Cyan
Write-Host "  May nay:   http://127.0.0.1:$Port" -ForegroundColor Green
if ($Lan) {
    $ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" -and $_.InterfaceAlias -notlike "vEthernet*" -and $_.InterfaceAlias -notlike "*Loopback*" } |
        Select-Object -ExpandProperty IPAddress
    foreach ($ip in $ips) { Write-Host "  Mang LAN: http://${ip}:$Port" -ForegroundColor Green }
    Write-Host "  Neu may khac khong vao duoc: mo cong $Port trong Windows Firewall (xem demo\README.md)." -ForegroundColor Yellow
    Write-Host "  Chi dung trong mang tin cay; khong co dang nhap, ai cung mang deu dung duoc." -ForegroundColor Yellow
}
Write-Host "  Tat demo: nhan Ctrl+C hoac dong cua so nay." -ForegroundColor Cyan
Write-Host ""

Set-Location $Root
& $Python -m uvicorn demo.api.app:app --host $BindHost --port $Port --workers 1
