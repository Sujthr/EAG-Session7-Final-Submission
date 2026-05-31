# start.ps1 — S7 RAG Agent launcher (OpenAI edition)
# Starts Gateway, Streamlit, then runs all benchmark queries in separate windows.
# Run via:  start.bat  (double-click)  OR  powershell -File start.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$gwDir    = Join-Path $root "Gateway\llm_gatewayV7"
$agentDir = Join-Path $root "S7Code\S7code"
$envFile  = Join-Path $root "Gateway\.env"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  S7 RAG Agent  |  OpenAI Edition" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# ── Pre-flight checks ──────────────────────────────────────────────────────────
if (-not (Test-Path $envFile)) {
    Write-Host "[ERROR] Gateway\.env not found." -ForegroundColor Red
    Write-Host "  Run:  copy `"$root\Gateway\.env.example`" `"$envFile`""
    Write-Host "  Then set OPENAI_API_KEY=sk-... inside it."
    Read-Host "Press Enter to exit"
    exit 1
}

$uvPath = (Get-Command uv -ErrorAction SilentlyContinue)?.Source
if (-not $uvPath) {
    Write-Host "[ERROR] uv not found. Install it:" -ForegroundColor Red
    Write-Host "  pip install uv"
    Read-Host "Press Enter to exit"
    exit 1
}

# ── Kill any stale process on 8107 ────────────────────────────────────────────
$stale = netstat -ano 2>$null | Select-String ":8107\s" | ForEach-Object {
    ($_ -split "\s+")[-1]
} | Select-Object -Unique | Where-Object { $_ -match "^\d+$" }
foreach ($pid in $stale) {
    Write-Host "  Clearing stale process on 8107 [PID $pid]" -ForegroundColor DarkYellow
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
}

# ── [1/3] Gateway ──────────────────────────────────────────────────────────────
Write-Host "[1/3] Starting LLM Gateway V7 on port 8107 ..." -ForegroundColor Cyan
Start-Process powershell `
    -WorkingDirectory $gwDir `
    -ArgumentList "-NoExit", "-Command", "& { `$host.UI.RawUI.WindowTitle = 'LLM Gateway V7'; uv run main.py }" `
    -WindowStyle Normal

Write-Host "  Waiting for Gateway ..." -NoNewline
$tries = 0
$ready = $false
while ($tries -lt 60) {
    Start-Sleep -Seconds 1
    $tries++
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8107/v1/status" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Write-Host "." -NoNewline
}
Write-Host ""
if (-not $ready) {
    Write-Host "[ERROR] Gateway didn't respond after 60s. Check the Gateway window." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Gateway ready at http://localhost:8107" -ForegroundColor Green

# ── [2/3] Streamlit ────────────────────────────────────────────────────────────
Write-Host "[2/3] Starting Streamlit dashboard on port 8501 ..." -ForegroundColor Cyan
Start-Process powershell `
    -WorkingDirectory $agentDir `
    -ArgumentList "-NoExit", "-Command", "& { `$host.UI.RawUI.WindowTitle = 'Streamlit Dashboard'; uv run streamlit run app.py }" `
    -WindowStyle Normal
Start-Sleep -Seconds 5
Start-Process "http://localhost:8501"
Write-Host "[OK] Dashboard opening at http://localhost:8501" -ForegroundColor Green

# ── [3/3] Query runner ─────────────────────────────────────────────────────────
Write-Host "[3/3] Launching all benchmark queries (A-H + Q1-Q5) ..." -ForegroundColor Cyan
Start-Process powershell `
    -WorkingDirectory $agentDir `
    -ArgumentList "-NoExit", "-Command", "& { `$host.UI.RawUI.WindowTitle = 'Query Runner'; uv run python run_all_queries.py }" `
    -WindowStyle Normal
Write-Host "[OK] Query Runner window opened" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  ALL SYSTEMS RUNNING" -ForegroundColor Green
Write-Host "   Gateway   : http://localhost:8107" -ForegroundColor White
Write-Host "   Dashboard : http://localhost:8501" -ForegroundColor White
Write-Host "   Queries   : watch the Query Runner window" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
