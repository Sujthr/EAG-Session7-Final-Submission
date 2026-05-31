# start.ps1 - S7 RAG Agent launcher (Windows PowerShell 5.1 compatible)
# Opens: LLM Gateway V7 window, Streamlit Dashboard window, Query Runner window.
# Invoked by start.bat (double-click) or:  powershell -File start.ps1

$root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$gwDir    = Join-Path $root "Gateway\llm_gatewayV7"
$agentDir = Join-Path $root "S7Code\S7code"
$envFile  = Join-Path $root "Gateway\.env"

function Pause-And-Exit($code) {
    Write-Host ""
    Read-Host "Press Enter to close"
    exit $code
}

trap {
    Write-Host ""
    Write-Host "[FATAL] $_" -ForegroundColor Red
    Pause-And-Exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  S7 RAG Agent  |  OpenAI Edition"         -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# ── Pre-flight checks ──────────────────────────────────────────────────────────
if (-not (Test-Path $envFile)) {
    Write-Host "[ERROR] Gateway\.env not found." -ForegroundColor Red
    Write-Host "  Create it:  copy `"$root\Gateway\.env.example`" `"$envFile`""
    Write-Host "  Then set:   OPENAI_API_KEY=sk-..."
    Pause-And-Exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] uv not found. Install:  pip install uv" -ForegroundColor Red
    Pause-And-Exit 1
}

$hasCurl = [bool](Get-Command curl.exe -ErrorAction SilentlyContinue)
if (-not $hasCurl) {
    Write-Host "[WARN] curl.exe not found - will use basic TCP check instead" -ForegroundColor Yellow
}

# ── Kill stale processes on 8107 and 8501 ─────────────────────────────────────
foreach ($port in @("8107","8501")) {
    (netstat -ano 2>$null) | Where-Object { $_ -match ":$port\s" } | ForEach-Object {
        $pid_ = ($_.Trim() -split "\s+")[-1]
        if ($pid_ -match "^\d+$" -and $pid_ -ne "0") {
            Write-Host "  Stopping stale process on port $port  [PID $pid_]" -ForegroundColor DarkYellow
            Stop-Process -Id ([int]$pid_) -Force -ErrorAction SilentlyContinue
        }
    }
}
Start-Sleep -Seconds 1

# ── [1/3] Gateway ──────────────────────────────────────────────────────────────
Write-Host "[1/3] Starting LLM Gateway V7 on port 8107 ..." -ForegroundColor Cyan
Start-Process cmd.exe `
    -WorkingDirectory $gwDir `
    -ArgumentList "/k", "title LLM Gateway V7 && uv run main.py" `
    -WindowStyle Normal

Write-Host "      Waiting for Gateway to respond" -NoNewline
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline

    if ($hasCurl) {
        $null = & curl.exe -sf "http://localhost:8107/v1/status" 2>$null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    } else {
        $conn = Test-NetConnection -ComputerName 127.0.0.1 -Port 8107 -InformationLevel Quiet -WarningAction SilentlyContinue 2>$null
        if ($conn) { $ready = $true; break }
    }
}
Write-Host ""

if (-not $ready) {
    Write-Host "[ERROR] Gateway did not respond after 90s." -ForegroundColor Red
    Write-Host "  Check the 'LLM Gateway V7' window for errors."
    Pause-And-Exit 1
}
Write-Host "[OK]  Gateway ready  http://localhost:8107" -ForegroundColor Green

# ── [2/3] Streamlit ────────────────────────────────────────────────────────────
Write-Host "[2/3] Starting Streamlit dashboard ..." -ForegroundColor Cyan
Start-Process cmd.exe `
    -WorkingDirectory $agentDir `
    -ArgumentList "/k", "title Streamlit Dashboard && uv run streamlit run app.py" `
    -WindowStyle Normal
Start-Sleep -Seconds 6
Start-Process "http://localhost:8501"
Write-Host "[OK]  Dashboard  http://localhost:8501" -ForegroundColor Green

# ── [3/3] Query runner ─────────────────────────────────────────────────────────
Write-Host "[3/3] Launching all benchmark queries (A-H + Q1-Q5) ..." -ForegroundColor Cyan
Start-Process cmd.exe `
    -WorkingDirectory $agentDir `
    -ArgumentList "/k", "title Query Runner && uv run python run_all_queries.py" `
    -WindowStyle Normal
Write-Host "[OK]  Query Runner window opened" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  ALL SYSTEMS RUNNING"                      -ForegroundColor Green
Write-Host "   Gateway   : http://localhost:8107"       -ForegroundColor White
Write-Host "   Dashboard : http://localhost:8501"       -ForegroundColor White
Write-Host "   Queries   : see the Query Runner window" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close this launcher"
