# Creates .lnk shortcuts with Windows system icons in this folder.
# Run once: right-click -> "Run with PowerShell"  OR  powershell -File create_shortcuts.ps1

$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$shell32 = "$env:SystemRoot\System32\shell32.dll"
$imares  = "$env:SystemRoot\System32\imageres.dll"
$wsh     = New-Object -ComObject WScript.Shell

function New-Shortcut {
    param($Name, $Target, $IconDll, $IconIdx, $Desc)
    $lnk = $wsh.CreateShortcut("$root\$Name.lnk")
    $lnk.TargetPath       = $Target
    $lnk.WorkingDirectory = $root
    $lnk.IconLocation     = "$IconDll,$IconIdx"
    $lnk.Description      = $Desc
    $lnk.Save()
    Write-Host "  Created: $Name.lnk" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Creating shortcuts in:" -ForegroundColor Green
Write-Host "  $root"
Write-Host ""

# ── Start Agent ────────────────────────────────────────────────────────────────
# shell32.dll index 137  =  green arrow / forward-facing icon
New-Shortcut `
    -Name     "Start Agent" `
    -Target   "$root\start.bat" `
    -IconDll  $shell32 `
    -IconIdx  137 `
    -Desc     "Start S7 RAG Agent (Gateway + Streamlit dashboard)"

# ── Stop Agent ─────────────────────────────────────────────────────────────────
# shell32.dll index 131  =  red X / stop icon
New-Shortcut `
    -Name     "Stop Agent" `
    -Target   "$root\stop.bat" `
    -IconDll  $shell32 `
    -IconIdx  131 `
    -Desc     "Stop S7 RAG Agent (kills Gateway and Streamlit)"

# ── Open Dashboard (browser) ───────────────────────────────────────────────────
# shell32.dll index 220  =  globe / browser-like icon
$lnk = $wsh.CreateShortcut("$root\Open Dashboard.lnk")
$lnk.TargetPath       = "http://localhost:8501"
$lnk.WorkingDirectory = $root
$lnk.IconLocation     = "$shell32,220"
$lnk.Description      = "Open Streamlit dashboard in browser (Gateway must be running)"
$lnk.Save()
Write-Host "  Created: Open Dashboard.lnk" -ForegroundColor Cyan

# ── Open Gateway API ────────────────────────────────────────────────────────────
# shell32.dll index 14  =  terminal / console icon
$lnk = $wsh.CreateShortcut("$root\Open Gateway API.lnk")
$lnk.TargetPath       = "http://localhost:8107"
$lnk.WorkingDirectory = $root
$lnk.IconLocation     = "$shell32,14"
$lnk.Description      = "Open Gateway V7 API dashboard in browser"
$lnk.Save()
Write-Host "  Created: Open Gateway API.lnk" -ForegroundColor Cyan

Write-Host ""
Write-Host "Done! You now have 4 shortcuts in the folder." -ForegroundColor Green
Write-Host "Double-click 'Start Agent' to launch the system." -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to close"
