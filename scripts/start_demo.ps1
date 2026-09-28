param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "Samsung Code Retrieval demo startup check" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

if (-not (Test-Path $Python)) {
    throw "Virtual environment Python not found at $Python. Create it with: py -3.11 -m venv .venv"
}

$llamaUrl = $env:JINA_EMBEDDING_SERVER_URL
if (-not $llamaUrl) { $llamaUrl = "http://127.0.0.1:8081" }

try {
    $health = Invoke-WebRequest -Uri "$llamaUrl/health" -UseBasicParsing -TimeoutSec 3
    Write-Host "Embedding server healthy: $llamaUrl ($($health.StatusCode))" -ForegroundColor Green
} catch {
    Write-Warning "Embedding server is not reachable at $llamaUrl"
    Write-Host "Start llama.cpp embedding server first, then rerun this script."
}

$apiUrl = "http://${ApiHost}:${ApiPort}"
$apiRunning = $false
try {
    $apiHealth = Invoke-WebRequest -Uri "$apiUrl/health" -UseBasicParsing -TimeoutSec 3
    $apiRunning = $true
    Write-Host "FastAPI already running: $apiUrl ($($apiHealth.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "Starting FastAPI on $apiUrl"
}

if (-not $apiRunning) {
    $logDir = Join-Path $ProjectRoot "data\api\demo_logs"
    New-Item -ItemType Directory -Force $logDir | Out-Null
    Start-Process -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "src.api.app:app", "--host", $ApiHost, "--port", "$ApiPort") `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput (Join-Path $logDir "uvicorn.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "uvicorn.stderr.log") `
        -WindowStyle Hidden
}

if (-not $SkipFrontend) {
    $FrontendDir = Join-Path $ProjectRoot "frontend"
    if (Test-Path (Join-Path $FrontendDir "package.json")) {
        Write-Host "Frontend command:" -ForegroundColor Cyan
        Write-Host "  cd frontend; npm install; npm run dev -- --port $FrontendPort"
        Start-Process -FilePath "powershell" `
            -ArgumentList @("-NoExit", "-Command", "cd '$FrontendDir'; npm run dev -- --port $FrontendPort") `
            -WindowStyle Normal
    } else {
        Write-Warning "frontend/package.json not found; skipping frontend startup."
    }
}

Write-Host "API:      $apiUrl" -ForegroundColor Cyan
Write-Host "Frontend: http://127.0.0.1:$FrontendPort" -ForegroundColor Cyan

