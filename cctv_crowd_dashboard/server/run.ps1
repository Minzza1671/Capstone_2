# 포트 8000 기존 프로세스 정리 후 서버 시작
$port = 8001
$pids = netstat -ano | Select-String ":$port\s" | ForEach-Object {
    ($_ -split '\s+')[-1]
} | Sort-Object -Unique | Where-Object { $_ -match '^\d+$' -and $_ -ne '0' }

foreach ($p in $pids) {
    try { taskkill /PID $p /F 2>$null } catch {}
}

Start-Sleep -Seconds 1
python -m uvicorn main:app --reload --port $port
