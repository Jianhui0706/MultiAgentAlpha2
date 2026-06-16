$ts = Get-Date -Format "yyyyMMdd_HHmmss"
uv run .\main.py --loop-count 10 2>&1 | Tee-Object -FilePath ".\log\run_$ts.log"
