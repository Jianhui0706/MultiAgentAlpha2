$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"
$env:UV_LINK_MODE = "copy"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = ".\log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$name = "无研报自主初始化 smoke test"
Remove-Item Env:FACTOR_LIBRARY_SUFFIX -ErrorAction SilentlyContinue
$env:RUN_STORE_PATH = "artifacts/run_store.sqlite"
$env:TRAJECTORY_POOL_PATH = "artifacts/trajectory_pool_smoke_no_report.jsonl"
$consoleLogPath = "$logDir\smoke_no_report_$ts.log"

Write-Host ""
Write-Host "============================================================"
Write-Host "开始运行：$name"
Write-Host "覆盖内容：市场状态 + 数据摘要 + 参数设定 -> H0 初始假设 -> mutation -> crossover"
Write-Host "SQLite 主存储：$env:RUN_STORE_PATH"
Write-Host "轨迹池文件：$env:TRAJECTORY_POOL_PATH"
Write-Host "因子库文件：factor_library/all_factors_library.json"
Write-Host "控制台日志：$consoleLogPath"
Write-Host "============================================================"

$argsList = @(
    "run",
    "python",
    "main.py",
    "--evolution-mode",
    "--evolution-profile", "quantaalpha",
    "--evolution-max-rounds", "3",
    "--evolution-num-initial-hypotheses", "2",
    "--evolution-mutation-tasks-per-round", "1",
    "--evolution-crossover-tasks-per-round", "1",
    "--evolution-mutation-parent-top-n", "4",
    "--evolution-crossover-parent-top-n", "4",
    "--evolution-mutation-top-n", "1",
    "--evolution-crossover-group-size", "2",
    "--evolution-crossover-n-groups", "1",
    "--retry-per-round", "1",
    "--train-sharpe-threshold", "2",
    "--no-report-pdf",
    "--initial-direction", "基于当前加密货币横截面市场状态、价格成交量结构和参数化窗口生成多样化可测试因子假设"
)

Write-Host "执行命令：uv $($argsList -join ' ')"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & uv @argsList 2>&1 | Tee-Object -FilePath $consoleLogPath
    $exitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousErrorActionPreference
}

if ($exitCode -ne 0) {
    throw "$name 运行失败，退出码 $exitCode。请查看日志：$consoleLogPath"
}

Write-Host ""
Write-Host "$name 运行完成。"
Write-Host "建议查看："
Write-Host "- $env:RUN_STORE_PATH"
Write-Host "- $env:TRAJECTORY_POOL_PATH"
Write-Host "- logs/alpha_factor_mining_loop/<run_id>/summary.json"
Write-Host "- $consoleLogPath"
