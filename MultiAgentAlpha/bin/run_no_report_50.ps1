$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:UV_LINK_MODE = "copy"
$env:DEBUG_CONSOLE_JSON = "0"
$env:DEBUG_FULL_TRACE = "0"

function Assert-FileExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Hint
    )

    if (-not (Test-Path $Path)) {
        throw "缺少文件：$Path。$Hint"
    }
}

Assert-FileExists ".\data\panel_data.parquet" "请先把 panel_data.parquet 放到 data 文件夹。"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = ".\log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$runKey = "no_report_50_$ts"
$name = "无研报 evolution 50 轮长跑"
Remove-Item Env:FACTOR_LIBRARY_SUFFIX -ErrorAction SilentlyContinue
$env:RUN_STORE_PATH = "artifacts/run_store.sqlite"
$env:TRAJECTORY_POOL_PATH = "artifacts/trajectory_pool_$runKey.jsonl"
$consoleLogPath = "$logDir\$runKey.log"

Write-Host ""
Write-Host "============================================================"
Write-Host "开始运行：$name"
Write-Host "预计阶段：50 个论文 iteration；每个 iteration = mutation phase + crossover phase"
Write-Host "SQLite 主存储：$env:RUN_STORE_PATH"
Write-Host "轨迹池文件：$env:TRAJECTORY_POOL_PATH"
Write-Host "因子库文件：factor_library/all_factors_library.json"
Write-Host "控制台日志：$consoleLogPath"
Write-Host "报告输出：reports/factor_mining_report_<run_id>_<时间>/"
Write-Host "完整 debug 原文：关闭"
Write-Host "============================================================"

$argsList = @(
    "run",
    "python",
    "main.py",
    "--evolution-mode",
    "--evolution-profile", "quantaalpha",
    "--evolution-iterations", "50",
    "--evolution-num-initial-hypotheses", "10",
    "--evolution-mutation-tasks-per-round", "1",
    "--evolution-crossover-tasks-per-round", "1",
    "--evolution-mutation-parent-top-n", "10",
    "--evolution-crossover-parent-top-n", "10",
    "--evolution-mutation-top-n", "1",
    "--evolution-crossover-group-size", "2",
    "--evolution-crossover-n-groups", "1",
    "--evolution-no-parallel",
    "--retry-per-round", "1",
    "--train-sharpe-threshold", "2",
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
Write-Host "- reports/factor_mining_report_<run_id>_<时间>/factor_mining_report.md 或 .pdf"
Write-Host "- $consoleLogPath"
