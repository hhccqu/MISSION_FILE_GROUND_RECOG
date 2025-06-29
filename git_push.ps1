# Git推送脚本 - PowerShell版本
Write-Host "🔄 开始Git推送流程..." -ForegroundColor Cyan
Write-Host ""

# 检查当前状态
Write-Host "📋 检查当前状态..." -ForegroundColor Yellow
git status
Write-Host ""

# 添加所有更改
Write-Host "➕ 添加所有更改..." -ForegroundColor Green
git add .
Write-Host ""

# 获取提交信息
Write-Host "📝 请输入提交信息 (按Enter使用默认信息):" -ForegroundColor Magenta
$commit_msg = Read-Host "提交信息"
if ([string]::IsNullOrEmpty($commit_msg)) {
    $commit_msg = "更新代码 - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}

# 提交更改
Write-Host ""
Write-Host "💾 提交更改: $commit_msg" -ForegroundColor Blue
git commit -m "$commit_msg"
Write-Host ""

# 推送到GitHub
Write-Host "🚀 推送到GitHub..." -ForegroundColor Red
$pushResult = git push origin master 2>&1
Write-Host ""

# 检查推送结果
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 推送成功！" -ForegroundColor Green
    Write-Host "🌐 查看你的仓库: https://github.com/hhccqu/MISSION_FILE_GROUND_RECOG" -ForegroundColor Blue
} else {
    Write-Host "❌ 推送失败，请检查网络连接或权限" -ForegroundColor Red
    Write-Host "错误信息: $pushResult" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 