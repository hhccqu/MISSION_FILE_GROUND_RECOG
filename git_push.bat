@echo off
echo 🔄 开始Git推送流程...
echo.

echo 📋 检查当前状态...
git status
echo.

echo ➕ 添加所有更改...
git add .
echo.

echo 📝 请输入提交信息 (按Enter使用默认信息):
set /p commit_msg="提交信息: "
if "%commit_msg%"=="" set commit_msg=更新代码

echo.
echo 💾 提交更改: %commit_msg%
git commit -m "%commit_msg%"
echo.

echo 🚀 推送到GitHub...
git push origin master
echo.

if %errorlevel% equ 0 (
    echo ✅ 推送成功！
) else (
    echo ❌ 推送失败，请检查网络连接或权限
)

echo.
echo 按任意键退出...
pause >nul 