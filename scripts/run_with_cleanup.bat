@echo off
setlocal enabledelayedexpansion

:: 获取脚本所在目录的父目录（项目根目录）
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo ========================================
echo   FinRAG-Advisor 启动脚本
echo ========================================
echo.
echo 项目目录: %PROJECT_ROOT%
echo.

:: 检查环境变量
if not exist ".env" (
    echo [警告] 未找到 .env 文件，使用默认配置
    echo.
)

:: 启动 Elasticsearch 和 Ollama（如果未运行）
echo [1/3] 检查依赖服务...
tasklist | findstr /i "ollama.exe" >nul 2>&1
if %errorlevel% neq 0 (
    echo       - Ollama 未运行，请先启动 Ollama
    echo       - 或者让系统自动检测
)

echo [2/3] 启动 Streamlit 应用...
echo       - 访问地址: http://localhost:8501
echo       - 按 Ctrl+C 停止
echo.

:: 启动 Streamlit（后台运行）
start /B streamlit run src\streamlit_app.py --server.port 8501 --server.headless true

:: 保存 Streamlit 进程 ID（用于后续清理）
set "STREAMLIT_PID=%errorlevel%"

echo [3/3] 应用已启动！
echo.
echo ========================================
echo   按任意键停止应用并清理资源...
echo ========================================
pause >nul

:: 清理流程
echo.
echo ========================================
echo   正在清理资源...
echo ========================================

:: 1. 停止 Streamlit
echo   [*] 停止 Streamlit...
taskkill /F /IM streamlit.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo       ✓ Streamlit 已停止
) else (
    echo       - Streamlit 未运行或已停止
)

:: 2. 清理 Python 进程（但不清理本脚本）
echo   [*] 清理 Python 进程...
for /f "tokens=2" %%i in ('tasklist ^| findstr /i "python.exe"') do (
    if %%i neq %STREAMLIT_PID% (
        taskkill /F /PID %%i >nul 2>&1
    )
)
echo       ✓ Python 进程已清理

:: 3. 清理 Ollama（可选，保留注释）
:: echo   [*] 清理 Ollama...
:: taskkill /F /IM ollama.exe >nul 2>&1
:: if %errorlevel% equ 0 (
::     echo       ✓ Ollama 已停止
:: ) else (
::     echo       - Ollama 未运行或已停止
:: )

echo.
echo ========================================
echo   清理完成！
echo ========================================
echo.
pause
endlocal
