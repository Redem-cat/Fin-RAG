@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   FinRAG-Advisor Daily Startup
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo [1/6] Checking Docker Desktop...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo       [ERROR] Docker is not running. Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)
echo       [OK] Docker is running
echo.

echo [2/6] Checking environment variables...
if not exist ".env" (
    echo       [WARNING] .env file not found. Please check configuration.
) else (
    echo       [OK] .env file exists
)
echo.

echo [3/6] Starting Elasticsearch...
docker start es-langchain >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Elasticsearch started
    echo       - URL: http://localhost:9200
) else (
    echo       [ERROR] Elasticsearch failed to start. Run install.bat first.
)
echo.

echo [4/6] Starting Neo4j (optional)...
docker start neo4j-langchain >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Neo4j started
    echo       - URL: http://localhost:7474
) else (
    echo       [SKIP] Neo4j not installed or failed to start (optional)
)
echo.

echo [5/6] Checking Ollama Embedding service...
tasklist | findstr /i "ollama.exe" >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Ollama is running
) else (
    echo       [WARNING] Ollama not running. Run 'ollama serve' manually.
)
echo.

echo [6/6] Cleaning Python cache...
if exist "src\__pycache__" (
    rmdir /s /q "src\__pycache__" 2>nul
    echo       [OK] Cache cleared
) else (
    echo       [SKIP] No cache to clear
)
echo.

echo ========================================
echo   Starting Streamlit App...
echo ========================================
echo.
start /B cmd /c "set PYTHONWARNINGS=ignore && streamlit run src\streamlit_app.py --server.port 8501 --server.headless true"

echo [OK] App started!
echo.
echo URL: http://localhost:8501
echo Press any key to stop all services...
pause >nul

echo.
echo Stopping services...
docker stop es-langchain >nul 2>&1
docker stop neo4j-langchain >nul 2>&1
taskkill /F /IM streamlit.exe >nul 2>&1
echo [OK] Services stopped
echo.
pause
endlocal
