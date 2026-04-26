@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo.
echo ===========================================
echo   FinRAG-Advisor Startup Script
echo ===========================================
echo.

echo [0/5] Checking environment config...
set "env_found=0"
if exist ".env" set "env_found=1"
if exist "elastic-start-local\.env" set "env_found=2"

if !env_found! equ 1 (
    echo       [OK] .env found
) else if !env_found! equ 2 (
    echo       [OK] .env found in elastic-start-local
) else (
    echo       [WARN] .env not found, using default config
)
echo.

echo [1/5] Checking Ollama Embedding service...
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | findstr /i "ollama.exe" >nul 2>&1
if !errorlevel! equ 0 (
    echo       [OK] Ollama is running
) else (
    echo       [*] Ollama not running, starting...
    start "" /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
    tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | findstr /i "ollama.exe" >nul 2>&1
    if !errorlevel! equ 0 (
        echo       [OK] Ollama started
    ) else (
        echo       [WARN] Failed to start Ollama. Please install from: https://ollama.com/download
    )
)

echo       [*] Checking Embedding model...
ollama list 2>nul | findstr /i "my-bge-m3" >nul 2>&1
if !errorlevel! neq 0 (
    echo       [WARN] Model my-bge-m3 not found. Run: ollama pull my-bge-m3
) else (
    echo       [OK] my-bge-m3 model ready
)
echo       [INFO] Chat LLM uses DeepSeek API, no Ollama chat model needed
echo.

echo [2/5] Checking Elasticsearch service...

:::: Check if es-langchain container exists
docker ps -a --format "{{.Names}}" 2>nul | findstr /i "^es-langchain$" >nul 2>&1
set "es_exists=!errorlevel!"

if !es_exists! equ 0 (
    :::: Container exists, check if running
    docker ps --format "{{.Names}}" 2>nul | findstr /i "^es-langchain$" >nul 2>&1
    if !errorlevel! equ 0 (
        echo       [OK] es-langchain container is running
    ) else (
        echo       [*] es-langchain container exists but not running, starting...
        docker start es-langchain >nul 2>&1
        if !errorlevel! equ 0 (
            echo       [OK] es-langchain container started
        ) else (
            echo       [WARN] Failed to start es-langchain container
        )
    )
) else (
    :::: es-langchain not found, check if another container is using port 9200
    for /f "tokens=*" %%a in ('docker ps --filter "publish=9200" --format "{{.Names}}" 2^>nul') do set "es_port_container=%%a"

    if defined es_port_container (
        echo       [*] Found container [!es_port_container!] using port 9200, removing...
        docker stop !es_port_container! >nul 2>&1
        docker rm !es_port_container! >nul 2>&1
        echo       [OK] Removed old container [!es_port_container!]
    )

    echo       [*] Creating container es-langchain...
    echo       [*] This may take a few minutes if downloading the image for the first time
    docker run -d --name es-langchain -p 9200:9200 -p 9300:9300 -e discovery.type=single-node -e xpack.security.enabled=false -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" -v es-data:/usr/share/elasticsearch/data docker.elastic.co/elasticsearch/elasticsearch:8.11.0 >nul 2>&1
    if !errorlevel! equ 0 (
        echo       [OK] Container es-langchain created. ES starting in background...
    ) else (
        echo       [WARN] Container creation failed. Is Docker running?
    )
)

:::: Final port check
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:9200' -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    echo       [OK] Elasticsearch responding on localhost:9200
) else (
    echo       [WARN] Elasticsearch not responding on localhost:9200 yet
)
echo.

echo [3/5] Checking Neo4j service (optional)...

powershell -Command "$c = New-Object Net.Sockets.TcpClient; $c.Connect('localhost', 7687); $c.Close(); exit 0" >nul 2>&1
if !errorlevel! equ 0 (
    echo       [OK] Neo4j running at localhost:7687
) else (
    docker ps -a --format "{{.Names}}" 2>nul | findstr /i "^neo4j-langchain$" >nul 2>&1
    set "neo_exists=!errorlevel!"
    if !neo_exists! neq 0 (
        echo       [*] Container neo4j-langchain not found, creating...
        docker run -d --name neo4j-langchain -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]' neo4j:5.15-community >nul 2>&1
        if !errorlevel! equ 0 (
            echo       [OK] Container neo4j-langchain created
        ) else (
            echo       [SKIP] Neo4j creation failed, KG feature unavailable
        )
    ) else (
        docker ps --format "{{.Names}}" 2>nul | findstr /i "^neo4j-langchain$" >nul 2>&1
        set "neo_running=!errorlevel!"
        if !neo_running! equ 0 (
            echo       [OK] Neo4j running at localhost:7687
        ) else (
            docker start neo4j-langchain >nul 2>&1
            if !errorlevel! equ 0 (
                echo       [OK] Neo4j container started
            ) else (
                echo       [SKIP] Neo4j not running, KG feature unavailable
            )
        )
    )
)
echo.

echo [4/5] Cleaning Python cache...
if exist "src\__pycache__" (
    rd /s /q "src\__pycache__" 2>nul
    echo       [OK] src\__pycache__ cleaned
) else (
    echo       [OK] No cache to clean
)
echo.

echo [5/5] Starting Streamlit application...
echo.
echo ===========================================
echo   Starting application...
echo   Access at: http://localhost:8501
echo   Press Ctrl+C to stop
echo ===========================================
echo.

streamlit run src/streamlit_app.py --server.port 8501

echo.
echo Exiting...
endlocal
