@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   FinRAG-Advisor First-Time Setup
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo [1/4] Checking Docker Desktop...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo       [ERROR] Docker is not running. Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)
echo       [OK] Docker is running
echo.

echo [2/4] Cleaning up old containers (if any)...
docker stop es-langchain >nul 2>&1
docker rm es-langchain >nul 2>&1
docker stop neo4j-langchain >nul 2>&1
docker rm neo4j-langchain >nul 2>&1
echo       [OK] Old containers cleaned
echo.

echo [3/4] Creating Elasticsearch container...
docker run -d ^
    --name es-langchain ^
    -p 9200:9200 -p 9300:9300 ^
    -e discovery.type=single-node ^
    -e xpack.security.enabled=false ^
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" ^
    -v es-data:/usr/share/elasticsearch/data ^
    docker.elastic.co/elasticsearch/elasticsearch:8.11.0
if %errorlevel% equ 0 (
    echo       [OK] Elasticsearch container created
) else (
    echo       [ERROR] Elasticsearch container creation failed
)
echo.

echo [4/4] Creating Neo4j container (optional)...
docker run -d ^
    --name neo4j-langchain ^
    -p 7474:7474 -p 7687:7687 ^
    -e NEO4J_AUTH=neo4j/password ^
    -e NEO4J_PLUGINS='["apoc"]' ^
    -v neo4j-data:/data ^
    neo4j:5.15-community
if %errorlevel% equ 0 (
    echo       [OK] Neo4j container created
) else (
    echo       [ERROR] Neo4j container creation failed
)
echo.

echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Notes:
echo   - Elasticsearch takes 1-2 minutes to initialize on first start
echo   - Check status with: docker logs es-langchain --tail 20
echo   - For daily use, run start.bat
echo.
pause
endlocal
