@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: ========================================
::   FinRAG-Advisor 一键启动脚本
::   自动检测并启动所有依赖服务
:: ========================================

:: 获取项目根目录
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo.
echo ╔══════════════════════════════════════════╗
echo ║     FinRAG-Advisor 一键启动脚本           ║
echo ╚══════════════════════════════════════════╝
echo.

:: ---- 0. 检查 .env 配置 ----
echo [0/5] 检查环境配置...
if not exist ".env" (
    echo       [!] 未找到 .env 文件，使用默认配置
) else (
    echo       [OK] .env 配置文件已加载
)
echo.

:: ---- 1. 检查并启动 Ollama ----
echo [1/5] 检查 Ollama 服务...
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | findstr /i "ollama.exe" >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Ollama 已运行
) else (
    echo       [*] Ollama 未运行，正在启动...
    start "" /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
    :: 再次检查
    tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | findstr /i "ollama.exe" >nul 2>&1
    if %errorlevel% equ 0 (
        echo       [OK] Ollama 启动成功
    ) else (
        echo       [!] Ollama 启动失败，请确保已安装 Ollama
        echo       [!] 下载地址: https://ollama.com/download
        echo       [!] 也可以在另一个终端手动运行: ollama serve
    )
)

:: 检查所需模型
echo       [*] 检查 Ollama 模型...
ollama list 2>nul | findstr /i "my-bge-m3" >nul 2>&1
if %errorlevel% neq 0 (
    echo       [!] 未找到 my-bge-m3 模型，请运行: ollama pull my-bge-m3
) else (
    echo       [OK] my-bge-m3 模型已就绪
)
ollama list 2>nul | findstr /i "my-qwen25" >nul 2>&1
if %errorlevel% neq 0 (
    echo       [!] 未找到 my-qwen25 模型，请运行: ollama pull my-qwen25
) else (
    echo       [OK] my-qwen25 模型已就绪
)
echo.

:: ---- 2. 检查并启动 Elasticsearch ----
echo [2/5] 检查 Elasticsearch 服务...
:: 先检查端口是否已经在监听
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:9200' -TimeoutSec 3 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Elasticsearch 已运行 (localhost:9200)
) else (
    :: 尝试用 Docker 启动
    docker ps -a --filter "name=es-langchain" --format "{{.Names}}" 2>nul | findstr "es-langchain" >nul 2>&1
    if %errorlevel% equ 0 (
        echo       [*] Elasticsearch 容器已存在，正在启动...
        docker start es-langchain >nul 2>&1
        if %errorlevel% equ 0 (
            echo       [OK] Elasticsearch 容器已启动
        ) else (
            echo       [!] Elasticsearch 容器启动失败
        )
    ) else (
        echo       [*] 未找到 Elasticsearch 容器，正在创建...
        docker run -d ^
            --name es-langchain ^
            -p 9200:9200 ^
            -p 9300:9300 ^
            -e discovery.type=single-node ^
            -e xpack.security.enabled=false ^
            -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" ^
            -v es-data:/usr/share/elasticsearch/data ^
            docker.elastic.co/elasticsearch/elasticsearch:8.11.0 >nul 2>&1
        if %errorlevel% equ 0 (
            echo       [OK] Elasticsearch 容器创建并启动成功
        ) else (
            echo       [!] Elasticsearch 创建失败，请检查 Docker 是否运行
            echo       [!] 也可以手动运行:
            echo           docker run -d --name es-langchain -p 9200:9200 -p 9300:9300 -e discovery.type=single-node -e xpack.security.enabled=false docker.elastic.co/elasticsearch/elasticsearch:8.11.0
        )
    )
    :: 等待 ES 启动
    echo       [*] 等待 Elasticsearch 就绪...
    timeout /t 15 /nobreak >nul
)
echo.

:: ---- 3. 检查并启动 Neo4j (可选) ----
echo [3/5] 检查 Neo4j 服务 (可选)...
powershell -Command "try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('localhost', 7687); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Neo4j 已运行 (localhost:7687)
) else (
    docker ps -a --filter "name=neo4j-langchain" --format "{{.Names}}" 2>nul | findstr "neo4j-langchain" >nul 2>&1
    if %errorlevel% equ 0 (
        echo       [*] Neo4j 容器已存在，正在启动...
        docker start neo4j-langchain >nul 2>&1
        if %errorlevel% equ 0 (
            echo       [OK] Neo4j 容器已启动
        ) else (
            echo       [!] Neo4j 容器启动失败
        )
    ) else (
        echo       [ - ] Neo4j 未配置，知识图谱功能不可用
        echo       [ - ] 如需启用，请运行:
        echo           docker run -d --name neo4j-langchain -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]' neo4j:5.15-community
    )
)
echo.

:: ---- 4. 清理 Python 缓存 ----
echo [4/5] 清理 Python 缓存...
if exist "src\__pycache__" (
    rd /s /q "src\__pycache__" 2>nul
    echo       [OK] 已清理 src\__pycache__
) else (
    echo       [OK] 无需清理
)
echo.

:: ---- 5. 启动 Streamlit ----
echo [5/5] 启动 Streamlit 应用...
echo.
echo ╔══════════════════════════════════════════╗
echo ║  应用启动中，请稍候...                     ║
echo ║  启动成功后访问: http://localhost:8501     ║
echo ║  按 Ctrl+C 停止应用                       ║
echo ╚══════════════════════════════════════════╝
echo.

streamlit run src/streamlit_app.py --server.port 8501

:: ---- 退出清理 (Ctrl+C 后执行) ----
echo.
echo 正在退出...
endlocal
