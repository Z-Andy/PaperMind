@echo off
REM ========================================
REM  多Agent协作研究平台 - 一键启动脚本
REM ========================================

title 多Agent协作研究平台

echo.
echo ========================================
echo   多Agent协作研究平台 v1.0
echo ========================================
echo.

REM 检查 Python 环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查 .env 文件
if not exist ".env" (
    echo [提示] 未找到 .env 文件，正在从 .env.example 创建...
    copy .env.example .env >nul
    echo [提示] 请编辑 .env 文件，填入你的 API Key 等信息
    echo.
)

REM 安装依赖
echo [1/3] 检查依赖...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo [2/3] 启动后端 API 服务 (端口 8000)...
start "API服务" cmd /c "python -m src.api.main"

REM 等待后端启动
echo 等待后端启动...
timeout /t 3 >nul

echo [3/3] 启动前端界面 (端口 8501)...
streamlit run src/ui/app.py --server.port 8501

echo.
echo 服务已关闭。
pause
