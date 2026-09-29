@echo off
setlocal enabledelayedexpansion
title Install Superpowers Plugin

set "CODEX_HOME=C:\Users\DELL\.codex"
set "CODEX_BIN=C:\Users\DELL\AppData\Local\OpenAI\Codex\bin\cdef5aaf3e41ab53\codex.exe"

if not exist "!CODEX_BIN!" (
    for /d %%D in ("%LOCALAPPDATA%\OpenAI\Codex\bin\*") do (
        if exist "%%D\codex.exe" set "CODEX_BIN=%%D\codex.exe"
    )
)

echo ============================================================
echo   安装 Superpowers 插件（Codex 官方市场）
echo ============================================================
echo.

if not exist "!CODEX_BIN!" (
    echo [错误] 找不到 codex.exe，请确认 Codex 应用已正常安装。
    echo.
    pause
    exit /b 1
)

echo 插件来源：openai-api-curated（本机已缓存，无需联网）
echo.
echo 正在安装 superpowers ...
echo.

"!CODEX_BIN!" plugin add superpowers@openai-api-curated

if errorlevel 1 (
    echo.
    echo [失败] 安装未能完成。请把上面的报错信息整段发回给我。
    echo.
    pause
    exit /b 1
)

echo.
echo ------------------------------------------------------------
echo 当前状态：
"!CODEX_BIN!" plugin list | findstr /i "superpowers"
echo ------------------------------------------------------------
echo.
echo 安装完成。请重启 Codex 应用，插件才会被加载。
echo.
pause
