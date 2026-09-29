@echo off
REM 双击即可：在 WSL2 里按顺序跑完整本地调试流程。
REM 不想用 WSL 时：set GIS_SPU_NATIVE=1 后再运行，走本机 python（无 SPU/PSI 实跑）。
chcp 65001 >nul 2>nul
setlocal
set SCRIPT_DIR=%~dp0
if defined GIS_SPU_NATIVE goto native

set P=%~dp0..
set TMPF=%TEMP%\gspu_wslpath.txt
wsl.exe wslpath -a "%P%" > "%TMPF%" 2>nul
if errorlevel 1 goto native
set WSLDIR=
set /p WSLDIR=<"%TMPF%"
del /q "%TMPF%" >nul 2>nul
if not defined WSLDIR goto native

echo [debug_local] WSL2 路径: %WSLDIR%
wsl.exe -d Ubuntu -u root -- bash "%WSLDIR%/scripts/debug_local.sh"
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo. & echo [debug_local] 脚本返回码 %RC%（向上看输出定位问题）
goto end

:native
echo [debug_local] 本机 python 模式：无 SPU/PSI 实跑，能力核查会列出阻断项
pushd "%SCRIPT_DIR%.."
set PYTHONIOENCODING=utf-8
python -m geosecure.cli ops
python -m geosecure.cli check
if exist tests\test_ir.py python -m pytest tests/ -q --no-header -p no:cacheprovider
python -m geosecure.cli build examples/route_conflict.py
popd

:end
echo.
pause