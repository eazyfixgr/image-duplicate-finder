@echo off
setlocal
title Image Duplicate Finder
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py "image_duplicate_finder.py"
) else (
    python "image_duplicate_finder.py"
)
if errorlevel 1 pause
