@echo off
setlocal enabledelayedexpansion

:: =========================================================
:: GitHub Repository Auto Creator + Project Pusher
:: Requirements:
:: - Git installed
:: - GitHub CLI (gh) installed and authenticated
:: =========================================================

:: Check if Git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not added to PATH.
    pause
    exit /b
)

:: Check if GitHub CLI is installed
where gh >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] GitHub CLI is not installed or not added to PATH.
    echo Install it from:
    echo https://cli.github.com/
    pause
    exit /b
)

:: Check GitHub authentication
gh auth status >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] You are not logged into GitHub CLI.
    echo Run:
    echo gh auth login
    pause
    exit /b
)

:: Get current folder name as default repo name
for %%I in ("%cd%") do set REPO_NAME=%%~nxI

echo ========================================
echo GitHub Repository Automation Script
echo ========================================
echo Current Project Folder: %REPO_NAME%
echo.

set /p CUSTOM_REPO=Enter repository name (Leave blank to use "%REPO_NAME%"): 

if not "%CUSTOM_REPO%"=="" (
    set REPO_NAME=%CUSTOM_REPO%
)

echo.
echo Using Repository Name: %REPO_NAME%
echo.

:: Check if inside a Git repository
if not exist ".git" (
    echo [INFO] Initializing Git repository...
    git init
)

:: Check if GitHub repository exists
echo [INFO] Checking if GitHub repository exists...

gh repo view "%REPO_NAME%" >nul 2>nul

if %errorlevel% neq 0 (
    echo [INFO] Repository does not exist.
    echo [INFO] Creating GitHub repository...

    gh repo create "%REPO_NAME%" --public --source=. --remote=origin --push

    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create repository.
        pause
        exit /b
    )

    echo.
    echo [SUCCESS] Repository created and project pushed.
    pause
    exit /b
)

echo [INFO] Repository already exists.

:: Check if remote origin exists
git remote get-url origin >nul 2>nul

if %errorlevel% neq 0 (
    echo [INFO] Adding remote origin...
    gh repo view "%REPO_NAME%" --json url -q .url > repo_url.txt

    set /p REPO_URL=<repo_url.txt
    del repo_url.txt

    git remote add origin !REPO_URL!
)

:: Stage files
echo [INFO] Staging files...
git add .

:: Commit changes
set /p COMMIT_MSG=Enter commit message: 

if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=Updated project files
)

git commit -m "%COMMIT_MSG%" 2>nul

:: Detect current branch
for /f %%i in ('git branch --show-current') do set BRANCH=%%i

if "%BRANCH%"=="" (
    set BRANCH=main
    git branch -M main
)

:: Push changes
echo [INFO] Pushing to GitHub...
git push -u origin %BRANCH%

echo.
echo ========================================
echo [SUCCESS] Project synced with GitHub.
echo ========================================

pause