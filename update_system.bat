@echo off
echo ========================================
echo   RAG System Update ^& Sync Automation
echo ========================================

echo 1. Running Indexer...
git config core.longpaths true
python -m src.indexer

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Indexing failed. Aborting sync.
    pause
    exit /b %ERRORLEVEL%
)

echo 2. Staging changes...
git add .

echo 3. Committing updates...
git commit -m "Update RAG FAISS index: %DATE% %TIME%"

echo 4. Pushing to GitHub...
:: Check if remote is configured
git remote -v >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] No GitHub remote found. 
    echo Please run: git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    echo Then run this script again.
    pause
    exit /b 1
)

git push origin main


echo ========================================
echo   Update Complete! Online version is synced.
echo ========================================
pause
