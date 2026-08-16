# SENTINEL-Vision Full Curriculum Pipeline

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Starting SENTINEL-Vision Curriculum Run " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host ">>> Stage A: Base Salience Training" -ForegroundColor Green
python src\training\train_stageA.py
if ($LASTEXITCODE -ne 0) { Write-Error "Stage A failed!"; exit $LASTEXITCODE }

Write-Host ">>> Stage B: Localization Training" -ForegroundColor Green
python src\training\train_stageB.py
if ($LASTEXITCODE -ne 0) { Write-Error "Stage B failed!"; exit $LASTEXITCODE }

Write-Host ">>> Stage C: Contextual Harm Training" -ForegroundColor Green
python src\training\train_stageC.py
if ($LASTEXITCODE -ne 0) { Write-Error "Stage C failed!"; exit $LASTEXITCODE }

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Curriculum Complete! Final model saved. " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
