# 实验运行脚本 - 使用machine-learning环境
# PowerShell脚本

$PYTHON = "D:\ANACONDA\envs\machine-learning\python.exe"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  实验二：单细胞RNA测序数据深度学习项目" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 检查环境
Write-Host "[检查] Python环境: machine-learning" -ForegroundColor Yellow
& $PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
Write-Host ""

# 菜单
Write-Host "请选择要运行的实验:" -ForegroundColor Green
Write-Host "  1. 初级实验（30分）- 自编码器 + 数据增强（预计10-15分钟）"
Write-Host "  2. 中级实验（40分）- SimCLR + 特征重要性（预计20-30分钟）"
Write-Host "  3. 高级实验（30分）- 提示学习 + 评估（预计30-60分钟）"
Write-Host "  4. 运行快速测试（测试所有模块，3轮训练）"
Write-Host "  5. 退出"
Write-Host ""

$choice = Read-Host "请输入选项 (1-5)"

switch ($choice) {
    "1" {
        Write-Host "`n[运行] 初级实验..." -ForegroundColor Green
        & $PYTHON main_basic.py
    }
    "2" {
        Write-Host "`n[运行] 中级实验..." -ForegroundColor Green
        & $PYTHON main_intermediate.py
    }
    "3" {
        Write-Host "`n[运行] 高级实验..." -ForegroundColor Green
        & $PYTHON main_advanced.py
    }
    "4" {
        Write-Host "`n[运行] 快速测试..." -ForegroundColor Green
        & $PYTHON test_all_experiments.py
    }
    "5" {
        Write-Host "`n退出。" -ForegroundColor Yellow
        exit
    }
    default {
        Write-Host "`n无效选项！" -ForegroundColor Red
        exit
    }
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "  实验完成！" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
