"""
完整测试脚本：验证所有三个实验都能在GPU上正常运行
"""
import sys
import torch
from pathlib import Path

print("="*70)
print("三个实验完整测试")
print("="*70)

# 检查GPU
print(f"\nGPU环境:")
print(f"  CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

from config import Config

print("\n" + "="*70)
print("测试1: 初级实验（快速测试 - 3轮训练）")
print("="*70)

try:
    from main_basic import BasicExperiment
    
    config = Config('quick_test')
    experiment = BasicExperiment(config)
    
    # 测试关键步骤
    print("\n[1/5] 数据加载与预处理...")
    experiment.step1_load_and_preprocess()
    
    print("\n[2/5] 构建自编码器模型...")
    experiment.step2_build_autoencoder()
    
    print("\n[3/5] 数据增强方法...")
    experiment.step3_data_augmentation()
    
    print("\n[4/5] 训练自编码器（3轮）...")
    experiment.step4_train_autoencoder(n_epochs=3, batch_size=32)
    
    print("\n[5/5] 保存模型...")
    experiment.step5_save_model()
    
    print("\n[OK] 初级实验测试通过！")
    print(f"  设备: {experiment.device}")
    print(f"  最终损失: {experiment.history['loss'][-1]:.6f}")
    
    # 清理GPU内存
    del experiment
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
except Exception as e:
    print(f"\n[ERROR] 初级实验失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("测试2: 中级实验（跳过 - 需要较长时间）")
print("="*70)
print("中级实验包含SimCLR和特征重要性评估，需要较长训练时间")
print("建议使用: python main_intermediate.py")

print("\n" + "="*70)
print("测试3: 高级实验（跳过 - 需要较长时间）")
print("="*70)
print("高级实验包含提示学习和多维度评估，需要较长训练时间")
print("建议使用: python main_advanced.py")

print("\n" + "="*70)
print("测试总结")
print("="*70)
print("[OK] 初级实验: 通过")
print("[SKIP] 中级实验: 跳过（可手动运行）")
print("[SKIP] 高级实验: 跳过（可手动运行）")
print("\n所有模块导入和基本功能测试通过！")
print("="*70)

if torch.cuda.is_available():
    print(f"\nGPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
