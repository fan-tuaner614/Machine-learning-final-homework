"""
快速测试脚本：验证三个实验的基本功能
"""
import sys
import torch
import numpy as np
from pathlib import Path

print("="*70)
print("实验环境检查")
print("="*70)

# 1. 检查PyTorch
print(f"\n[1] PyTorch环境:")
print(f"  版本: {torch.__version__}")
print(f"  CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 2. 检查必要模块
print(f"\n[2] 模块导入测试:")
try:
    from config import Config
    print("  ✓ Config")
except Exception as e:
    print(f"  ✗ Config: {e}")
    
try:
    from src.data_loader import SingleCellDataLoader
    print("  ✓ SingleCellDataLoader")
except Exception as e:
    print(f"  ✗ SingleCellDataLoader: {e}")

try:
    from src.preprocessing import SingleCellPreprocessor
    print("  ✓ SingleCellPreprocessor")
except Exception as e:
    print(f"  ✗ SingleCellPreprocessor: {e}")

try:
    from models.pretrain import AutoEncoder, SimCLR
    print("  ✓ AutoEncoder, SimCLR")
except Exception as e:
    print(f"  ✗ AutoEncoder, SimCLR: {e}")

try:
    from models.prompt_learning import PromptLearningModel
    print("  ✓ PromptLearningModel")
except Exception as e:
    print(f"  ✗ PromptLearningModel: {e}")

try:
    from models.feature_importance import FeatureImportanceEvaluator
    print("  ✓ FeatureImportanceEvaluator")
except Exception as e:
    print(f"  ✗ FeatureImportanceEvaluator: {e}")

# 3. 检查数据文件
print(f"\n[3] 数据文件检查:")
data_dir = Path('data')
required_files = [
    'SC-1_dense.h5ad',
    'SC-2_dense.h5ad',
    '10_percent.npy',
    '30_percent.npy',
    '50_percent.npy',
    'all.npy'
]

for file_name in required_files:
    file_path = data_dir / file_name
    if file_path.exists():
        size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"  ✓ {file_name} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {file_name} (缺失)")

# 4. 测试实验脚本导入
print(f"\n[4] 实验脚本导入测试:")
try:
    from main_basic import BasicExperiment
    print("  ✓ main_basic.py")
except Exception as e:
    print(f"  ✗ main_basic.py: {e}")

try:
    from main_intermediate import IntermediateExperiment
    print("  ✓ main_intermediate.py")
except Exception as e:
    print(f"  ✗ main_intermediate.py: {e}")

try:
    from main_advanced import AdvancedExperiment
    print("  ✓ main_advanced.py")
except Exception as e:
    print(f"  ✗ main_advanced.py: {e}")

# 5. 简单模型测试
print(f"\n[5] 模型初始化测试:")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  使用设备: {device}")

try:
    # 测试AutoEncoder
    input_dim = 50
    model = AutoEncoder(input_dim=input_dim, hidden_dims=[128, 64], latent_dim=32).to(device)
    x = torch.randn(16, input_dim).to(device)
    x_recon, z = model(x)
    print(f"  ✓ AutoEncoder: 输入{x.shape} -> 重构{x_recon.shape}, 潜在{z.shape}")
    del model, x, x_recon, z
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    print(f"  ✗ AutoEncoder: {e}")

try:
    # 测试SimCLR
    from models.pretrain import SimCLR
    model = SimCLR(input_dim=50, hidden_dims=[128, 64], latent_dim=32, projection_dim=64).to(device)
    x = torch.randn(16, 50).to(device)
    z = model.encoder(x)
    proj = model.projection_head(z)
    print(f"  ✓ SimCLR: 输入{x.shape} -> 编码{z.shape} -> 投影{proj.shape}")
    del model, x, z, proj
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    print(f"  ✗ SimCLR: {e}")

try:
    # 测试PromptLearningModel
    from models.pretrain import AutoEncoder
    autoencoder = AutoEncoder(input_dim=50, hidden_dims=[128, 64], latent_dim=32).to(device)
    encoder = autoencoder.encoder
    model = PromptLearningModel(
        pretrained_encoder=encoder,
        latent_dim=32,
        num_classes=4,
        prompt_length=5,
        prompt_dim=32
    ).to(device)
    x = torch.randn(16, 50).to(device)
    output = model(x)
    print(f"  ✓ PromptLearningModel: 输入{x.shape} -> 输出{output.shape}")
    del autoencoder, encoder, model, x, output
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    print(f"  ✗ PromptLearningModel: {e}")

print("\n" + "="*70)
print("环境检查完成！")
print("="*70)

# 内存清理
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"\nGPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
