"""
快速测试初级实验（basic）
"""
import sys
sys.path.insert(0, 'e:\\桌面\\机器学习\\final')

from main_basic import BasicExperiment
from config import Config

print("="*70)
print("初级实验快速测试")
print("="*70)

# 创建快速测试配置
config = Config('quick_test')
config.pretrain_epochs = 3  # 只训练3轮用于测试
config.batch_size = 32

# 创建实验实例
experiment = BasicExperiment(config)

try:
    print("\n[测试] 步骤1: 数据加载与预处理")
    experiment.step1_load_and_preprocess()
    
    print("\n[测试] 步骤2: 构建自编码器模型")
    experiment.step2_build_autoencoder()
    
    print("\n[测试] 步骤3: 数据增强方法")
    experiment.step3_data_augmentation()
    
    print("\n[测试] 步骤4: 训练自编码器（3轮）")
    experiment.step4_train_autoencoder(n_epochs=3, batch_size=32, learning_rate=0.001)
    
    print("\n[测试] 步骤5: 保存模型")
    experiment.step5_save_model()
    
    print("\n" + "="*70)
    print("[OK] 初级实验快速测试通过！")
    print("="*70)
    
except Exception as e:
    print(f"\n[ERROR] 测试失败: {e}")
    import traceback
    traceback.print_exc()
