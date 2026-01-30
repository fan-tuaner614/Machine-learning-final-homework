"""
实验配置文件
可以修改此文件来调整实验参数
"""

# 实验配置字典
EXPERIMENT_CONFIG = {
    # ==================== 数据配置 ====================
    'data_dir': 'data',                    # 数据目录
    'output_dir': 'outputs',               # 输出目录
    
    # ==================== 预处理配置 ====================
    'min_genes': 200,                      # 每个细胞最少基因数
    'min_cells': 3,                        # 每个基因最少细胞数
    'n_top_genes': 2000,                   # 高变基因数量
    'n_pca_components': 50,                # PCA主成分数量
    'use_pca_components': 50,              # 实际使用的主成分数量
    
    # ==================== 数据增强配置 ====================
    'use_smote': True,                     # 是否使用SMOTE处理类别不平衡
    'val_split': 0.3,                      # 验证集比例
    
    # ==================== 模型配置 ====================
    'model_type': 'attention',             # 模型类型: 'basic', 'attention', 'residual'
    'hidden_dims': [512, 256, 128],        # 隐藏层维度列表
    'dropout_rate': 0.4,                   # Dropout比率 (0.0-1.0)
    
    # ==================== 训练配置 ====================
    'n_epochs': 100,                       # 最大训练轮数
    'batch_size': 64,                      # 批次大小
    'learning_rate': 0.001,                # 初始学习率
    'weight_decay': 1e-4,                  # 权重衰减（L2正则化）
    
    # ==================== 优化器配置 ====================
    'optimizer': 'adamw',                  # 优化器: 'adam', 'adamw', 'sgd'
    'scheduler': 'reduce_lr',              # 学习率调度器: 'reduce_lr', 'cosine', 'onecycle', None
    'loss_type': 'focal',                  # 损失函数: 'ce', 'focal', 'label_smoothing'
    'use_class_weights': True,             # 是否使用类别权重
    
    # ==================== 早停配置 ====================
    'early_stopping_patience': 15,         # 早停容忍轮数
    
    # ==================== GPU配置 ====================
    'use_cuda': True,                      # 是否使用CUDA（如果可用）
    'device': 'cuda',                      # 设备: 'cuda' 或 'cpu'
    
    # ==================== 高级实验配置 ====================
    # 自监督预训练配置
    'pretrain_model': 'autoencoder',       # 预训练模型: 'autoencoder', 'simclr'
    'pretrain_epochs': 100,                # 预训练轮数
    'pretrain_batch_size': 64,             # 预训练批次大小
    'pretrain_lr': 0.001,                  # 预训练学习率
    'latent_dim': 64,                      # 潜在特征维度
    'encoder_hidden_dims': [256, 128],     # 编码器隐藏层
    
    # SimCLR配置
    'simclr_temperature': 0.5,             # SimCLR温度参数
    'simclr_projection_dim': 64,           # SimCLR投影维度
    
    # 提示学习配置
    'prompt_length': 10,                   # 提示向量数量
    'prompt_dim': None,                    # 提示向量维度（None则使用latent_dim）
    'prompt_dropout': 0.2,                 # 提示学习Dropout
    'prompt_epochs': 100,                  # 提示学习训练轮数
    'prompt_patience': 20,                 # 提示学习早停耐心
    'prompt_lr': 0.001,                    # 提示学习学习率
    
    # 评估配置
    'eval_n_runs': 3,                      # 特征稳定性评估运行次数
    'eval_cv_folds': 5,                    # 交叉验证折数
}

# 不同场景的预设配置
PRESET_CONFIGS = {
    # 快速测试配置（小规模，快速验证）
    'quick_test': {
        'n_top_genes': 500,
        'n_pca_components': 20,
        'use_pca_components': 20,
        'model_type': 'basic',
        'hidden_dims': [128, 64],
        'n_epochs': 10,
        'batch_size': 32,
        'early_stopping_patience': 3,
    },
    
    # 标准训练配置（平衡性能和时间）
    'standard': {
        'n_top_genes': 2000,
        'n_pca_components': 50,
        'use_pca_components': 50,
        'model_type': 'attention',
        'hidden_dims': [512, 256, 128],
        'n_epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
        'early_stopping_patience': 15,
    },
    
    # 高性能配置（追求最佳效果）
    'high_performance': {
        'n_top_genes': 3000,
        'n_pca_components': 100,
        'use_pca_components': 100,
        'model_type': 'residual',
        'hidden_dims': [1024, 512, 256, 128],
        'dropout_rate': 0.3,
        'n_epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.0005,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'loss_type': 'focal',
        'early_stopping_patience': 30,
    },
    
    # CPU训练配置（无GPU时使用）
    'cpu_mode': {
        'model_type': 'basic',
        'hidden_dims': [256, 128],
        'n_epochs': 50,
        'batch_size': 32,
        'use_cuda': False,
        'device': 'cpu',
    },
}


def get_config(preset: str = None):
    """
    获取配置
    
    Args:
        preset: 预设配置名称，如果为None则使用默认配置
        
    Returns:
        配置字典
    """
    config = EXPERIMENT_CONFIG.copy()
    
    if preset and preset in PRESET_CONFIGS:
        config.update(PRESET_CONFIGS[preset])
        print(f"[OK] 使用预设配置: {preset}")
    
    return config


def print_config(config: dict):
    """打印配置信息"""
    print("\n" + "="*60)
    print("实验配置")
    print("="*60)
    
    categories = {
        '数据配置': ['data_dir', 'output_dir'],
        '预处理配置': ['min_genes', 'min_cells', 'n_top_genes', 'n_pca_components'],
        '模型配置': ['model_type', 'hidden_dims', 'dropout_rate'],
        '训练配置': ['n_epochs', 'batch_size', 'learning_rate', 'weight_decay'],
        '优化配置': ['optimizer', 'scheduler', 'loss_type', 'use_class_weights'],
        '其他配置': ['use_smote', 'val_split', 'early_stopping_patience', 'device'],
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            if key in config:
                value = config[key]
                if isinstance(value, list):
                    value = str(value)
                print(f"  {key:30s}: {value}")
    
    print("\n" + "="*60)


# 配置验证
def validate_config(config: dict) -> bool:
    """
    验证配置合法性
    
    Args:
        config: 配置字典
        
    Returns:
        是否合法
    """
    errors = []
    
    # 检查必需的键
    required_keys = [
        'data_dir', 'output_dir', 'model_type', 'n_epochs',
        'batch_size', 'learning_rate'
    ]
    
    for key in required_keys:
        if key not in config:
            errors.append(f"缺少必需的配置项: {key}")
    
    # 检查模型类型
    if config.get('model_type') not in ['basic', 'attention', 'residual']:
        errors.append(f"无效的模型类型: {config.get('model_type')}")
    
    # 检查数值范围
    if config.get('dropout_rate', 0) < 0 or config.get('dropout_rate', 0) > 1:
        errors.append(f"dropout_rate必须在0-1之间: {config.get('dropout_rate')}")
    
    if config.get('val_split', 0) <= 0 or config.get('val_split', 0) >= 1:
        errors.append(f"val_split必须在0-1之间: {config.get('val_split')}")
    
    if config.get('batch_size', 0) <= 0:
        errors.append(f"batch_size必须大于0: {config.get('batch_size')}")
    
    if config.get('learning_rate', 0) <= 0:
        errors.append(f"learning_rate必须大于0: {config.get('learning_rate')}")
    
    # 打印错误
    if errors:
        print("\n❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("\n[OK] 配置验证通过")
    return True


class Config:
    """配置类，将字典配置转换为类属性"""
    
    def __init__(self, preset=None):
        """
        初始化配置
        
        Args:
            preset: 预设配置名称
        """
        config_dict = get_config(preset)
        
        # 将所有配置项设置为类属性
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """转换为字典"""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def __repr__(self):
        """字符串表示"""
        return f"Config({self.to_dict()})"
    
    def __str__(self):
        """打印配置"""
        lines = ["Configuration:"]
        for key, value in sorted(self.to_dict().items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


if __name__ == '__main__':
    # 测试配置
    print("可用的预设配置:")
    for preset_name in PRESET_CONFIGS.keys():
        print(f"  - {preset_name}")
    
    print("\n" + "="*60)
    print("默认配置:")
    config = get_config()
    print_config(config)
    validate_config(config)
    
    print("\n" + "="*60)
    print("快速测试配置:")
    config_test = get_config('quick_test')
    print_config(config_test)
    validate_config(config_test)
