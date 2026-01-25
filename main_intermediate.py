"""
中级实验主脚本
实验二的中级要求：对比学习与特征重要性评估

实验流程：
1. 使用SimCLR进行对比学习预训练
2. 在预训练编码器上添加自注意力层
3. 使用注意力机制评估特征重要性（Top 50基因）
4. 使用集成梯度方法评估特征重要性（Top 50基因）
5. 比较两种方法的结果重叠率
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入模块
from src.data_loader import SingleCellDataLoader
from src.preprocessing import SingleCellPreprocessor
from models.pretrain import SimCLR, SingleCellDataset, train_simclr
from models.feature_importance import (
    EncoderWithAttention,
    FeatureImportanceEvaluator,
    train_encoder_with_attention
)
from utils.visualization import Visualizer
from config import Config


class IntermediateExperiment:
    """中级实验类"""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path('outputs_intermediate')
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        # 初始化模块
        self.data_loader = SingleCellDataLoader(self.config.data_dir if hasattr(self.config, 'data_dir') else 'data')
        self.preprocessor = SingleCellPreprocessor(
            min_genes=self.config.min_genes if hasattr(self.config, 'min_genes') else 200,
            min_cells=self.config.min_cells if hasattr(self.config, 'min_cells') else 3,
            n_top_genes=self.config.n_top_genes if hasattr(self.config, 'n_top_genes') else 2000,
            n_pca_components=self.config.n_pca_components if hasattr(self.config, 'n_pca_components') else 50
        )
        self.visualizer = Visualizer(self.output_dir / 'plots')
        
        # 结果存储
        self.results = {}
    
    def step1_simclr_pretrain(self):
        """
        步骤1：SimCLR对比学习预训练
        任务2：SimCLR对比学习实现
        """
        print("\n" + "="*60)
        print("步骤1: SimCLR对比学习预训练")
        print("="*60)
        
        # 加载SC-2数据
        datasets = self.data_loader.load_all_data()
        
        if 'SC-2_dense' not in datasets:
            raise ValueError("未找到SC-2_dense数据集")
        
        adata = datasets['SC-2_dense']
        
        # 预处理
        print("\n预处理数据...")
        adata_processed = self.preprocessor.preprocess(adata)
        
        # 获取PCA特征
        X_pca = adata_processed.obsm['X_pca']
        input_dim = X_pca.shape[1]
        
        print(f"输入维度: {input_dim}")
        print(f"样本数量: {len(X_pca)}")
        
        # 保存预处理后的数据
        self.X_pca_full = X_pca
        self.adata_processed = adata_processed
        self.input_dim = input_dim
        
        # 创建数据集和数据加载器（带数据增强）
        dataset = SingleCellDataset(X_pca, augment=True)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 创建SimCLR模型
        model = SimCLR(
            input_dim=input_dim,
            hidden_dims=[256, 128],
            latent_dim=64,
            projection_dim=64
        ).to(self.device)
        
        # 训练
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("\n开始训练SimCLR...")
        history = train_simclr(
            model, dataloader, optimizer, self.device, 
            epochs=100, temperature=0.5
        )
        
        # 保存模型
        model_path = self.output_dir / 'models' / 'pretrained_simclr.pth'
        torch.save(model.state_dict(), model_path)
        print(f"\nSimCLR模型已保存: {model_path}")
        
        # 保存训练历史
        history_path = self.output_dir / 'simclr_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # 绘制训练曲线
        self._plot_loss(history, 'SimCLR Contrastive Learning')
        
        self.simclr_model = model
        self.simclr_encoder = model.encoder
        
        return model
    
    def step2_attention_training(self):
        """
        步骤2：训练带注意力的编码器
        任务3的第1部分：添加自注意力层
        """
        print("\n" + "="*60)
        print("步骤2: 训练带自注意力的编码器")
        print("="*60)
        
        # 创建带注意力的编码器
        attention_model = EncoderWithAttention(
            pretrained_encoder=self.simclr_encoder,
            input_dim=self.input_dim,
            attention_hidden_dim=128
        ).to(self.device)
        
        print(f"\n模型结构:")
        print(f"  编码器参数（冻结）: {sum(p.numel() for p in attention_model.encoder.parameters()):,}")
        print(f"  注意力层参数（可训练）: {sum(p.numel() for p in attention_model.attention.parameters()):,}")
        
        # 创建数据加载器
        dataset = SingleCellDataset(self.X_pca_full, augment=False)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 只训练注意力层
        optimizer = optim.Adam(attention_model.attention.parameters(), lr=0.001)
        
        print("\n训练注意力层...")
        history = train_encoder_with_attention(
            self.simclr_encoder, attention_model, dataloader,
            optimizer, self.device, epochs=50
        )
        
        # 保存模型
        model_path = self.output_dir / 'models' / 'encoder_with_attention.pth'
        torch.save(attention_model.state_dict(), model_path)
        print(f"\n带注意力的编码器已保存: {model_path}")
        
        self.attention_model = attention_model
        
        return attention_model
    
    def step3_feature_importance_evaluation(self):
        """
        步骤3：特征重要性评估
        任务3：完整的特征重要性评估
        """
        print("\n" + "="*60)
        print("步骤3: 特征重要性评估")
        print("="*60)
        
        # 创建数据加载器
        dataset = SingleCellDataset(self.X_pca_full, augment=False)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 创建评估器
        evaluator = FeatureImportanceEvaluator(
            model=self.attention_model,
            device=self.device
        )
        
        # 综合评估
        results = evaluator.comprehensive_evaluation(
            dataloader=dataloader,
            top_k=50,
            save_dir=self.output_dir / 'plots'
        )
        
        # 保存结果
        results_to_save = {
            'attention_top_features': results['attention_top_features'].tolist(),
            'gradient_top_features': results['gradient_top_features'].tolist(),
            'overlap_features': [int(x) for x in results['overlap_features']],
            'overlap_rate': float(results['overlap_rate']),
            'correlation': float(results['correlation'])
        }
        
        results_path = self.output_dir / 'feature_importance_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        
        print(f"\n特征重要性结果已保存: {results_path}")
        
        self.feature_importance_results = results
        
        return results
    
    def _plot_loss(self, history, title):
        """绘制损失曲线"""
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        
        # 设置中文字体
        try:
            font_path = 'C:/Windows/Fonts/msyh.ttc'
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
        except:
            pass
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, label='Contrastive Loss')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots' / 'simclr_training_loss.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: {save_path}")
    
    def run_full_experiment(self):
        """
        运行完整的中级实验
        """
        print("\n" + "="*70)
        print(" " * 10 + "中级实验：对比学习与特征重要性评估")
        print("="*70)
        
        # 步骤1：SimCLR预训练
        self.step1_simclr_pretrain()
        
        # 步骤2：训练注意力层
        self.step2_attention_training()
        
        # 步骤3：特征重要性评估
        self.step3_feature_importance_evaluation()
        
        print("\n" + "="*70)
        print(" " * 20 + "实验完成！")
        print(f" " * 15 + f"所有结果保存在: {self.output_dir}")
        print("="*70)


def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    # 创建实验实例
    experiment = IntermediateExperiment(config)
    
    # 运行完整实验
    experiment.run_full_experiment()


if __name__ == '__main__':
    main()
